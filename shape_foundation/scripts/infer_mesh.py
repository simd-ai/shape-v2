"""Inference CLI for Shape foundation model.

Load a mesh, preprocess, run the backbone + all heads, and output
a structured JSON recommendation.

Usage:
    python -m shape_foundation.scripts.infer_mesh \
        --checkpoint checkpoints/checkpoint_final.pt \
        --mesh path/to/mesh.stl \
        --output result.json

    python -m shape_foundation.scripts.infer_mesh \
        --checkpoint checkpoints/checkpoint_final.pt \
        --mesh path/to/mesh.obj \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from shape_foundation.configs.default import ShapeConfig, load_config
from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.data.preprocessing import MeshPreprocessor
from shape_foundation.data.sampling import SurfaceSampler
from shape_foundation.preprocessing.mesh_io import load_mesh
from shape_foundation.tasks.topology_reduction import ReductionRecommender
from shape_foundation.tasks.captioning import GeometryCaptioner


def infer(
    checkpoint: str,
    mesh_path: str,
    config_path: str | None = None,
    device: str = "cuda",
    verbose: bool = False,
) -> dict:
    """Run full inference on a single mesh.

    Returns the structured recommendation dict.
    """
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if config_path:
        cfg = load_config(config_path)
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        cfg = ShapeConfig()

    # Build model
    model = GAOTBackbone(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if verbose:
        print(f"Model: {model.get_num_params():,} params")

    # Load and preprocess mesh
    if verbose:
        print(f"Loading mesh: {mesh_path}")
    mesh = load_mesh(mesh_path)
    if verbose:
        print(f"  Vertices: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]}")

    preprocessor = MeshPreprocessor(cfg.input)
    sampler = SurfaceSampler(cfg.input)

    processed = preprocessor(mesh.vertices, mesh.faces, mesh.normals)
    sampled = sampler.sample(
        processed["vertices"], processed["faces"],
        processed["normals"], processed.get("curvature"),
    )

    features = preprocessor.build_features(
        sampled["points"], sampled["normals"], sampled.get("curvature"),
    )

    points = torch.from_numpy(sampled["points"]).unsqueeze(0)
    feat_tensor = torch.from_numpy(features).unsqueeze(0)
    normals = torch.from_numpy(sampled["normals"]).unsqueeze(0)
    curvature = None
    if sampled.get("curvature") is not None:
        crv = sampled["curvature"]
        curvature = torch.from_numpy(crv[:, None] if crv.ndim == 1 else crv).unsqueeze(0)

    if verbose:
        print(f"  Sampled points: {points.shape[1]}")
        print(f"  Feature dim: {feat_tensor.shape[2]}")

    # Run inference
    if verbose:
        print("Running inference...")
    recommender = ReductionRecommender(model, device=device)
    result = recommender.recommend(points, feat_tensor, normals, curvature)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Shape foundation model inference on a mesh")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True, help="Path to mesh file (STL, OBJ, PLY, etc.)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="JSON output path (default: stdout)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = infer(
        checkpoint=args.checkpoint,
        mesh_path=args.mesh,
        config_path=args.config,
        device=args.device,
        verbose=args.verbose,
    )

    # Output
    output_json = json.dumps(result, indent=2, default=_json_default)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Saved to {args.output}")
    else:
        print(output_json)

    # Summary
    hints = result.get("simulation_hints", {})
    reduction = hints.get("recommended_reduction", "none")
    confidence = hints.get("confidence", 0.0)
    reasoning = hints.get("reasoning", [])

    print(f"\n{'=' * 50}")
    print(f"Recommended reduction: {reduction} (confidence: {confidence:.2f})")
    if reasoning:
        print("Reasoning:")
        for r in reasoning:
            print(f"  - {r}")
    print(f"\nDescription: {result.get('description', 'N/A')}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


if __name__ == "__main__":
    main()
