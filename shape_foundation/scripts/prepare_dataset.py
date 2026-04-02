"""Dataset preparation script.

Downloads, preprocesses, and converts mesh datasets into
training-ready .pt files with optional synthetic label generation.

Usage:
    python -m shape_foundation.scripts.prepare_dataset --config configs/medium.yaml
    python -m shape_foundation.scripts.prepare_dataset --source abc --root /data/abc --output data_cache/abc
    python -m shape_foundation.scripts.prepare_dataset --generate-synthetic --n-per-type 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from shape_foundation.configs.default import ShapeConfig, load_config
from shape_foundation.data.preprocessing import MeshPreprocessor
from shape_foundation.data.sampling import SurfaceSampler
from shape_foundation.data.synthetic_labels import SyntheticLabelGenerator
from shape_foundation.preprocessing.mesh_io import load_mesh


def prepare_source(
    source_root: Path,
    output_dir: Path,
    cfg: ShapeConfig,
    max_samples: int = -1,
) -> int:
    """Preprocess mesh files from a source directory into .pt files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = MeshPreprocessor(cfg.input)
    sampler = SurfaceSampler(cfg.input)
    label_gen = SyntheticLabelGenerator(cfg.data.synthetic_labels) if cfg.data.synthetic_labels.enabled else None

    mesh_exts = {".obj", ".stl", ".ply", ".off", ".msh", ".step", ".stp"}
    mesh_files = sorted(
        f for f in source_root.rglob("*")
        if f.suffix.lower() in mesh_exts
    )

    if max_samples > 0:
        mesh_files = mesh_files[:max_samples]

    count = 0
    labels_all = []

    for mesh_path in tqdm(mesh_files, desc=f"Processing {source_root.name}"):
        try:
            mesh = load_mesh(mesh_path)
            if mesh.vertices.shape[0] < 4 or mesh.faces.shape[0] < 1:
                continue

            processed = preprocessor(mesh.vertices, mesh.faces, mesh.normals)
            sampled = sampler.sample(
                processed["vertices"], processed["faces"],
                processed["normals"], processed.get("curvature"),
            )

            features = preprocessor.build_features(
                sampled["points"], sampled["normals"], sampled.get("curvature"),
            )

            data = {
                "points": torch.from_numpy(sampled["points"]),
                "features": torch.from_numpy(features),
                "normals": torch.from_numpy(sampled["normals"]),
            }
            if sampled.get("curvature") is not None:
                crv = sampled["curvature"]
                data["curvature"] = torch.from_numpy(crv[:, None] if crv.ndim == 1 else crv)

            # Synthetic labels
            if label_gen is not None:
                labels = label_gen.generate_all(
                    processed["vertices"], processed["faces"], processed["normals"],
                )
                data["symmetry_label"] = torch.tensor(
                    {"none": 0, "mirror_half": 1, "mirror_quarter": 2, "axisymmetric": 3, "cyclic_sector": 4}
                    .get(labels["symmetry_type"], 0)
                )
                data["reduction_label"] = torch.tensor(
                    {"none": 0, "mirror_half": 1, "mirror_quarter": 2, "axisymmetric_2d": 3, "cyclic_sector": 4, "extrusion_2d": 5}
                    .get(labels["reduction_type"], 0)
                )
                labels["filename"] = mesh_path.stem
                labels_all.append(labels)

            out_path = output_dir / f"{mesh_path.stem}.pt"
            torch.save(data, out_path)
            count += 1

        except Exception as e:
            print(f"  Skipped {mesh_path.name}: {e}")

    # Save labels index
    if labels_all:
        with open(output_dir / "labels.json", "w") as f:
            json.dump(labels_all, f, indent=2, default=_json_default)

    return count


def generate_synthetic_dataset(
    output_dir: Path, cfg: ShapeConfig, n_per_type: int = 100,
) -> int:
    """Generate procedural shapes with known labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = MeshPreprocessor(cfg.input)
    sampler = SurfaceSampler(cfg.input)

    noise = cfg.data.synthetic_labels.perturbation_scale if cfg.data.synthetic_labels.generate_perturbed else 0.0
    shapes = SyntheticLabelGenerator.generate_procedural_shapes(n_per_type, noise)

    labels_all = []
    count = 0

    for i, shape in enumerate(tqdm(shapes, desc="Generating synthetic data")):
        try:
            verts = shape["vertices"]
            faces = shape["faces"]
            processed = preprocessor(verts, faces)
            sampled = sampler.sample(
                processed["vertices"], processed["faces"],
                processed["normals"], processed.get("curvature"),
            )
            features = preprocessor.build_features(
                sampled["points"], sampled["normals"], sampled.get("curvature"),
            )

            data = {
                "points": torch.from_numpy(sampled["points"]),
                "features": torch.from_numpy(features),
                "normals": torch.from_numpy(sampled["normals"]),
            }

            sym_map = {"none": 0, "mirror_half": 1, "mirror_quarter": 2, "axisymmetric": 3, "cyclic_sector": 4}
            red_map = {"none": 0, "mirror_half": 1, "mirror_quarter": 2, "axisymmetric_2d": 3, "cyclic_sector": 4, "extrusion_2d": 5}

            data["symmetry_label"] = torch.tensor(sym_map.get(shape.get("symmetry_type", "none"), 0))
            data["reduction_label"] = torch.tensor(red_map.get(shape.get("reduction_type", "none"), 0))

            if "symmetry_planes" in shape:
                planes = shape["symmetry_planes"]
                data["symmetry_planes"] = torch.tensor(planes, dtype=torch.float32)
            if "symmetry_axes" in shape:
                axes = shape["symmetry_axes"]
                data["symmetry_axes"] = torch.tensor(axes, dtype=torch.float32)

            out_path = output_dir / f"synthetic_{i:06d}.pt"
            torch.save(data, out_path)

            label_entry = {
                "filename": f"synthetic_{i:06d}",
                "symmetry_type": shape.get("symmetry_type", "none"),
                "reduction_type": shape.get("reduction_type", "none"),
            }
            labels_all.append(label_entry)
            count += 1

        except Exception as e:
            print(f"  Skipped shape {i}: {e}")

    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels_all, f, indent=2)

    return count


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Shape foundation model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--source", type=str, default=None, help="Dataset source name (abc, shapenet, etc.)")
    parser.add_argument("--root", type=str, default=None, help="Source root directory")
    parser.add_argument("--output", type=str, default="data_cache", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max samples to process")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic labeled data")
    parser.add_argument("--n-per-type", type=int, default=100, help="Samples per synthetic type")
    parser.add_argument("--no-labels", action="store_true", help="Skip synthetic label generation")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else ShapeConfig()
    if args.no_labels:
        cfg.data.synthetic_labels.enabled = False

    if args.generate_synthetic:
        output = Path(args.output) / "synthetic"
        count = generate_synthetic_dataset(output, cfg, args.n_per_type)
        print(f"Generated {count} synthetic samples in {output}")
    elif args.source and args.root:
        output = Path(args.output) / args.source
        count = prepare_source(Path(args.root), output, cfg, args.max_samples)
        print(f"Processed {count} samples from {args.source} to {output}")
    else:
        # process all configured sources
        for src in cfg.data.sources:
            root = Path(src.root) if src.root else Path(cfg.data.cache_dir) / src.name
            if not root.exists():
                print(f"Skipping {src.name}: {root} not found")
                continue
            output = Path(cfg.data.cache_dir) / src.name / "processed"
            count = prepare_source(root, output, cfg, src.max_samples)
            print(f"Processed {count} samples from {src.name}")


if __name__ == "__main__":
    main()
