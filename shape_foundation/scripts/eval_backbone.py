"""Evaluation script for Shape foundation model.

Runs all evaluation benchmarks on a trained checkpoint.

Usage:
    python -m shape_foundation.scripts.eval_backbone \
        --checkpoint checkpoints/checkpoint_final.pt \
        --config configs/medium.yaml \
        --output eval_results.json

    python -m shape_foundation.scripts.eval_backbone \
        --checkpoint checkpoints/checkpoint_final.pt \
        --robustness
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from shape_foundation.configs.default import ShapeConfig, load_config
from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.training.eval import Evaluator
from shape_foundation.data.dataset import build_dataloader


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shape foundation model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--robustness", action="store_true", help="Run robustness evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Config from checkpoint or file
    if args.config:
        cfg = load_config(args.config)
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        cfg = ShapeConfig()

    # Build model
    model = GAOTBackbone(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model: {model.get_num_params():,} params")

    # Build evaluator
    evaluator = Evaluator(model, device=args.device)

    # Build dataloader
    dataloader = build_dataloader(
        cfg.data, cfg.input, split=args.split,
        batch_size=args.batch_size, num_workers=4, pin_memory=True,
    )
    print(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches")

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate_all(dataloader)

    # Robustness
    if args.robustness:
        print("\nRunning robustness evaluation...")
        robust_results = evaluator.eval_robustness(dataloader)
        results.update(robust_results)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
