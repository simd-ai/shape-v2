"""Evaluation entry point for a finished Shape foundation checkpoint.

Runs the existing Evaluator on a chosen split, writes a metrics JSON, and
prints a readable console summary. Designed to be runnable with just a
single `--checkpoint` argument; everything else has sensible defaults.

Usage:
    # Minimal: load a final checkpoint and write results next to it
    python -m shape_foundation.scripts.eval_backbone \
        --checkpoint /data/shape-v2/checkpoints/checkpoint_final.pt

    # Pick a different split and output path, add robustness benchmarks
    python -m shape_foundation.scripts.eval_backbone \
        --checkpoint /data/shape-v2/checkpoints/checkpoint_final.pt \
        --split test --output results/eval.json --robustness
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import torch

from shape_foundation.configs.default import ShapeConfig, load_config
from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.training.eval import Evaluator
from shape_foundation.training.losses import LossComputer, compute_raw_geo_stats_dim
from shape_foundation.data.dataset import build_dataloader


def _default_output_path(checkpoint: str, split: str) -> Path:
    """Put the metrics JSON next to the checkpoint so runs stay grouped."""
    ckpt = Path(checkpoint)
    return ckpt.with_name(f"{ckpt.stem}_eval_{split}.json")


def _print_summary(results: dict, checkpoint: str, split: str) -> None:
    bar = "=" * 60
    print("\n" + bar)
    print(f"Evaluation summary")
    print(bar)
    print(f"  checkpoint : {checkpoint}")
    print(f"  split      : {split}")
    print(f"  n_metrics  : {len(results)}")
    print("-" * 60)
    if not results:
        print("  (no metrics reported)")
    else:
        width = max(len(k) for k in results)
        for k in sorted(results):
            v = results[k]
            if isinstance(v, (int, float)):
                print(f"  {k:<{width}} : {v:.4f}")
            else:
                print(f"  {k:<{width}} : {v}")
    print(bar)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Shape foundation checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint_*.pt produced by training")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML override; falls back to config embedded in the checkpoint")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to evaluate on (default: val)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: <checkpoint>_eval_<split>.json next to the checkpoint)")
    parser.add_argument("--robustness", action="store_true",
                        help="Additionally run robustness benchmarks")
    parser.add_argument("--history", type=str, default=None,
                        help="Optional path to a shared CSV file; each run "
                             "appends a new row with its metrics for "
                             "cross-run comparison and plotting.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Config: explicit YAML wins, else checkpoint-embedded, else defaults.
    if args.config:
        cfg = load_config(args.config)
        print(f"Config: {args.config}")
    elif "config" in ckpt:
        cfg = ckpt["config"]
        print("Config: (loaded from checkpoint)")
    else:
        cfg = ShapeConfig()
        print("Config: (defaults)")

    # Build model + load weights
    model = GAOTBackbone(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    print(f"Model: {model.get_num_params():,} params")

    # Build LossComputer from the checkpoint so the evaluator uses the same
    # calibrated per-dimension target normalization buffers and the same
    # masked-token mask generator that were used during training. This keeps
    # reconstruction metrics directly comparable to the training loss curves.
    loss_computer = LossComputer(
        cfg.train.loss,
        token_dim=cfg.tokenizer.latent.token_dim,
        recon_target_dim=compute_raw_geo_stats_dim(cfg.tokenizer.geo_embed),
    )
    if "loss_computer_state_dict" in ckpt:
        try:
            loss_computer.load_state_dict(
                ckpt["loss_computer_state_dict"], strict=False,
            )
        except Exception as e:
            print(f"  Warning: could not restore loss_computer state: {e}")
    loss_computer.set_grid_shape(cfg.tokenizer.latent.latent_shape)
    if loss_computer.is_recon_target_calibrated():
        print("Recon target normalization: using calibrated buffers from checkpoint")
    else:
        print("Recon target normalization: uncalibrated (reconstruction metrics "
              "will be in raw target space)")

    evaluator = Evaluator(model, device=args.device, loss_computer=loss_computer)

    dataloader = build_dataloader(
        cfg.data, cfg.input, split=args.split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )
    print(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches "
          f"(split={args.split})")

    print("\nRunning evaluation...")
    results = evaluator.evaluate_all(dataloader)

    if args.robustness:
        print("Running robustness evaluation...")
        robust_results = evaluator.eval_robustness(dataloader)
        results.update(robust_results)

    _print_summary(results, str(checkpoint_path), args.split)

    # ------------------------------------------------------------------
    # Save metrics — JSON (structured) + CSV (flat, plot-friendly)
    # ------------------------------------------------------------------
    output_path = Path(args.output) if args.output else _default_output_path(
        str(checkpoint_path), args.split,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    metric_row = {
        "timestamp": timestamp,
        "checkpoint": Path(checkpoint_path).name,
        "split": args.split,
        **{k: float(v) for k, v in results.items() if isinstance(v, (int, float))},
    }

    # 1. JSON — structured payload for programmatic access
    payload = {
        "timestamp": timestamp,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "metrics": {k: float(v) for k, v in results.items() if isinstance(v, (int, float))},
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nSaved metrics to: {output_path}")

    # 2. CSV — single-row, header included, for plotting and spreadsheets
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_row.keys()))
        writer.writeheader()
        writer.writerow(metric_row)
    print(f"Saved CSV to: {csv_path}")

    # 3. History CSV — appended row in a shared file for multi-run tracking
    if args.history:
        history_path = Path(args.history)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = history_path.exists()
        # Read existing header so we can grow the columns monotonically if
        # future runs add new metrics.
        existing_fields: list[str] = []
        if file_exists:
            with open(history_path, "r", newline="") as f:
                reader = csv.reader(f)
                try:
                    existing_fields = next(reader)
                except StopIteration:
                    existing_fields = []
        all_fields = list(dict.fromkeys(existing_fields + list(metric_row.keys())))
        # Rewrite the file with the unified header so old rows stay readable.
        rows: list[dict] = []
        if file_exists and existing_fields:
            with open(history_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        rows.append({k: metric_row.get(k, "") for k in all_fields})
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in all_fields})
        print(f"Appended to history: {history_path}")


if __name__ == "__main__":
    main()
