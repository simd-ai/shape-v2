"""Pretraining script for Shape foundation model.

Stage 1: Self-supervised pretraining with masked token modeling,
multi-resolution contrastive learning, and partial inpainting.

Usage:
    # Single GPU
    python -m shape_foundation.scripts.train_pretrain --config configs/medium.yaml

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 -m shape_foundation.scripts.train_pretrain --config configs/large.yaml

    # Smoke test
    python -m shape_foundation.scripts.train_pretrain --config configs/smoke_test.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

from shape_foundation.configs.default import ShapeConfig, load_config, save_config
from shape_foundation.training.trainer import Trainer


def setup_distributed() -> bool:
    """Initialize DDP if running under torchrun.

    Uses NCCL backend for CUDA (optimal for NVLink/NVSwitch on H100),
    falls back to gloo for CPU.
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Pretrain Shape foundation model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Run one batch to verify setup")

    # Override specific config values
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Setup distributed
    is_distributed = setup_distributed()
    is_main = not is_distributed or dist.get_rank() == 0

    # Load config
    cfg = load_config(args.config) if args.config else ShapeConfig()

    # Apply overrides
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.optimizer.lr = args.lr
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        cfg.train.log_dir = args.log_dir
    if args.seed is not None:
        cfg.train.seed = args.seed

    # For pretraining, disable supervised losses (no labels yet)
    # Keep self-supervised: masked_token, contrastive, inpainting
    # Supervised heads still train if labels happen to be present

    # Set seed
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    if is_main:
        print("=" * 60)
        print("Shape Foundation Model — Pretraining")
        print("=" * 60)
        print(f"Config: {args.config or 'default'}")
        print(f"Distributed: {is_distributed}" + (f" ({dist.get_world_size()} GPUs)" if is_distributed else ""))
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPUs visible: {torch.cuda.device_count()}")
        print(f"Epochs: {cfg.train.epochs}")
        print(f"Batch size: {cfg.train.batch_size}" + (f" (effective: {cfg.train.batch_size * cfg.train.gradient_accumulation_steps * (dist.get_world_size() if is_distributed else 1)})" if cfg.train.gradient_accumulation_steps > 1 or is_distributed else ""))
        print(f"LR: {cfg.train.optimizer.lr}")
        print(f"Mixed precision: {cfg.train.mixed_precision}")
        print()

    # Save config
    if is_main:
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        save_config(cfg, os.path.join(cfg.train.checkpoint_dir, "config.yaml"))

    # Build trainer
    trainer = Trainer(cfg, resume_from=args.resume)

    if args.dry_run:
        if is_main:
            print("Dry run: building dataloaders...")
        train_loader, val_loader = trainer.build_dataloaders()
        if is_main:
            print(f"Train: {len(train_loader)} batches")
            if val_loader:
                print(f"Val: {len(val_loader)} batches")

        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            if is_main:
                print(f"Batch keys: {list(batch.keys())}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.shape} {v.dtype}")

            if is_main:
                print("Running one forward+backward pass...")
            out = trainer._forward_batch(batch)
            if is_main:
                print(f"Output keys: {list(out.keys())}")
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.shape}")
                print("Dry run successful!")
        else:
            if is_main:
                print("No training data found. Add data to sources or generate synthetic data first.")
        return

    # Train
    trainer.train()

    if is_distributed:
        dist.destroy_process_group()

    if is_main:
        print("Pretraining complete.")


if __name__ == "__main__":
    main()
