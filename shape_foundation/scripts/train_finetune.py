"""Fine-tuning script for Shape foundation model.

Stage 2+: Fine-tune pretrained backbone with supervised heads
for symmetry, primitive, part, and reduction tasks.

Usage:
    python -m shape_foundation.scripts.train_finetune \
        --config configs/medium.yaml \
        --pretrained checkpoints/checkpoint_final.pt \
        --freeze-backbone-epochs 5

    torchrun --nproc_per_node=4 -m shape_foundation.scripts.train_finetune \
        --config configs/large.yaml \
        --pretrained checkpoints/checkpoint_final.pt
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from shape_foundation.configs.default import ShapeConfig, load_config, save_config
from shape_foundation.training.trainer import Trainer


def setup_distributed() -> bool:
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
    parser = argparse.ArgumentParser(description="Fine-tune Shape foundation model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pretrained", type=str, required=True, help="Pretrained checkpoint path")
    parser.add_argument("--resume", type=str, default=None, help="Fine-tuning checkpoint to resume from")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                        help="Freeze encoder+processor for N epochs, only train heads")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_finetune")
    parser.add_argument("--log-dir", type=str, default="runs_finetune")
    args = parser.parse_args()

    is_distributed = setup_distributed()
    is_main = not is_distributed or dist.get_rank() == 0

    cfg = load_config(args.config) if args.config else ShapeConfig()

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.optimizer.lr = args.lr
    cfg.train.checkpoint_dir = args.checkpoint_dir
    cfg.train.log_dir = args.log_dir

    # Fine-tuning: lower LR, enable all supervised losses
    if args.lr is None:
        cfg.train.optimizer.lr = cfg.train.optimizer.lr * 0.1  # 10x lower for fine-tuning

    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    if is_main:
        print("=" * 60)
        print("Shape Foundation Model — Fine-tuning")
        print("=" * 60)
        print(f"Pretrained: {args.pretrained}")
        print(f"Freeze backbone: {args.freeze_backbone_epochs} epochs")
        print()

    # Save config
    if is_main:
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        save_config(cfg, os.path.join(cfg.train.checkpoint_dir, "config_finetune.yaml"))

    # Build trainer (loads pretrained weights)
    trainer = Trainer(cfg, resume_from=args.pretrained)

    # Freeze backbone if requested
    freeze_epochs = args.freeze_backbone_epochs
    if freeze_epochs > 0:
        if is_main:
            print(f"Freezing encoder + processor for {freeze_epochs} epochs...")
        for name, param in trainer.raw_model.named_parameters():
            if "encoder" in name or "processor" in name:
                param.requires_grad = False

        # Rebuild optimizer with only trainable params
        trainer.optimizer = trainer._build_optimizer()
        trainer.scheduler = trainer._build_scheduler()

    # Train
    train_loader, val_loader = trainer.build_dataloaders()

    for epoch in range(cfg.train.epochs):
        # Unfreeze after freeze_epochs
        if epoch == freeze_epochs and freeze_epochs > 0:
            if is_main:
                print("Unfreezing backbone...")
            for param in trainer.raw_model.parameters():
                param.requires_grad = True
            trainer.optimizer = trainer._build_optimizer()
            trainer.scheduler = trainer._build_scheduler()

        train_metrics = trainer.train_epoch(train_loader, epoch)
        if is_main:
            print(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))

        if val_loader is not None:
            val_metrics = trainer.evaluate(val_loader)
            if is_main:
                print(f"  Val: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

    if is_main:
        trainer._save_checkpoint(cfg.train.epochs - 1, final=True)
        print("Fine-tuning complete.")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
