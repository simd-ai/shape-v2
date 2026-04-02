"""DDP-friendly training loop with mixed precision, gradient accumulation, and logging."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from shape_foundation.configs.default import ShapeConfig
from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.training.losses import LossComputer
from shape_foundation.data.dataset import MeshDataset, CollateFunction


class Trainer:
    """Training loop for Shape foundation model.

    Supports:
        - Single GPU and DDP multi-GPU (tested on 8xH100)
        - Mixed precision (bf16/fp16/fp32) on CUDA and CPU
        - Gradient accumulation
        - torch.compile
        - Checkpointing with resume
        - TensorBoard logging
        - Contrastive augmentation (two-view forward)
    """

    def __init__(self, cfg: ShapeConfig, resume_from: str | None = None):
        self.cfg = cfg
        self.tcfg = cfg.train
        self.device = self._setup_device()
        self.is_main = self._is_main_process()
        self.use_cuda = self.device.type == "cuda"

        # Model
        self.model = GAOTBackbone(cfg).to(self.device)
        if self.tcfg.compile_model and self.use_cuda:
            self.model = torch.compile(self.model)

        # DDP — use find_unused_parameters=False for speed; the architecture
        # should not have unused params in the default forward path.
        if dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.use_cuda else None,
                output_device=self.device.index if self.use_cuda else None,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,  # saves memory on multi-GPU
            )
        self.raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        # Losses
        self.loss_computer = LossComputer(self.tcfg.loss).to(self.device)
        self.loss_computer.set_grid_shape(cfg.tokenizer.latent.latent_shape)

        # Optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Mixed precision — device-agnostic
        self.scaler = None
        self.amp_dtype = torch.float32
        self.amp_enabled = False
        if self.tcfg.mixed_precision == "bf16":
            self.amp_dtype = torch.bfloat16
            self.amp_enabled = True
        elif self.tcfg.mixed_precision == "fp16":
            self.amp_dtype = torch.float16
            self.amp_enabled = True
            if self.use_cuda:
                self.scaler = torch.amp.GradScaler("cuda")
            # fp16 on CPU has no GradScaler; bf16 is preferred for CPU

        # autocast device type
        self._amp_device_type = "cuda" if self.use_cuda else "cpu"

        # Logging
        self.writer = None
        if self.is_main:
            log_dir = Path(self.tcfg.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))

        # Checkpointing
        self.ckpt_dir = Path(self.tcfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.start_epoch = 0

        if resume_from:
            self._load_checkpoint(resume_from)

        if self.is_main:
            n_params = self.raw_model.get_num_params()
            n_train = self.raw_model.get_num_trainable_params()
            print(f"Model: {n_params:,} params ({n_train:,} trainable)")
            print(f"Device: {self.device}  AMP: {self.tcfg.mixed_precision}  DDP: {dist.is_initialized()}")
            if self.use_cuda:
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.0f} GB)")

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            return device
        return torch.device("cpu")

    def _is_main_process(self) -> bool:
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def _build_optimizer(self) -> torch.optim.Optimizer:
        ocfg = self.tcfg.optimizer
        # separate weight decay: no decay for biases, norms, embeddings
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": ocfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if ocfg.name == "adamw":
            return torch.optim.AdamW(param_groups, lr=ocfg.lr, betas=ocfg.betas, fused=self.use_cuda)
        elif ocfg.name == "adam":
            return torch.optim.Adam(param_groups, lr=ocfg.lr, betas=ocfg.betas, fused=self.use_cuda)
        else:
            return torch.optim.SGD(param_groups, lr=ocfg.lr, momentum=0.9)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        ocfg = self.tcfg.optimizer
        if ocfg.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.tcfg.epochs * 1000, eta_min=1e-6,
            )
        elif ocfg.lr_schedule == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01,
                total_iters=self.tcfg.epochs * 1000,
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def _warmup_lr(self) -> None:
        warmup = self.tcfg.optimizer.warmup_steps
        if self.global_step < warmup:
            factor = self.global_step / max(warmup, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.tcfg.optimizer.lr * factor

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """Build train and optional val dataloaders."""
        train_dataset = MeshDataset(self.cfg.data, self.cfg.input, split="train")
        val_dataset = MeshDataset(self.cfg.data, self.cfg.input, split="val")

        train_sampler = None
        val_sampler = None
        if dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            if len(val_dataset) > 0:
                val_sampler = DistributedSampler(val_dataset, shuffle=False)

        # pin_memory only useful on CUDA
        pin = self.tcfg.pin_memory and self.use_cuda

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.tcfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.tcfg.num_workers,
            pin_memory=pin,
            collate_fn=CollateFunction(),
            drop_last=True,
            persistent_workers=self.tcfg.num_workers > 0,
            prefetch_factor=2 if self.tcfg.num_workers > 0 else None,
        )

        val_loader = None
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.tcfg.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.tcfg.num_workers,
                pin_memory=pin,
                collate_fn=CollateFunction(),
                persistent_workers=self.tcfg.num_workers > 0,
                prefetch_factor=2 if self.tcfg.num_workers > 0 else None,
            )

        # update scheduler total steps
        steps_per_epoch = max(1, len(train_loader) // self.tcfg.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.tcfg.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6,
        )

        return train_loader, val_loader

    def _forward_batch(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run forward pass on a batch."""
        points = batch["points"].to(self.device, non_blocking=True)
        features = batch["features"].to(self.device, non_blocking=True)
        normals = batch.get("normals")
        curvature = batch.get("curvature")

        if normals is not None:
            normals = normals.to(self.device, non_blocking=True)
        if curvature is not None:
            curvature = curvature.to(self.device, non_blocking=True)

        with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            out = self.model(points, features, normals, curvature)
        return out

    def train_epoch(
        self, train_loader: DataLoader, epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        accum = self.tcfg.gradient_accumulation_steps
        epoch_losses: dict[str, list[float]] = {}

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not self.is_main)
        self.optimizer.zero_grad(set_to_none=True)

        for step_in_epoch, batch in enumerate(pbar):
            self._warmup_lr()

            # Forward
            out = self._forward_batch(batch)

            # Second augmentation for contrastive loss
            out_aug = None
            if self.cfg.train.loss.contrastive.enabled:
                out_aug = self._forward_batch(batch)

            # Move labels to device
            label_keys = [
                "symmetry_label", "symmetry_planes", "symmetry_axes",
                "primitive_labels", "part_labels", "reduction_label",
                "repeated_sectors", "constant_cross_section",
            ]
            batch_device = {}
            for k in label_keys:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch_device[k] = batch[k].to(self.device, non_blocking=True)

            # Compute losses
            with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                losses = self.loss_computer(out, batch_device, out_aug)

            loss = losses["total"] / accum

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (step_in_epoch + 1) % accum == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Log
                if self.is_main and self.global_step % self.tcfg.log_every == 0:
                    for k, v in losses.items():
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        self.writer.add_scalar(f"train/{k}", val, self.global_step)
                    self.writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step,
                    )
                    if self.use_cuda:
                        mem_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
                        self.writer.add_scalar("train/gpu_mem_gb", mem_gb, self.global_step)

                # Save
                if self.is_main and self.global_step % self.tcfg.save_every == 0:
                    self._save_checkpoint(epoch)

            # Track epoch losses
            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                epoch_losses.setdefault(k, []).append(val)

            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        all_losses: dict[str, list[float]] = {}

        for batch in tqdm(val_loader, desc="Eval", disable=not self.is_main):
            out = self._forward_batch(batch)

            label_keys = [
                "symmetry_label", "symmetry_planes", "symmetry_axes",
                "primitive_labels", "part_labels", "reduction_label",
            ]
            batch_device = {}
            for k in label_keys:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch_device[k] = batch[k].to(self.device, non_blocking=True)

            with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                losses = self.loss_computer(out, batch_device)

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                all_losses.setdefault(k, []).append(val)

        avg = {k: sum(v) / len(v) for k, v in all_losses.items()}

        if self.is_main and self.writer:
            for k, v in avg.items():
                self.writer.add_scalar(f"val/{k}", v, self.global_step)

        return avg

    def train(self) -> None:
        """Full training loop."""
        train_loader, val_loader = self.build_dataloaders()

        if self.is_main:
            print(f"Training: {len(train_loader)} batches/epoch, {self.tcfg.epochs} epochs")
            if dist.is_initialized():
                print(f"DDP: {dist.get_world_size()} processes")

        for epoch in range(self.start_epoch, self.tcfg.epochs):
            train_metrics = self.train_epoch(train_loader, epoch)

            if self.is_main:
                print(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))

            if val_loader is not None and (epoch + 1) % max(1, self.tcfg.eval_every // max(1, len(train_loader))) == 0:
                val_metrics = self.evaluate(val_loader)
                if self.is_main:
                    print(f"  Val: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        # Final save
        if self.is_main:
            self._save_checkpoint(self.tcfg.epochs - 1, final=True)
            if self.writer:
                self.writer.close()

    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        tag = "final" if final else f"step_{self.global_step}"
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
        }, path)
        if self.is_main:
            print(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            if self.is_main:
                print(f"Warning: could not restore optimizer state (param groups changed); starting fresh optimizer")
        if "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        self.global_step = ckpt.get("global_step", 0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        if self.is_main:
            print(f"Resumed from {path} (epoch {self.start_epoch}, step {self.global_step})")
