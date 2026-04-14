"""DDP-friendly training loop with mixed precision, gradient accumulation, and logging."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import dataclasses as dc

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from shape_foundation.configs.default import ShapeConfig
from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.training.losses import LossComputer, compute_raw_geo_stats_dim
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

        # Model: DDP first, compile second (PyTorch recommended order)
        base_model = GAOTBackbone(cfg).to(self.device)

        if dist.is_initialized():
            base_model = DDP(
                base_model,
                device_ids=[self.device.index] if self.use_cuda else None,
                output_device=self.device.index if self.use_cuda else None,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )

        if self.tcfg.compile_model and self.use_cuda:
            base_model = torch.compile(base_model)

        self.model = base_model
        self.raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        # Losses — all trainable submodules (including the reconstruction
        # projection head) are built inside LossComputer.__init__ so every
        # parameter is registered with the optimizer from step 0. No lazy
        # first-forward construction.
        token_dim = cfg.tokenizer.latent.token_dim
        recon_target_dim = compute_raw_geo_stats_dim(cfg.tokenizer.geo_embed)
        self.loss_computer = LossComputer(
            self.tcfg.loss,
            token_dim=token_dim,
            recon_target_dim=recon_target_dim,
        ).to(self.device)
        self.loss_computer.set_grid_shape(cfg.tokenizer.latent.latent_shape)

        # Optimizer (sees both model and loss_computer params)
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

        # Logging — env vars override the YAML/config value so that a single
        # training invocation can be redirected to a different disk without
        # editing config files.
        log_dir_str = os.environ.get("SHAPE_LOG_DIR", self.tcfg.log_dir)
        self.log_dir = Path(log_dir_str)
        self.writer = None
        self.wandb_run = None
        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

            # Weights & Biases
            if self.tcfg.wandb.enabled:
                try:
                    import wandb
                    wcfg = self.tcfg.wandb
                    self.wandb_run = wandb.init(
                        project=wcfg.project,
                        entity=wcfg.entity or None,
                        name=wcfg.name or None,
                        tags=wcfg.tags or None,
                        config=dc.asdict(cfg),
                        resume="allow",
                    )
                except ImportError:
                    print("Warning: wandb not installed, skipping W&B logging. pip install wandb")
                except Exception as e:
                    print(f"Warning: wandb init failed: {e}")

        # Checkpointing — env var override mirrors SHAPE_LOG_DIR semantics.
        ckpt_dir_str = os.environ.get("SHAPE_CHECKPOINT_DIR", self.tcfg.checkpoint_dir)
        self.ckpt_dir = Path(ckpt_dir_str)
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
            reg = self.tcfg.loss.regression
            print(f"Regression loss: {reg.kind}" + (f" (beta={reg.beta})" if reg.kind == "smooth_l1" else ""))
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
        # include both model and loss_computer (recon_proj) parameters
        decay_params = []
        no_decay_params = []
        all_modules = [("model", self.model), ("loss_computer", self.loss_computer)]
        for module_name, module in all_modules:
            for name, param in module.named_parameters():
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

        # Loud-fail: if the config asked for validation but the val dataset
        # is empty, refuse to start. Previously this silently skipped eval
        # for the entire run, making overfit impossible to detect.
        if self.cfg.data.val_fraction > 0.0 and val_loader is None:
            raise RuntimeError(
                f"cfg.data.val_fraction={self.cfg.data.val_fraction} but the val "
                f"dataset is empty. Train samples: {len(train_dataset)}. "
                "This usually means your sources produce too few files for the "
                "hash-split threshold, or the source roots are wrong. Either "
                "lower val_fraction, add more data, or set val_fraction=0.0 to "
                "opt out of validation."
            )
        if self.is_main and val_loader is not None:
            print(
                f"Dataset split (val_fraction={self.cfg.data.val_fraction}): "
                f"train={len(train_dataset):,}, val={len(val_dataset):,}"
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
        augment: bool = False,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run forward pass on a batch.

        Args:
            augment: if True, apply random jitter and point dropout for
                     contrastive learning (second view).
            mask: (B, T) bool token mask for masked-token pretraining.
                  Passed through to the model so the processor replaces
                  masked patch tokens with a learnable mask embedding.
        """
        points = batch["points"].to(self.device, non_blocking=True)
        features = batch["features"].to(self.device, non_blocking=True)
        normals = batch.get("normals")
        curvature = batch.get("curvature")

        if normals is not None:
            normals = normals.to(self.device, non_blocking=True)
        if curvature is not None:
            curvature = curvature.to(self.device, non_blocking=True)

        if augment and self.model.training:
            # Augmentation strengths are read from contrastive config so
            # that tuning them changes both the concat path and this
            # legacy two-forward fallback identically.
            aug = self.cfg.train.loss.contrastive
            # Jitter point positions
            noise = torch.randn_like(points) * aug.jitter_std
            points = points + noise

            # Keep features aligned with jittered points (first 3 dims are xyz)
            features = features.clone()
            if features.shape[-1] >= 3:
                features[..., :3] = points

            # Random point dropout
            drop_mask = torch.rand(points.shape[:2], device=points.device) < aug.point_dropout
            keep = (~drop_mask).unsqueeze(-1).float()
            points = points * keep
            features = features * keep
            if normals is not None:
                normals = normals * keep
            if curvature is not None:
                curvature = curvature * keep

        with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            out = self.model(points, features, normals, curvature, mask)
        return out

    def _forward_concat(
        self, batch: dict[str, torch.Tensor], token_mask: torch.Tensor | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Single DDP forward with masked + augmented views concatenated.

        Avoids two DDP forwards before one backward, which causes
        gradient sync issues with find_unused_parameters=True.
        """
        # View A: masked (or unmasked if no mask)
        points_a = batch["points"].to(self.device, non_blocking=True)
        features_a = batch["features"].to(self.device, non_blocking=True)
        normals_a = batch.get("normals")
        curvature_a = batch.get("curvature")
        if normals_a is not None:
            normals_a = normals_a.to(self.device, non_blocking=True)
        if curvature_a is not None:
            curvature_a = curvature_a.to(self.device, non_blocking=True)

        # View B: augmented (jitter + dropout, no mask). Strengths come
        # from the contrastive config so raising them doesn't require a
        # code change.
        aug = self.cfg.train.loss.contrastive
        noise = torch.randn_like(points_a) * aug.jitter_std
        points_b = points_a + noise
        features_b = features_a.clone()
        if features_b.shape[-1] >= 3:
            features_b[..., :3] = points_b
        drop_mask = torch.rand(points_b.shape[:2], device=points_b.device) < aug.point_dropout
        keep = (~drop_mask).unsqueeze(-1).float()
        points_b = points_b * keep
        features_b = features_b * keep
        normals_b = normals_a * keep if normals_a is not None else None
        curvature_b = curvature_a * keep if curvature_a is not None else None

        B = points_a.shape[0]

        # Concat along batch dim
        points_cat = torch.cat([points_a, points_b], dim=0)
        features_cat = torch.cat([features_a, features_b], dim=0)
        normals_cat = torch.cat([normals_a, normals_b], dim=0) if normals_a is not None else None
        curvature_cat = torch.cat([curvature_a, curvature_b], dim=0) if curvature_a is not None else None

        # Mask: view A gets token_mask, view B gets no mask (all False)
        if token_mask is not None:
            mask_b = torch.zeros_like(token_mask)
            mask_cat = torch.cat([token_mask, mask_b], dim=0)
        else:
            mask_cat = None

        # Single forward
        with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            out_cat = self.model(points_cat, features_cat, normals_cat, curvature_cat, mask_cat)

        # Split outputs back into two views
        def split_dict(d: dict, B: int) -> tuple[dict, dict]:
            a, b = {}, {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == 2 * B:
                    a[k] = v[:B]
                    b[k] = v[B:]
                elif isinstance(v, dict):
                    a[k], b[k] = split_dict(v, B)
                elif isinstance(v, list):
                    a[k] = [t[:B] if isinstance(t, torch.Tensor) and t.shape[0] == 2 * B else t for t in v]
                    b[k] = [t[B:] if isinstance(t, torch.Tensor) and t.shape[0] == 2 * B else t for t in v]
                else:
                    a[k] = v
                    b[k] = v
            return a, b

        out, out_aug = split_dict(out_cat, B)
        return out, out_aug

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

        use_token_mask = (
            self.cfg.train.loss.masked_token.enabled
            and self.loss_computer.masked_token is not None
        )

        for step_in_epoch, batch in enumerate(pbar):
            self._warmup_lr()
            B = batch["points"].shape[0]

            # Create masked-token mask BEFORE forward so the model sees masked input
            token_mask = None
            if use_token_mask:
                T = self.raw_model.encoder.num_tokens
                token_mask = self.loss_computer.masked_token.create_mask(B, T, self.device)

            # Contrastive: by default concat masked + augmented views into a
            # single DDP forward (safer under find_unused_parameters=True).
            # Legacy two-forward path is still selectable via
            # train.loss.contrastive.concat_forward = false for comparison.
            do_contrastive = self.cfg.train.loss.contrastive.enabled
            use_concat = self.cfg.train.loss.contrastive.concat_forward
            out_aug = None

            if do_contrastive and use_concat:
                out, out_aug = self._forward_concat(batch, token_mask)
            elif do_contrastive:
                out = self._forward_batch(batch, mask=token_mask)
                out_aug = self._forward_batch(batch, augment=True, mask=None)
            else:
                out = self._forward_batch(batch, mask=token_mask)

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

            # Compute losses — pass same token_mask so reconstruction uses same positions
            with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                losses = self.loss_computer(out, batch_device, out_aug, token_mask=token_mask)

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
                    log_dict = {}
                    for k, v in losses.items():
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        self.writer.add_scalar(f"train/{k}", val, self.global_step)
                        log_dict[f"train/{k}"] = val
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    log_dict["train/lr"] = lr
                    if self.use_cuda:
                        mem_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
                        self.writer.add_scalar("train/gpu_mem_gb", mem_gb, self.global_step)
                        log_dict["train/gpu_mem_gb"] = mem_gb
                    if self.wandb_run is not None:
                        import wandb
                        wandb.log(log_dict, step=self.global_step)

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

        use_token_mask = (
            self.cfg.train.loss.masked_token.enabled
            and self.loss_computer.masked_token is not None
        )

        # Mirror training: if contrastive is enabled, use the concat forward
        # so val sees both views and val/contrastive_raw gets logged.
        do_contrastive = self.cfg.train.loss.contrastive.enabled
        use_concat = self.cfg.train.loss.contrastive.concat_forward

        for batch in tqdm(val_loader, desc="Eval", disable=not self.is_main):
            B = batch["points"].shape[0]

            # Mirror training: create mask and pass into model + loss
            token_mask = None
            if use_token_mask:
                T = self.raw_model.encoder.num_tokens
                token_mask = self.loss_computer.masked_token.create_mask(B, T, self.device)

            out_aug = None
            if do_contrastive and use_concat:
                out, out_aug = self._forward_concat(batch, token_mask)
            else:
                out = self._forward_batch(batch, mask=token_mask)

            label_keys = [
                "symmetry_label", "symmetry_planes", "symmetry_axes",
                "primitive_labels", "part_labels", "reduction_label",
                "repeated_sectors", "constant_cross_section",
            ]
            batch_device = {}
            for k in label_keys:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch_device[k] = batch[k].to(self.device, non_blocking=True)

            with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                losses = self.loss_computer(out, batch_device, out_aug, token_mask=token_mask)

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                all_losses.setdefault(k, []).append(val)

        avg = {k: sum(v) / len(v) for k, v in all_losses.items()}

        if self.is_main and self.writer:
            for k, v in avg.items():
                self.writer.add_scalar(f"val/{k}", v, self.global_step)
            if self.wandb_run is not None:
                import wandb
                wandb.log({f"val/{k}": v for k, v in avg.items()}, step=self.global_step)

        return avg

    @torch.no_grad()
    def _calibrate_recon_target_stats(self, train_loader: DataLoader) -> None:
        """Compute per-dim (mean, std) of raw_geo_stats on the training split.

        Runs once at the start of training and writes the stats into the
        `recon_target_mean` / `recon_target_std` buffers on LossComputer
        so subsequent train / val / eval forward calls all use the same
        normalization. Uses only the training dataloader — no val/test
        data is touched — so validation statistics cannot leak into the
        training targets.

        DDP: all ranks run the calibration forward on their own local
        shard, then sum/sumsq/count are all-reduced across the process
        group so every rank ends up with identical global stats.

        Skipped entirely if:
        - recon_target_norm.enabled is False in config, OR
        - LossComputer has no recon_proj (no reconstruction losses), OR
        - buffers are already calibrated (e.g. from a resumed checkpoint).
        """
        lc = self.loss_computer
        cfg_norm = self.tcfg.loss.recon_target_norm

        if not cfg_norm.enabled:
            if self.is_main:
                print("Recon target normalization: disabled")
            return
        if lc.recon_proj is None:
            if self.is_main:
                print("Recon target normalization: skipped (no reconstruction losses)")
            return
        if lc.is_recon_target_calibrated():
            if self.is_main:
                m = lc.recon_target_mean
                s = lc.recon_target_std
                print(
                    f"Recon target normalization: using calibrated buffers from checkpoint "
                    f"(D={m.numel()}, mean∈[{m.min().item():+.3f}, {m.max().item():+.3f}], "
                    f"std∈[{s.min().item():+.3e}, {s.max().item():+.3e}])"
                )
            return

        n_batches = int(cfg_norm.n_calib_batches)
        if n_batches <= 0:
            if self.is_main:
                print("Recon target normalization: n_calib_batches <= 0, skipping")
            return

        if self.is_main:
            print(f"Calibrating recon target normalization on {n_batches} training batches...")

        was_training = self.model.training
        self.model.eval()

        D = int(lc.recon_target_dim)
        sum_ = torch.zeros(D, dtype=torch.float64, device=self.device)
        sumsq = torch.zeros(D, dtype=torch.float64, device=self.device)
        count = torch.zeros(1, dtype=torch.float64, device=self.device)

        from itertools import islice
        seen_raw_stats = False
        for batch in islice(train_loader, n_batches):
            out = self._forward_batch(batch, augment=False, mask=None)
            if "raw_geo_stats" not in out:
                break
            seen_raw_stats = True
            stats = out["raw_geo_stats"].reshape(-1, D).to(torch.float64)
            sum_ += stats.sum(dim=0)
            sumsq += (stats * stats).sum(dim=0)
            count += stats.shape[0]

        # DDP: pool across ranks so every rank gets the same global stats.
        if dist.is_initialized():
            dist.all_reduce(sum_, op=dist.ReduceOp.SUM)
            dist.all_reduce(sumsq, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        self.model.train(mode=was_training)

        n = count.item()
        if not seen_raw_stats or n < 2:
            if self.is_main:
                print(
                    "Warning: calibration saw insufficient samples "
                    f"(seen_raw_stats={seen_raw_stats}, n={n}); "
                    "skipping normalization (targets will pass through)."
                )
            return

        mean = (sum_ / n).to(torch.float32)
        var = (sumsq / n - (sum_ / n) ** 2).clamp(min=0.0).to(torch.float32)
        std = var.sqrt()

        lc.set_recon_target_stats(mean, std)

        if self.is_main:
            print(
                f"Recon target normalization calibrated (D={D}, n={int(n):,} tokens):\n"
                f"  mean range: [{mean.min().item():+.4f}, {mean.max().item():+.4f}]\n"
                f"  std  range: [{std.min().item():+.4e}, {std.max().item():+.4e}]"
            )

    def train(self) -> None:
        """Full training loop."""
        train_loader, val_loader = self.build_dataloaders()

        if self.is_main:
            print(f"Training: {len(train_loader)} batches/epoch, {self.tcfg.epochs} epochs")
            if dist.is_initialized():
                print(f"DDP: {dist.get_world_size()} processes")

        # Calibrate per-dim reconstruction-target normalization stats from
        # training data only. Runs once before epoch 0 unless a resumed
        # checkpoint already provided calibrated buffers.
        self._calibrate_recon_target_stats(train_loader)

        for epoch in range(self.start_epoch, self.tcfg.epochs):
            train_metrics = self.train_epoch(train_loader, epoch)

            if self.is_main:
                print(f"Epoch {epoch} " + ", ".join(
                    f"train_epoch/{k}={v:.4f}" for k, v in sorted(train_metrics.items())
                ))
                if self.writer:
                    for k, v in train_metrics.items():
                        self.writer.add_scalar(f"train_epoch/{k}", v, epoch)
                if self.wandb_run is not None:
                    import wandb
                    wandb.log(
                        {f"train_epoch/{k}": v for k, v in train_metrics.items()},
                        step=self.global_step,
                    )

            if val_loader is not None and (epoch + 1) % max(1, self.tcfg.eval_every // max(1, len(train_loader))) == 0:
                val_metrics = self.evaluate(val_loader)
                if self.is_main:
                    print(f"Epoch {epoch} " + ", ".join(
                        f"val/{k}={v:.4f}" for k, v in sorted(val_metrics.items())
                    ))

        # Final save
        if self.is_main:
            self._save_checkpoint(self.tcfg.epochs - 1, final=True)
            if self.writer:
                self.writer.close()
            if self.wandb_run is not None:
                import wandb
                wandb.finish()

    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        """Rank-0 atomic checkpoint save.

        Writes the payload to a sibling `<name>.tmp` file, fsyncs, then
        os.replace()s it onto the final path. If torch.save raises (e.g.
        disk full), the partial .tmp is removed and the previous final
        checkpoint is left untouched. os.replace is atomic within a single
        filesystem, which is guaranteed because the temp file lives in the
        same directory as the target.
        """
        tag = "final" if final else f"step_{self.global_step}"
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        tmp_path = path.with_name(path.name + ".tmp")
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.raw_model.state_dict(),
            "loss_computer_state_dict": self.loss_computer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
        }
        try:
            with open(tmp_path, "wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if self.is_main:
                print(f"ERROR: checkpoint save failed at {path}: {e}")
            raise
        if self.is_main:
            print(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "loss_computer_state_dict" in ckpt:
            try:
                self.loss_computer.load_state_dict(ckpt["loss_computer_state_dict"], strict=False)
            except Exception:
                if self.is_main:
                    print("Warning: could not restore loss_computer state; recon_proj reinitialized")
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
