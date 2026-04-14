"""Loss functions for Shape foundation model training.

Implements:
    1. Masked latent-token modeling loss (mask created BEFORE forward, passed in)
    2. Multi-resolution contrastive consistency loss
    3. Partial-shape inpainting loss
    4. Supervised symmetry/primitive/part/reduction losses
    5. Optional 3D-text alignment loss
    6. Configurable per-loss scalar weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_foundation.configs.default import GeoEmbedConfig, LossConfig


def regression_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    beta: float,
) -> torch.Tensor:
    """Dispatch between MSE and SmoothL1 for geometry regression losses.

    Single source of truth used by MaskedTokenLoss, PartialInpaintingLoss,
    and the symmetry plane/axis regression terms. Both branches use the
    default `reduction='mean'` so the returned scalar is consistent with
    the per-step / epoch-average logging pipeline.

    Args:
        kind:  "mse" | "smooth_l1"
        beta:  SmoothL1 transition point (ignored when kind == "mse")
    """
    if kind == "mse":
        return F.mse_loss(predicted, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(predicted, target, beta=beta)
    raise ValueError(
        f"Unknown regression loss kind: {kind!r}. Expected 'mse' or 'smooth_l1'."
    )


def compute_raw_geo_stats_dim(geo_cfg: GeoEmbedConfig) -> int:
    """Compute the per-token raw_geo_stats dimension the encoder will emit.

    Must stay in sync with `GeoEmbed.__init__` in
    `shape_foundation/models/tokenizer_magno.py`: relative positions
    contribute 3 dims per stat, optional normals add 3 per stat, optional
    curvature adds 1 per stat. Used by LossComputer to build the
    reconstruction projection head eagerly (before optimizer construction).
    """
    n_stats = len(geo_cfg.stat_features)
    dim = n_stats * 3
    if geo_cfg.augment_normals:
        dim += n_stats * 3
    if geo_cfg.augment_curvature:
        dim += n_stats * 1
    return dim


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

class MaskedTokenLoss(nn.Module):
    """Predict geometry statistics at masked latent tokens.

    The mask is created externally (in the trainer) and passed into both
    the model forward and this loss so the same positions are masked and
    reconstructed.
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.mask_ratio = cfg.masked_token.mask_ratio
        self.mask_strategy = cfg.masked_token.mask_strategy
        self.block_size = cfg.masked_token.block_size
        self.reg_kind = cfg.regression.kind
        self.reg_beta = cfg.regression.beta
        self._grid_shape: tuple[int, int, int] | None = None

    def set_grid_shape(self, grid_shape: tuple[int, int, int]) -> None:
        self._grid_shape = grid_shape

    def create_mask(
        self, B: int, T: int, device: torch.device,
    ) -> torch.Tensor:
        """Create a boolean mask (True = masked).

        Strategies:
            random: uniform random masking
            block: mask contiguous linear blocks
            hybrid: mix of random and block
            spatial_3d: mask cubic regions in the 3D latent grid
        """
        n_mask = int(T * self.mask_ratio)

        if self.mask_strategy == "random":
            mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for b in range(B):
                idx = torch.randperm(T, device=device)[:n_mask]
                mask[b, idx] = True
            return mask

        elif self.mask_strategy == "block":
            mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            block = self.block_size ** 3
            n_blocks = max(1, n_mask // block)
            for b in range(B):
                starts = torch.randperm(max(1, T - block), device=device)[:n_blocks]
                for s in starts:
                    mask[b, s:s + block] = True
            return mask

        elif self.mask_strategy == "spatial_3d":
            # Mask cubic regions in the 3D latent grid
            if self._grid_shape is None:
                raise RuntimeError(
                    "spatial_3d mask strategy requires grid_shape. "
                    "Call set_grid_shape() before creating masks."
                )
            H, W, D = self._grid_shape
            assert T == H * W * D, f"T={T} != H*W*D={H*W*D}"
            mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            # cube side length targeting mask_ratio volume fraction
            cube_side = max(1, int(round((self.mask_ratio * H * W * D) ** (1 / 3))))
            cube_side = min(cube_side, H - 1, W - 1, D - 1)
            # number of cubes to reach approximate mask_ratio
            cube_vol = cube_side ** 3
            n_cubes = max(1, n_mask // cube_vol)
            for b in range(B):
                for _ in range(n_cubes):
                    sh = torch.randint(0, H - cube_side + 1, (1,)).item()
                    sw = torch.randint(0, W - cube_side + 1, (1,)).item()
                    sd = torch.randint(0, D - cube_side + 1, (1,)).item()
                    for h in range(sh, sh + cube_side):
                        for w in range(sw, sw + cube_side):
                            for d in range(sd, sd + cube_side):
                                mask[b, h * W * D + w * D + d] = True
            return mask

        else:  # hybrid
            mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            n_random = n_mask // 2
            n_block = n_mask - n_random
            block = self.block_size ** 3
            n_blocks = max(1, n_block // block)
            for b in range(B):
                idx = torch.randperm(T, device=device)[:n_random]
                mask[b, idx] = True
                starts = torch.randperm(max(1, T - block), device=device)[:n_blocks]
                for s in starts:
                    mask[b, s:s + block] = True
            return mask

    def forward(
        self,
        predicted: torch.Tensor,  # (B, T, D) projected predictions
        target: torch.Tensor,     # (B, T, D) geo stats from encoder
        mask: torch.Tensor,       # (B, T) bool
    ) -> torch.Tensor:
        """Regression loss at masked positions only (kind from cfg.regression)."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predicted.device)
        return regression_loss(
            predicted[mask], target[mask], self.reg_kind, self.reg_beta,
        )


class MultiResContrastiveLoss(nn.Module):
    """Same mesh under different samplings should embed similarly (InfoNCE)."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.temperature = cfg.contrastive.temperature

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        a = F.normalize(embeddings_a, dim=-1)
        b = F.normalize(embeddings_b, dim=-1)
        logits_ab = a @ b.T / self.temperature
        logits_ba = b @ a.T / self.temperature
        labels = torch.arange(a.shape[0], device=a.device)
        return (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels)) / 2


class PartialInpaintingLoss(nn.Module):
    """Crop a spatial region and predict masked-token targets there."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.crop_ratio = cfg.inpainting.crop_ratio
        self.crop_strategy = cfg.inpainting.crop_strategy
        self.reg_kind = cfg.regression.kind
        self.reg_beta = cfg.regression.beta
        self._grid_shape: tuple[int, int, int] | None = None

    def set_grid_shape(self, grid_shape: tuple[int, int, int]) -> None:
        self._grid_shape = grid_shape

    def create_spatial_mask(
        self, B: int, device: torch.device,
    ) -> torch.Tensor:
        """Create a spatial crop mask on the latent grid.

        Requires grid_shape to have been set via set_grid_shape().
        """
        if self._grid_shape is None:
            raise RuntimeError(
                "PartialInpaintingLoss requires grid_shape for spatial masking. "
                "Call set_grid_shape() before creating masks."
            )
        H, W, D = self._grid_shape
        T = H * W * D
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        crop_size = int(round(self.crop_ratio ** (1 / 3) * H))
        crop_size = max(1, min(crop_size, H - 1))

        for b in range(B):
            if self.crop_strategy == "spatial":
                sh = torch.randint(0, H - crop_size + 1, (1,)).item()
                sw = torch.randint(0, W - crop_size + 1, (1,)).item()
                sd = torch.randint(0, D - crop_size + 1, (1,)).item()
                for h in range(sh, sh + crop_size):
                    for w in range(sw, sw + crop_size):
                        for d in range(sd, sd + crop_size):
                            idx = h * W * D + w * D + d
                            mask[b, idx] = True
            else:
                n_mask = int(T * self.crop_ratio)
                idx = torch.randperm(T, device=device)[:n_mask]
                mask[b, idx] = True

        return mask

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predicted.device)
        return regression_loss(
            predicted[mask], target[mask], self.reg_kind, self.reg_beta,
        )


class SymmetryLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cls_weight = cfg.supervised.symmetry_weight
        self.plane_weight = cfg.supervised.plane_regression_weight
        self.axis_weight = cfg.supervised.axis_regression_weight
        self.reg_kind = cfg.regression.kind
        self.reg_beta = cfg.regression.beta

    def forward(self, head_out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=head_out["logits"].device)
        if "symmetry_label" in labels:
            # Classification term — unchanged.
            loss = loss + self.cls_weight * F.cross_entropy(head_out["logits"], labels["symmetry_label"])
        if "symmetry_planes" in labels and "planes" in head_out:
            tp, pp = labels["symmetry_planes"], head_out["planes"]
            if tp.ndim == 3 and pp.ndim == 3:
                mb, mk = min(tp.shape[0], pp.shape[0]), min(tp.shape[1], pp.shape[1])
                if mb > 0 and mk > 0:
                    loss = loss + self.plane_weight * regression_loss(
                        pp[:mb, :mk], tp[:mb, :mk], self.reg_kind, self.reg_beta,
                    )
        if "symmetry_axes" in labels and "axes" in head_out:
            ta, pa = labels["symmetry_axes"], head_out["axes"]
            if ta.ndim == 3 and pa.ndim == 3:
                mb, mk = min(ta.shape[0], pa.shape[0]), min(ta.shape[1], pa.shape[1])
                if mb > 0 and mk > 0:
                    loss = loss + self.axis_weight * regression_loss(
                        pa[:mb, :mk], ta[:mb, :mk], self.reg_kind, self.reg_beta,
                    )
        return loss


class PrimitiveLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.weight = cfg.supervised.primitive_weight

    def forward(self, head_out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
        if "primitive_labels" not in labels:
            return torch.tensor(0.0, device=head_out["primitive_logits"].device)
        logits = head_out["primitive_logits"]
        target = labels["primitive_labels"]
        if target.shape[1] != logits.shape[1]:
            return torch.tensor(0.0, device=logits.device)
        return self.weight * F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-1)


class PartLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.weight = cfg.supervised.part_weight

    def forward(self, head_out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
        if "part_labels" not in labels:
            return torch.tensor(0.0, device=head_out["part_logits"].device)
        logits = head_out["part_logits"]
        target = labels["part_labels"]
        if target.shape[1] != logits.shape[1]:
            return torch.tensor(0.0, device=logits.device)
        return self.weight * F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-1)


class ReductionLoss(nn.Module):
    def forward(self, head_out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
        if "reduction_label" not in labels:
            return torch.tensor(0.0, device=head_out["reduction_logits"].device)
        return F.cross_entropy(head_out["reduction_logits"], labels["reduction_label"])


class TextAlignLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.temperature = cfg.text_align.temperature

    def forward(self, geo_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        g = F.normalize(geo_embed, dim=-1)
        t = F.normalize(text_embed, dim=-1)
        logits = g @ t.T / self.temperature
        labels = torch.arange(g.shape[0], device=g.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ---------------------------------------------------------------------------
# Combined loss computer
# ---------------------------------------------------------------------------

class LossComputer(nn.Module):
    """Orchestrates all loss functions.

    Key design: the masked-token mask is created externally (in the trainer)
    and passed into both the model forward and this loss computer via
    the `token_mask` argument, ensuring the same positions are masked
    during forward and used for reconstruction loss.

    All trainable submodules (including the reconstruction projection head
    used by masked-token and inpainting losses) are built in __init__ so
    every parameter is registered with the optimizer from step 0.
    """

    def __init__(
        self,
        cfg: LossConfig,
        token_dim: int,
        recon_target_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.weights = cfg.weights
        self.token_dim = token_dim
        self.recon_target_dim = recon_target_dim

        self.masked_token = MaskedTokenLoss(cfg) if cfg.masked_token.enabled else None
        self.contrastive = MultiResContrastiveLoss(cfg) if cfg.contrastive.enabled else None
        self.inpainting = PartialInpaintingLoss(cfg) if cfg.inpainting.enabled else None
        self.symmetry = SymmetryLoss(cfg)
        self.primitive = PrimitiveLoss(cfg)
        self.part = PartLoss(cfg)
        self.reduction = ReductionLoss()
        self.text_align = TextAlignLoss(cfg) if cfg.text_align.enabled else None

        # Reconstruction projection head — built eagerly whenever either of
        # the reconstruction-based losses is enabled. This keeps all trainable
        # parameters visible to the optimizer at construction time and avoids
        # silent first-forward initialization.
        if cfg.masked_token.enabled or cfg.inpainting.enabled:
            hidden = token_dim * 2
            self.recon_proj: nn.Module | None = nn.Sequential(
                nn.Linear(token_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, recon_target_dim),
            )
        else:
            self.recon_proj = None

        # Per-dimension normalization stats for raw_geo_stats reconstruction
        # targets. Registered as buffers so they are saved in state_dict and
        # survive checkpoint save/load. Initialized to identity (mean=0,
        # std=1, calibrated=0) so that if calibration never runs, targets
        # pass through unchanged — this is the backward-compat path.
        self.recon_norm_enabled = cfg.recon_target_norm.enabled
        self.recon_norm_eps = cfg.recon_target_norm.eps
        D = max(1, int(recon_target_dim))
        self.register_buffer(
            "recon_target_mean", torch.zeros(D, dtype=torch.float32),
        )
        self.register_buffer(
            "recon_target_std", torch.ones(D, dtype=torch.float32),
        )
        # uint8 {0, 1} flag — whether buffers above hold valid calibrated
        # stats. Checked at forward time to decide pass-through vs normalize.
        self.register_buffer(
            "recon_target_calibrated", torch.zeros(1, dtype=torch.uint8),
        )

        # Tracks which supervised losses have already fired the
        # "enabled in config but labels missing from batch" warning, so
        # we warn once per loss name instead of spamming every step.
        self._missing_label_warned: set[str] = set()

    def set_grid_shape(self, grid_shape: tuple[int, int, int]) -> None:
        """Inject latent grid shape for spatial masking strategies."""
        self._grid_shape = grid_shape
        if self.masked_token is not None:
            self.masked_token.set_grid_shape(grid_shape)
        if self.inpainting is not None:
            self.inpainting.set_grid_shape(grid_shape)

    def is_recon_target_calibrated(self) -> bool:
        """Whether `recon_target_mean`/`_std` currently hold valid stats.

        Set to True after a successful calibration pass, or after loading
        a checkpoint that was saved with calibrated buffers. Read by the
        trainer to decide whether to run a calibration pass.
        """
        return bool(self.recon_target_calibrated.item())

    def set_recon_target_stats(
        self, mean: torch.Tensor, std: torch.Tensor,
    ) -> None:
        """Write calibrated per-dim stats into the registered buffers.

        The two tensors must have shape (recon_target_dim,). After this
        call, `is_recon_target_calibrated()` returns True and
        `_maybe_normalize_target` will apply `(x - mean) / (std + eps)`
        to the reconstruction targets.
        """
        assert mean.shape == self.recon_target_mean.shape, (
            f"mean shape {tuple(mean.shape)} != buffer shape "
            f"{tuple(self.recon_target_mean.shape)}"
        )
        assert std.shape == self.recon_target_std.shape
        self.recon_target_mean.data.copy_(mean.to(self.recon_target_mean.dtype))
        self.recon_target_std.data.copy_(std.to(self.recon_target_std.dtype))
        self.recon_target_calibrated.data.fill_(1)

    def _check_supervised_labels(
        self,
        name: str,
        batch: dict[str, torch.Tensor],
        required_any_of: tuple[str, ...],
    ) -> bool:
        """Return True if any required label for `name` is in the batch.

        Fires a one-time warning if the loss has weight > 0 in the config
        but none of the expected label keys are present — this catches the
        silent zero-loss failure mode where a head is enabled but the
        dataset never provides labels for it, so loss silently contributes
        0 and the head never trains. Weight == 0 losses don't warn because
        zero-weighted losses are a legitimate "disabled" pattern.
        """
        if any(k in batch for k in required_any_of):
            return True
        weight = float(getattr(self.weights, name, 0.0))
        if weight > 0.0 and name not in self._missing_label_warned:
            print(
                f"[LossComputer] WARNING: '{name}' loss has weight={weight} "
                f"but none of {required_any_of} are present in the batch. "
                f"Loss will contribute 0.0 every step — check that your "
                f"dataset is actually emitting these labels."
            )
            self._missing_label_warned.add(name)
        return False

    def _maybe_normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        """Apply per-dim normalization to a reconstruction target.

        Pass-through if normalization is disabled in the config OR if the
        stats have not been calibrated yet. The normalization is applied
        in the target's own dtype (bf16/fp32 autocast-safe) by casting the
        fp32 buffers into the target's dtype before the arithmetic.
        """
        if not self.recon_norm_enabled:
            return target
        if not self.is_recon_target_calibrated():
            return target
        mean = self.recon_target_mean.to(dtype=target.dtype)
        std = self.recon_target_std.to(dtype=target.dtype)
        return (target - mean) / (std + self.recon_norm_eps)

    def forward(
        self,
        model_out: dict,
        batch: dict[str, torch.Tensor],
        model_out_aug: dict | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all applicable losses.

        Returns a flat dict keyed by stable names. For every active loss
        component `<name>`, the result contains both `<name>_raw` (the
        unweighted value, detached) and `<name>_weighted` (value * cfg
        weight, detached). `total` is the sum of all `_weighted` values
        and is the exact tensor used for .backward().

        Args:
            model_out: output from backbone.forward_features()
            batch: data batch with optional labels
            model_out_aug: second augmentation output for contrastive loss
            token_mask: (B, T) bool mask created before forward, True = masked.
                        Used for masked-token reconstruction loss.
        """
        losses: dict[str, torch.Tensor] = {}
        device = model_out["token_embeddings"].device

        # --- Sanity-check recon dims against the actual encoder output. ---
        # recon_proj was built in __init__; if it exists and the encoder is
        # emitting raw stats, dims must line up with what we allocated.
        if self.recon_proj is not None and "raw_geo_stats" in model_out:
            got_token_dim = model_out["token_embeddings"].shape[-1]
            got_target_dim = model_out["raw_geo_stats"].shape[-1]
            if got_token_dim != self.token_dim or got_target_dim != self.recon_target_dim:
                raise RuntimeError(
                    "LossComputer.recon_proj dims do not match the encoder: "
                    f"configured token_dim={self.token_dim}, target_dim={self.recon_target_dim}; "
                    f"got token_dim={got_token_dim}, target_dim={got_target_dim}. "
                    "Rebuild LossComputer with the correct dims before training."
                )

        # --- Masked token loss (uses externally-created mask) ---
        if (
            self.masked_token is not None
            and "raw_geo_stats" in model_out
            and token_mask is not None
            and self.recon_proj is not None
        ):
            target = self._maybe_normalize_target(model_out["raw_geo_stats"])
            predicted = self.recon_proj(model_out["token_embeddings"])
            losses["masked_token"] = self.masked_token(predicted, target, token_mask)

        # --- Contrastive loss ---
        if self.contrastive is not None and model_out_aug is not None:
            losses["contrastive"] = self.contrastive(
                model_out["pooled_embedding"],
                model_out_aug["pooled_embedding"],
            )

        # --- Inpainting loss (separate spatial mask, independent of token_mask) ---
        if (
            self.inpainting is not None
            and "raw_geo_stats" in model_out
            and self.recon_proj is not None
        ):
            target = self._maybe_normalize_target(model_out["raw_geo_stats"])
            predicted = self.recon_proj(model_out["token_embeddings"])
            B = predicted.shape[0]
            inpaint_mask = self.inpainting.create_spatial_mask(B, device)
            losses["inpainting"] = self.inpainting(predicted, target, inpaint_mask)

        # --- Supervised losses ---
        # Each branch is gated on (a) its head being in the model output
        # and (b) the required labels being in the batch. The second check
        # warns once if the loss is expected but no labels are provided,
        # preventing the old silent-zero failure mode.
        heads = model_out.get("heads", {})

        if "symmetry" in heads and self._check_supervised_labels(
            "symmetry", batch,
            ("symmetry_label", "symmetry_planes", "symmetry_axes"),
        ):
            sym_loss = self.symmetry(heads["symmetry"], batch)
            if sym_loss.item() > 0:
                losses["symmetry"] = sym_loss

        if "primitive" in heads and self._check_supervised_labels(
            "primitive", batch, ("primitive_labels",),
        ):
            prim_loss = self.primitive(heads["primitive"], batch)
            if prim_loss.item() > 0:
                losses["primitive"] = prim_loss

        if "part" in heads and self._check_supervised_labels(
            "part", batch, ("part_labels",),
        ):
            part_loss = self.part(heads["part"], batch)
            if part_loss.item() > 0:
                losses["part"] = part_loss

        if "reduction" in heads and self._check_supervised_labels(
            "reduction", batch, ("reduction_label",),
        ):
            red_loss = self.reduction(heads["reduction"], batch)
            if red_loss.item() > 0:
                losses["reduction"] = red_loss

        # --- Text alignment ---
        if self.text_align is not None and "text_embedding" in batch:
            losses["text_align"] = self.text_align(
                model_out["pooled_embedding"], batch["text_embedding"],
            )

        # --- Apply per-loss weights and compute total ---
        # Emit both *_raw (unweighted) and *_weighted for every active loss so
        # per-step logging, epoch averages, and `total` all use the same
        # definitions. `total` is exactly the loss used for .backward().
        result: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name, val in losses.items():
            w = float(getattr(self.weights, name, 1.0))
            weighted_val = val * w
            result[f"{name}_raw"] = val.detach()
            result[f"{name}_weighted"] = weighted_val.detach()
            total = weighted_val if total is None else total + weighted_val

        if total is None:
            total = torch.tensor(0.0, device=device)
        result["total"] = total
        return result
