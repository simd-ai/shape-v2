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

from shape_foundation.configs.default import LossConfig


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
        """MSE loss at masked positions only."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predicted.device)
        return F.mse_loss(predicted[mask], target[mask])


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
        return F.mse_loss(predicted[mask], target[mask])


class SymmetryLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cls_weight = cfg.supervised.symmetry_weight
        self.plane_weight = cfg.supervised.plane_regression_weight
        self.axis_weight = cfg.supervised.axis_regression_weight

    def forward(self, head_out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=head_out["logits"].device)
        if "symmetry_label" in labels:
            loss = loss + self.cls_weight * F.cross_entropy(head_out["logits"], labels["symmetry_label"])
        if "symmetry_planes" in labels and "planes" in head_out:
            tp, pp = labels["symmetry_planes"], head_out["planes"]
            if tp.ndim == 3 and pp.ndim == 3:
                mb, mk = min(tp.shape[0], pp.shape[0]), min(tp.shape[1], pp.shape[1])
                if mb > 0 and mk > 0:
                    loss = loss + self.plane_weight * F.l1_loss(pp[:mb, :mk], tp[:mb, :mk])
        if "symmetry_axes" in labels and "axes" in head_out:
            ta, pa = labels["symmetry_axes"], head_out["axes"]
            if ta.ndim == 3 and pa.ndim == 3:
                mb, mk = min(ta.shape[0], pa.shape[0]), min(ta.shape[1], pa.shape[1])
                if mb > 0 and mk > 0:
                    loss = loss + self.axis_weight * F.l1_loss(pa[:mb, :mk], ta[:mb, :mk])
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
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.weights = cfg.weights
        self.masked_token = MaskedTokenLoss(cfg) if cfg.masked_token.enabled else None
        self.contrastive = MultiResContrastiveLoss(cfg) if cfg.contrastive.enabled else None
        self.inpainting = PartialInpaintingLoss(cfg) if cfg.inpainting.enabled else None
        self.symmetry = SymmetryLoss(cfg)
        self.primitive = PrimitiveLoss(cfg)
        self.part = PartLoss(cfg)
        self.reduction = ReductionLoss()
        self.text_align = TextAlignLoss(cfg) if cfg.text_align.enabled else None

        # Reconstruction projection head (lazily built on first forward)
        self.recon_proj = None

    def _ensure_recon_proj(self, token_dim: int, target_dim: int, device: torch.device) -> None:
        """Build a deeper projection head on first forward."""
        if self.recon_proj is None:
            self.recon_proj = nn.Sequential(
                nn.Linear(token_dim, token_dim * 2),
                nn.GELU(),
                nn.Linear(token_dim * 2, token_dim * 2),
                nn.GELU(),
                nn.Linear(token_dim * 2, target_dim),
            ).to(device)

    def set_grid_shape(self, grid_shape: tuple[int, int, int]) -> None:
        """Inject latent grid shape for spatial masking strategies."""
        self._grid_shape = grid_shape
        if self.masked_token is not None:
            self.masked_token.set_grid_shape(grid_shape)
        if self.inpainting is not None:
            self.inpainting.set_grid_shape(grid_shape)

    def forward(
        self,
        model_out: dict,
        batch: dict[str, torch.Tensor],
        model_out_aug: dict | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all applicable losses.

        Args:
            model_out: output from backbone.forward_features()
            batch: data batch with optional labels
            model_out_aug: second augmentation output for contrastive loss
            token_mask: (B, T) bool mask created before forward, True = masked.
                        Used for masked-token reconstruction loss.
        """
        losses: dict[str, torch.Tensor] = {}
        device = model_out["token_embeddings"].device

        # --- Build reconstruction projection if needed ---
        if "raw_geo_stats" in model_out:
            token_dim = model_out["token_embeddings"].shape[-1]
            target_dim = model_out["raw_geo_stats"].shape[-1]
            self._ensure_recon_proj(token_dim, target_dim, device)

        # --- Masked token loss (uses externally-created mask) ---
        if self.masked_token is not None and "raw_geo_stats" in model_out and token_mask is not None:
            target = model_out["raw_geo_stats"]
            predicted = self.recon_proj(model_out["token_embeddings"])
            losses["masked_token"] = self.masked_token(predicted, target, token_mask)

        # --- Contrastive loss ---
        if self.contrastive is not None and model_out_aug is not None:
            losses["contrastive"] = self.contrastive(
                model_out["pooled_embedding"],
                model_out_aug["pooled_embedding"],
            )

        # --- Inpainting loss (separate spatial mask, independent of token_mask) ---
        if self.inpainting is not None and "raw_geo_stats" in model_out:
            target = model_out["raw_geo_stats"]
            predicted = self.recon_proj(model_out["token_embeddings"])
            B = predicted.shape[0]
            inpaint_mask = self.inpainting.create_spatial_mask(B, device)
            losses["inpainting"] = self.inpainting(predicted, target, inpaint_mask)

        # --- Supervised losses ---
        heads = model_out.get("heads", {})

        if "symmetry" in heads:
            sym_loss = self.symmetry(heads["symmetry"], batch)
            if sym_loss.item() > 0:
                losses["symmetry"] = sym_loss

        if "primitive" in heads:
            prim_loss = self.primitive(heads["primitive"], batch)
            if prim_loss.item() > 0:
                losses["primitive"] = prim_loss

        if "part" in heads:
            part_loss = self.part(heads["part"], batch)
            if part_loss.item() > 0:
                losses["part"] = part_loss

        if "reduction" in heads:
            red_loss = self.reduction(heads["reduction"], batch)
            if red_loss.item() > 0:
                losses["reduction"] = red_loss

        # --- Text alignment ---
        if self.text_align is not None and "text_embedding" in batch:
            losses["text_align"] = self.text_align(
                model_out["pooled_embedding"], batch["text_embedding"],
            )

        # --- Apply per-loss weights and compute total ---
        weighted: dict[str, torch.Tensor] = {}
        for name, val in losses.items():
            w = getattr(self.weights, name, 1.0)
            weighted[name] = val * w

        total = sum(weighted.values()) if weighted else torch.tensor(0.0, device=device)
        # Return unweighted individual losses for logging, plus weighted total
        losses["total"] = total
        return losses
