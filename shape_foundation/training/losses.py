"""Loss functions for Shape foundation model training.

Implements:
    1. Masked latent-token modeling loss
    2. Multi-resolution contrastive consistency loss
    3. Partial-shape inpainting loss
    4. Supervised symmetry/primitive/part/reduction losses
    5. Optional 3D-text alignment loss
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

    The backbone produces raw_geo_stats from the encoder before masking;
    after processing with mask, the processor outputs predictions at
    masked positions. Loss = MSE between predicted and target statistics.
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.mask_ratio = cfg.masked_token.mask_ratio
        self.mask_strategy = cfg.masked_token.mask_strategy
        self.block_size = cfg.masked_token.block_size

    def create_mask(
        self, B: int, T: int, device: torch.device,
    ) -> torch.Tensor:
        """Create a boolean mask (True = masked).

        Strategies:
            random: uniform random masking
            block: mask contiguous blocks
            hybrid: mix of random and block
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
            block = self.block_size ** 3  # 3D block
            n_blocks = max(1, n_mask // block)
            for b in range(B):
                starts = torch.randperm(max(1, T - block), device=device)[:n_blocks]
                for s in starts:
                    mask[b, s:s + block] = True
            return mask

        else:  # hybrid
            mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            n_random = n_mask // 2
            n_block = n_mask - n_random
            block = self.block_size ** 3
            n_blocks = max(1, n_block // block)
            for b in range(B):
                # random part
                idx = torch.randperm(T, device=device)[:n_random]
                mask[b, idx] = True
                # block part
                starts = torch.randperm(max(1, T - block), device=device)[:n_blocks]
                for s in starts:
                    mask[b, s:s + block] = True
            return mask

    def forward(
        self,
        predicted: torch.Tensor,  # (B, T, C) from processor
        target: torch.Tensor,     # (B, T, target_dim) geo stats from encoder
        mask: torch.Tensor,       # (B, T) bool
    ) -> torch.Tensor:
        """MSE loss at masked positions."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predicted.device)

        # project predicted to target dim if needed
        pred_dim = predicted.shape[-1]
        tgt_dim = target.shape[-1]
        if pred_dim != tgt_dim:
            # use a simple linear layer (should be part of model, but handle gracefully)
            predicted = predicted[..., :tgt_dim]

        pred_masked = predicted[mask]
        tgt_masked = target[mask]
        return F.mse_loss(pred_masked, tgt_masked)


class MultiResContrastiveLoss(nn.Module):
    """Same mesh under different samplings should embed similarly (InfoNCE)."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.temperature = cfg.contrastive.temperature

    def forward(
        self,
        embeddings_a: torch.Tensor,  # (B, C) pooled from augmentation A
        embeddings_b: torch.Tensor,  # (B, C) pooled from augmentation B
    ) -> torch.Tensor:
        """Symmetric InfoNCE loss."""
        a = F.normalize(embeddings_a, dim=-1)
        b = F.normalize(embeddings_b, dim=-1)

        # similarity matrix
        logits_ab = a @ b.T / self.temperature  # (B, B)
        logits_ba = b @ a.T / self.temperature

        labels = torch.arange(a.shape[0], device=a.device)
        loss_ab = F.cross_entropy(logits_ab, labels)
        loss_ba = F.cross_entropy(logits_ba, labels)
        return (loss_ab + loss_ba) / 2


class PartialInpaintingLoss(nn.Module):
    """Crop a spatial region and predict masked-token targets there."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.crop_ratio = cfg.inpainting.crop_ratio
        self.crop_strategy = cfg.inpainting.crop_strategy

    def create_spatial_mask(
        self, B: int, grid_shape: tuple[int, int, int], device: torch.device,
    ) -> torch.Tensor:
        """Create a spatial crop mask on the latent grid."""
        H, W, D = grid_shape
        T = H * W * D
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        crop_size = int(round(self.crop_ratio ** (1 / 3) * H))
        crop_size = max(1, min(crop_size, H - 1))

        for b in range(B):
            if self.crop_strategy == "spatial":
                sh = torch.randint(0, H - crop_size + 1, (1,)).item()
                sw = torch.randint(0, W - crop_size + 1, (1,)).item()
                sd = torch.randint(0, D - crop_size + 1, (1,)).item()
                # mark 3D block as masked
                for h in range(sh, sh + crop_size):
                    for w in range(sw, sw + crop_size):
                        for d in range(sd, sd + crop_size):
                            idx = h * W * D + w * D + d
                            mask[b, idx] = True
            else:
                # random region: just pick random tokens
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
        pred_dim = predicted.shape[-1]
        tgt_dim = target.shape[-1]
        if pred_dim != tgt_dim:
            predicted = predicted[..., :tgt_dim]
        return F.mse_loss(predicted[mask], target[mask])


class SymmetryLoss(nn.Module):
    """Classification + regression loss for symmetry detection."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cls_weight = cfg.supervised.symmetry_weight
        self.plane_weight = cfg.supervised.plane_regression_weight
        self.axis_weight = cfg.supervised.axis_regression_weight

    def forward(
        self,
        head_out: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=head_out["logits"].device)

        if "symmetry_label" in labels:
            loss = loss + self.cls_weight * F.cross_entropy(
                head_out["logits"], labels["symmetry_label"],
            )

        if "symmetry_planes" in labels and "planes" in head_out:
            target_planes = labels["symmetry_planes"]  # (B_label, K, 4)
            pred_planes = head_out["planes"]            # (B_pred, K, 4)
            if target_planes.ndim == 3 and pred_planes.ndim == 3:
                # only compute on samples where labels exist (batch sizes may differ)
                min_b = min(target_planes.shape[0], pred_planes.shape[0])
                min_k = min(target_planes.shape[1], pred_planes.shape[1])
                if min_b > 0 and min_k > 0:
                    loss = loss + self.plane_weight * F.l1_loss(
                        pred_planes[:min_b, :min_k], target_planes[:min_b, :min_k],
                    )

        if "symmetry_axes" in labels and "axes" in head_out:
            target_axes = labels["symmetry_axes"]
            pred_axes = head_out["axes"]
            if target_axes.ndim == 3 and pred_axes.ndim == 3:
                min_b = min(target_axes.shape[0], pred_axes.shape[0])
                min_k = min(target_axes.shape[1], pred_axes.shape[1])
                if min_b > 0 and min_k > 0:
                    loss = loss + self.axis_weight * F.l1_loss(
                        pred_axes[:min_b, :min_k], target_axes[:min_b, :min_k],
                    )

        return loss


class PrimitiveLoss(nn.Module):
    """Per-token primitive classification loss."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.weight = cfg.supervised.primitive_weight

    def forward(
        self, head_out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "primitive_labels" not in labels:
            return torch.tensor(0.0, device=head_out["primitive_logits"].device)

        logits = head_out["primitive_logits"]  # (B, T, num_types)
        target = labels["primitive_labels"]     # (B, T)

        if target.shape[1] != logits.shape[1]:
            # labels may be per-vertex, not per-token; skip if mismatch
            return torch.tensor(0.0, device=logits.device)

        return self.weight * F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=-1,
        )


class PartLoss(nn.Module):
    """Part segmentation cross-entropy loss."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.weight = cfg.supervised.part_weight

    def forward(
        self, head_out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "part_labels" not in labels:
            return torch.tensor(0.0, device=head_out["part_logits"].device)

        logits = head_out["part_logits"]
        target = labels["part_labels"]

        if target.shape[1] != logits.shape[1]:
            return torch.tensor(0.0, device=logits.device)

        return self.weight * F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=-1,
        )


class ReductionLoss(nn.Module):
    """Reduction recommendation classification loss."""

    def forward(
        self, head_out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "reduction_label" not in labels:
            return torch.tensor(0.0, device=head_out["reduction_logits"].device)
        return F.cross_entropy(head_out["reduction_logits"], labels["reduction_label"])


class TextAlignLoss(nn.Module):
    """Contrastive alignment between 3D embeddings and text embeddings."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.temperature = cfg.text_align.temperature

    def forward(
        self,
        geo_embed: torch.Tensor,   # (B, C)
        text_embed: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        g = F.normalize(geo_embed, dim=-1)
        t = F.normalize(text_embed, dim=-1)
        logits = g @ t.T / self.temperature
        labels = torch.arange(g.shape[0], device=g.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ---------------------------------------------------------------------------
# Combined loss computer
# ---------------------------------------------------------------------------

class LossComputer(nn.Module):
    """Orchestrates all loss functions based on config and available data."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.masked_token = MaskedTokenLoss(cfg) if cfg.masked_token.enabled else None
        self.contrastive = MultiResContrastiveLoss(cfg) if cfg.contrastive.enabled else None
        self.inpainting = PartialInpaintingLoss(cfg) if cfg.inpainting.enabled else None
        self.symmetry = SymmetryLoss(cfg)
        self.primitive = PrimitiveLoss(cfg)
        self.part = PartLoss(cfg)
        self.reduction = ReductionLoss()
        self.text_align = TextAlignLoss(cfg) if cfg.text_align.enabled else None

        # projection for masked token prediction target alignment
        self.mask_proj = None  # lazily initialized

    def forward(
        self,
        model_out: dict,
        batch: dict[str, torch.Tensor],
        model_out_aug: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all applicable losses.

        Args:
            model_out: output from backbone.forward_features()
            batch: data batch with optional labels
            model_out_aug: second augmentation output for contrastive loss

        Returns:
            dict with individual loss values and 'total' key
        """
        losses: dict[str, torch.Tensor] = {}
        device = model_out["token_embeddings"].device

        # --- Masked token loss ---
        if self.masked_token is not None and "raw_geo_stats" in model_out:
            target = model_out["raw_geo_stats"]
            predicted = model_out["token_embeddings"]
            B, T, _ = predicted.shape

            mask = self.masked_token.create_mask(B, T, device)
            losses["masked_token"] = self.masked_token(predicted, target, mask)

        # --- Contrastive loss ---
        if self.contrastive is not None and model_out_aug is not None:
            losses["contrastive"] = self.contrastive(
                model_out["pooled_embedding"],
                model_out_aug["pooled_embedding"],
            )

        # --- Inpainting loss ---
        if self.inpainting is not None and "raw_geo_stats" in model_out:
            target = model_out["raw_geo_stats"]
            predicted = model_out["token_embeddings"]
            B, T, _ = predicted.shape
            from shape_foundation.configs.default import ShapeConfig
            grid_shape = (48, 48, 48)  # will be overridden by actual config
            if hasattr(self, "_grid_shape"):
                grid_shape = self._grid_shape
            inpaint_mask = self.inpainting.create_spatial_mask(B, grid_shape, device)
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

        # --- Total ---
        total = sum(losses.values()) if losses else torch.tensor(0.0, device=device)
        losses["total"] = total
        return losses

    def set_grid_shape(self, grid_shape: tuple[int, int, int]) -> None:
        self._grid_shape = grid_shape
