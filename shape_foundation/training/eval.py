"""Self-supervised evaluation suite for the Shape foundation model.

The evaluator measures the model along the axes it was actually trained on:
masked-token reconstruction and multi-resolution contrastive consistency.
All metrics are computed in the same normalized target space used during
pretraining (per-dim z-scoring calibrated on the training split), so the
reported numbers are directly comparable to the training loss curves.

Metrics
-------
Reconstruction (self-supervised objective)
    recon_smoothl1   SmoothL1 loss (β = 1.0) at masked positions, in normalized
                     target space. Directly comparable to `train/masked_token_raw`.
    recon_mse        Mean squared error at masked positions (normalized space).
    recon_r2         Coefficient of determination on normalized targets;
                     interpretable scale-free measure of how much target
                     variance the reconstruction captures.

Contrastive (embedding geometry; Wang & Isola, 2020)
    contrastive_alignment   E[‖f(x) − f(y)‖²] over positive pairs (same mesh,
                            two augmentations). Lower = augmentation-invariant.
    contrastive_uniformity  log E[exp(−t ‖f(x) − f(y)‖²)] over random pairs,
                            t = 2. More negative = more uniformly spread
                            embedding distribution on the unit sphere.
    contrastive_infonce     Symmetric InfoNCE loss (τ = 0.07) between clean
                            and augmented views. Matches the training objective.
    contrastive_top1_acc    Fraction of queries whose positive pair is the
                            top-1 nearest neighbor under cosine similarity.

Embedding geometry (descriptive, not a quality score)
    embedding_norm_mean/std  Distribution of pre-normalization embedding norms.
    pairwise_cosine_mean/std Random-pair cosine similarity distribution; close
                             to zero indicates well-spread embeddings.

Robustness (via `eval_robustness`)
    robustness_noise_{σ}       Mean cosine agreement between clean and jittered
                               pooled embeddings at Gaussian noise σ.
    robustness_decimate_{r}    Mean cosine agreement under random point
                               dropout keeping fraction r of surface points.

Supervised task heads (symmetry, primitive, part, reduction) are intentionally
NOT evaluated: they are trained with weight 0.0 in the v3 release because the
stock synthetic labels do not generalize. Any accuracy number on those heads
would reflect random initialization only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from shape_foundation.models.gaot_backbone import GAOTBackbone


class Evaluator:
    """Self-supervised evaluation for a pretrained Shape foundation model.

    Args:
        model: ``GAOTBackbone`` with pretrained weights.
        device: torch device for forward passes.
        loss_computer: Optional ``LossComputer`` from the same checkpoint.
            When provided, the evaluator uses its calibrated per-dimension
            target-normalization buffers and its masked-token mask
            generator, so reconstruction metrics are computed in the exact
            target space the model was trained on. When ``None``, the
            reconstruction section is skipped.
    """

    def __init__(
        self,
        model: GAOTBackbone,
        device: torch.device | str = "cuda",
        loss_computer: Any = None,
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device).eval()
        self.loss_computer = loss_computer
        if loss_computer is not None:
            loss_computer.to(self.device).eval()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_all(
        self,
        dataloader: DataLoader,
        mask_ratio: float = 0.5,
        contrastive_jitter: float = 0.02,
        contrastive_dropout: float = 0.30,
        contrastive_temperature: float = 0.07,
        uniformity_t: float = 2.0,
        subsample_for_pairwise: int = 2048,
    ) -> dict[str, float]:
        """Run all self-supervised evaluation metrics over the full dataloader.

        Args:
            mask_ratio: fraction of latent tokens to mask for the reconstruction
                metric. Should match the training ``masked_token.mask_ratio``.
            contrastive_jitter: per-coordinate Gaussian jitter σ used to produce
                the augmented view; should match the training
                ``contrastive.jitter_std``.
            contrastive_dropout: probability of point dropout for the augmented
                view; should match ``contrastive.point_dropout``.
            contrastive_temperature: InfoNCE temperature; should match
                ``contrastive.temperature``.
            uniformity_t: Wang & Isola (2020) uniformity kernel parameter.
            subsample_for_pairwise: cap on the number of embeddings used for
                pairwise (O(N²)) metrics such as uniformity and pairwise
                cosine statistics. Does not affect alignment or InfoNCE.
        """
        has_recon = (
            self.loss_computer is not None
            and self.loss_computer.recon_proj is not None
        )

        # Streaming accumulators — avoid holding per-token tensors in RAM.
        recon = {
            "sum_smoothl1": 0.0,
            "sum_mse": 0.0,
            "sum_ss_res": 0.0,
            "sum_target": 0.0,
            "sum_target_sq": 0.0,
            "n_elem": 0,
        }

        clean_embeddings: list[torch.Tensor] = []
        aug_embeddings: list[torch.Tensor] = []

        for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            points = batch["points"].to(self.device, non_blocking=True)
            features = batch["features"].to(self.device, non_blocking=True)
            normals = batch.get("normals")
            curvature = batch.get("curvature")
            if normals is not None:
                normals = normals.to(self.device, non_blocking=True)
            if curvature is not None:
                curvature = curvature.to(self.device, non_blocking=True)

            B = points.shape[0]

            # ------------------------------------------------------------
            # Clean forward (with mask, for reconstruction + clean embedding)
            # ------------------------------------------------------------
            T = (
                self.model.module.encoder.num_tokens
                if hasattr(self.model, "module")
                else self.model.encoder.num_tokens
            )
            mask = self._make_eval_mask(B, T, mask_ratio)

            out_clean = self.model(points, features, normals, curvature, mask)
            clean_embeddings.append(out_clean["pooled_embedding"].detach().cpu())

            # ------------------------------------------------------------
            # Reconstruction metrics in normalized target space
            # ------------------------------------------------------------
            if has_recon and "raw_geo_stats" in out_clean:
                target_raw = out_clean["raw_geo_stats"]
                target = self.loss_computer._maybe_normalize_target(target_raw)
                predicted = self.loss_computer.recon_proj(out_clean["token_embeddings"])

                pred_m = predicted[mask]        # (M_batch, D)
                tgt_m = target[mask]             # (M_batch, D)

                if pred_m.numel() > 0:
                    recon["sum_smoothl1"] += F.smooth_l1_loss(
                        pred_m, tgt_m, reduction="sum", beta=1.0,
                    ).item()
                    recon["sum_mse"] += F.mse_loss(
                        pred_m, tgt_m, reduction="sum",
                    ).item()
                    recon["sum_ss_res"] += (pred_m - tgt_m).pow(2).sum().item()
                    recon["sum_target"] += tgt_m.sum().item()
                    recon["sum_target_sq"] += tgt_m.pow(2).sum().item()
                    recon["n_elem"] += pred_m.numel()

            # ------------------------------------------------------------
            # Augmented forward (for contrastive metrics)
            # ------------------------------------------------------------
            aug_points, aug_features, aug_normals, aug_curvature = self._augment_view(
                points, features, normals, curvature,
                jitter=contrastive_jitter, dropout=contrastive_dropout,
            )
            out_aug = self.model(
                aug_points, aug_features, aug_normals, aug_curvature, None,
            )
            aug_embeddings.append(out_aug["pooled_embedding"].detach().cpu())

        # ======================================================================
        # Aggregate
        # ======================================================================
        results: dict[str, float] = {}

        # --- Reconstruction ---
        if recon["n_elem"] > 0:
            n = recon["n_elem"]
            mean_target = recon["sum_target"] / n
            ss_tot = recon["sum_target_sq"] - n * (mean_target ** 2)
            ss_res = recon["sum_ss_res"]
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

            results["recon_smoothl1"] = recon["sum_smoothl1"] / n
            results["recon_mse"] = recon["sum_mse"] / n
            results["recon_r2"] = r2

        # --- Contrastive + embedding geometry ---
        clean = torch.cat(clean_embeddings, dim=0)
        aug = torch.cat(aug_embeddings, dim=0)
        N = clean.shape[0]

        if N >= 2:
            clean_norm = F.normalize(clean, dim=-1)
            aug_norm = F.normalize(aug, dim=-1)

            # Alignment (Wang & Isola 2020, Eq. 1): E[||f(x) - f(y)||²] for positive pairs
            alignment = (clean_norm - aug_norm).pow(2).sum(dim=-1).mean().item()
            results["contrastive_alignment"] = alignment

            # Uniformity (Wang & Isola 2020, Eq. 2): log E[exp(-t ||x - y||²)]
            M = min(N, subsample_for_pairwise)
            idx = torch.randperm(N)[:M]
            sub = clean_norm[idx]
            pdist_sq = torch.cdist(sub, sub).pow(2)
            off_diag = ~torch.eye(M, dtype=torch.bool)
            uniformity = torch.log(
                torch.exp(-uniformity_t * pdist_sq[off_diag]).mean() + 1e-12
            ).item()
            results["contrastive_uniformity"] = uniformity

            # InfoNCE on (clean, aug) pairs — symmetric
            sub_clean = clean_norm[idx]
            sub_aug = aug_norm[idx]
            logits = sub_clean @ sub_aug.T / contrastive_temperature
            labels = torch.arange(M)
            loss_ab = F.cross_entropy(logits, labels).item()
            loss_ba = F.cross_entropy(logits.T, labels).item()
            results["contrastive_infonce"] = 0.5 * (loss_ab + loss_ba)

            # Top-1 retrieval accuracy: for each query, is its positive pair
            # the top-1 cosine-nearest neighbor in the augmented view pool?
            pred_ab = logits.argmax(dim=-1)
            results["contrastive_top1_acc"] = (pred_ab == labels).float().mean().item()

            # Embedding norm distribution
            norms = clean.norm(dim=-1)
            results["embedding_norm_mean"] = norms.mean().item()
            results["embedding_norm_std"] = norms.std().item()

            # Random-pair cosine similarity distribution
            idx_a = torch.randperm(N)[:M]
            idx_b = torch.randperm(N)[:M]
            valid = idx_a != idx_b
            if valid.any():
                sims = (clean_norm[idx_a[valid]] * clean_norm[idx_b[valid]]).sum(dim=-1)
                results["pairwise_cosine_mean"] = sims.mean().item()
                results["pairwise_cosine_std"] = sims.std().item()

        return results

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_robustness(
        self,
        dataloader: DataLoader,
        noise_levels: list[float] | None = None,
        decimation_ratios: list[float] | None = None,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Embedding stability under input perturbations.

        For each perturbation magnitude, reports the mean cosine agreement
        between pooled embeddings of the clean and perturbed mesh, averaged
        over the full validation set (or up to ``max_batches`` batches).
        A perfectly robust model yields 1.0 across all conditions.
        """
        if noise_levels is None:
            noise_levels = [0.001, 0.005, 0.01, 0.05]
        if decimation_ratios is None:
            decimation_ratios = [0.5, 0.25, 0.1]

        accum: dict[str, list[torch.Tensor]] = defaultdict(list)

        total = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
        for i, batch in enumerate(tqdm(dataloader, desc="Robustness", total=total)):
            if max_batches is not None and i >= max_batches:
                break

            points = batch["points"].to(self.device, non_blocking=True)
            features = batch["features"].to(self.device, non_blocking=True)
            normals = batch.get("normals")
            if normals is not None:
                normals = normals.to(self.device, non_blocking=True)

            out_clean = self.model.forward_tokens(points, features, normals)
            emb_clean = F.normalize(out_clean["pooled_embedding"], dim=-1)

            # Noise perturbation
            for sigma in noise_levels:
                noisy_points = points + torch.randn_like(points) * sigma
                noisy_features = features.clone()
                if noisy_features.shape[-1] >= 3:
                    noisy_features[..., :3] = noisy_points
                out_noisy = self.model.forward_tokens(noisy_points, noisy_features, normals)
                emb_noisy = F.normalize(out_noisy["pooled_embedding"], dim=-1)
                cos = (emb_clean * emb_noisy).sum(dim=-1)
                accum[f"robustness_noise_{sigma}"].append(cos.detach().cpu())

            # Decimation perturbation
            for ratio in decimation_ratios:
                n_keep = max(1, int(points.shape[1] * ratio))
                idx = torch.randperm(points.shape[1], device=self.device)[:n_keep]
                dec_points = points[:, idx]
                dec_features = features[:, idx]
                dec_normals = normals[:, idx] if normals is not None else None
                out_dec = self.model.forward_tokens(dec_points, dec_features, dec_normals)
                emb_dec = F.normalize(out_dec["pooled_embedding"], dim=-1)
                cos = (emb_clean * emb_dec).sum(dim=-1)
                accum[f"robustness_decimate_{ratio}"].append(cos.detach().cpu())

        results: dict[str, float] = {}
        for key, vals in accum.items():
            if vals:
                results[key] = torch.cat(vals).mean().item()
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_eval_mask(self, B: int, T: int, mask_ratio: float) -> torch.Tensor:
        """Generate a token mask using the same strategy as training.

        Prefers the loss computer's ``MaskedTokenLoss.create_mask`` to keep
        the evaluated mask distribution identical to training. Falls back
        to uniform random masking if the loss computer is unavailable.
        """
        if (
            self.loss_computer is not None
            and self.loss_computer.masked_token is not None
        ):
            orig = self.loss_computer.masked_token.mask_ratio
            self.loss_computer.masked_token.mask_ratio = mask_ratio
            try:
                return self.loss_computer.masked_token.create_mask(B, T, self.device)
            finally:
                self.loss_computer.masked_token.mask_ratio = orig

        n_mask = int(T * mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        for b in range(B):
            idx = torch.randperm(T, device=self.device)[:n_mask]
            mask[b, idx] = True
        return mask

    def _augment_view(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None,
        curvature: torch.Tensor | None,
        jitter: float,
        dropout: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Produce an augmented view matching the training-time augmentation."""
        noise = torch.randn_like(points) * jitter
        aug_points = points + noise

        aug_features = features.clone()
        if aug_features.shape[-1] >= 3:
            aug_features[..., :3] = aug_points

        drop = torch.rand(aug_points.shape[:2], device=aug_points.device) < dropout
        keep = (~drop).unsqueeze(-1).float()
        aug_points = aug_points * keep
        aug_features = aug_features * keep
        aug_normals = normals * keep if normals is not None else None
        aug_curvature = curvature * keep if curvature is not None else None
        return aug_points, aug_features, aug_normals, aug_curvature
