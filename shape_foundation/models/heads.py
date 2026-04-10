"""Task heads for the Shape foundation model.

Each head is config-driven and operates on token-level and/or pooled
embeddings from the backbone.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_foundation.configs.default import (
    SymmetryHeadConfig,
    PrimitiveHeadConfig,
    PartHeadConfig,
    CaptionHeadConfig,
    ReductionHeadConfig,
)


# ---------------------------------------------------------------------------
# Geometry embedding head
# ---------------------------------------------------------------------------

class GeometryEmbeddingHead(nn.Module):
    """Produces L2-normalized pooled and token-level embeddings for retrieval/transfer."""

    def __init__(self, token_dim: int, emb_dim: int):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, emb_dim),
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(
        self, token_emb: torch.Tensor, pooled: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token_out = F.normalize(self.token_proj(token_emb), dim=-1)
        pool_out = F.normalize(self.pool_proj(pooled), dim=-1)
        return {"token_embedding": token_out, "pooled_embedding": pool_out}


# ---------------------------------------------------------------------------
# Symmetry head
# ---------------------------------------------------------------------------

class SymmetryHead(nn.Module):
    """Classify symmetry reduction type and regress plane/axis candidates.

    Classes: none, mirror_half, mirror_quarter, axisymmetric, cyclic_sector
    """

    LABELS = ["none", "mirror_half", "mirror_quarter", "axisymmetric", "cyclic_sector"]

    def __init__(self, emb_dim: int, cfg: SymmetryHeadConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, cfg.num_classes),
        )

        # Plane regression: K candidates, each (normal_xyz + offset) = 4
        if cfg.regress_plane:
            self.plane_regressor = nn.Sequential(
                nn.Linear(emb_dim, h),
                nn.GELU(),
                nn.Linear(h, cfg.max_candidates * 4),
            )

        # Axis regression: K candidates, each (direction_xyz + point_xyz) = 6
        if cfg.regress_axis:
            self.axis_regressor = nn.Sequential(
                nn.Linear(emb_dim, h),
                nn.GELU(),
                nn.Linear(h, cfg.max_candidates * 6),
            )

        # Per-candidate confidence
        self.confidence = nn.Sequential(
            nn.Linear(emb_dim, h),
            nn.GELU(),
            nn.Linear(h, cfg.max_candidates),
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        B = pooled.shape[0]
        logits = self.classifier(pooled)  # (B, num_classes)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": logits.softmax(dim=-1),
            "confidence": self.confidence(pooled),  # (B, K)
        }

        if self.cfg.regress_plane:
            planes = self.plane_regressor(pooled).view(B, self.cfg.max_candidates, 4)
            # normalize plane normals
            planes_norm = F.normalize(planes[..., :3], dim=-1)
            planes = torch.cat([planes_norm, planes[..., 3:]], dim=-1)
            result["planes"] = planes

        if self.cfg.regress_axis:
            axes = self.axis_regressor(pooled).view(B, self.cfg.max_candidates, 6)
            axes_dir = F.normalize(axes[..., :3], dim=-1)
            axes = torch.cat([axes_dir, axes[..., 3:]], dim=-1)
            result["axes"] = axes

        return result


# ---------------------------------------------------------------------------
# Primitive / topology head
# ---------------------------------------------------------------------------

class PrimitiveTopologyHead(nn.Module):
    """Per-token primitive type classification + topology detection.

    Primitive types: plane, cylinder, cone, sphere, torus, generic_freeform
    Also detects repeated sectors, rings, and constant cross-section regions.
    """

    PRIMITIVES = ["plane", "cylinder", "cone", "sphere", "torus", "generic_freeform"]

    def __init__(self, token_dim: int, emb_dim: int, cfg: PrimitiveHeadConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        # Per-token primitive classification
        self.token_classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, h),
            nn.GELU(),
            nn.Linear(h, cfg.num_primitive_types),
        )

        # Global topology features from pooled embedding
        topology_out = 0
        if cfg.detect_repeated_sectors:
            topology_out += 1  # sector count regression
        if cfg.detect_constant_cross_section:
            topology_out += 1  # binary flag

        if topology_out > 0:
            self.topology_head = nn.Sequential(
                nn.Linear(emb_dim, h),
                nn.GELU(),
                nn.Linear(h, topology_out),
            )
        else:
            self.topology_head = None

        self.detect_repeated_sectors = cfg.detect_repeated_sectors
        self.detect_constant_cross_section = cfg.detect_constant_cross_section

    def forward(
        self, token_emb: torch.Tensor, pooled: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = token_emb.shape

        prim_logits = self.token_classifier(token_emb)  # (B, T, num_types)
        result: dict[str, torch.Tensor] = {
            "primitive_logits": prim_logits,
            "primitive_probs": prim_logits.softmax(dim=-1),
        }

        if self.topology_head is not None:
            topo = self.topology_head(pooled)  # (B, topo_out)
            idx = 0
            if self.detect_repeated_sectors:
                result["repeated_sector_count"] = F.softplus(topo[:, idx])
                idx += 1
            if self.detect_constant_cross_section:
                result["constant_cross_section"] = topo[:, idx].sigmoid()
                idx += 1

        return result


# ---------------------------------------------------------------------------
# Part / region head
# ---------------------------------------------------------------------------

class PartRegionHead(nn.Module):
    """Token-level part segmentation with optional hierarchical grouping."""

    def __init__(self, token_dim: int, cfg: PartHeadConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.segmentation = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, h),
            nn.GELU(),
            nn.Linear(h, cfg.max_parts),
        )

        if cfg.hierarchical:
            # second level: group parts into coarser clusters
            self.hier_head = nn.Sequential(
                nn.Linear(token_dim, h),
                nn.GELU(),
                nn.Linear(h, cfg.max_parts // 4),
            )
        else:
            self.hier_head = None

    def forward(self, token_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        seg_logits = self.segmentation(token_emb)  # (B, T, max_parts)
        result: dict[str, torch.Tensor] = {
            "part_logits": seg_logits,
            "part_probs": seg_logits.softmax(dim=-1),
        }
        if self.hier_head is not None:
            hier_logits = self.hier_head(token_emb)
            result["hier_logits"] = hier_logits
            result["hier_probs"] = hier_logits.softmax(dim=-1)
        return result


# ---------------------------------------------------------------------------
# Caption / description head
# ---------------------------------------------------------------------------

class CaptionHead(nn.Module):
    """Generate structural descriptions from pooled embedding.

    Modes:
        template: predict attribute tokens then fill templates
        retrieval: nearest-neighbor in a bank of descriptions (not implemented here)
        llm: placeholder for LLM-based generation (optional)
    """

    def __init__(self, emb_dim: int, cfg: CaptionHeadConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        if cfg.mode == "template":
            # predict discrete attribute tokens
            self.attr_head = nn.Sequential(
                nn.Linear(emb_dim, h),
                nn.GELU(),
                nn.Linear(h, cfg.vocab_size),
            )
            # autoregressive token prediction
            self.token_embed = nn.Embedding(cfg.vocab_size, h)
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=h, nhead=4, dim_feedforward=h * 2,
                    dropout=0.1, batch_first=True,
                ),
                num_layers=2,
            )
            self.output_proj = nn.Linear(h, cfg.vocab_size)
            self.mem_proj = nn.Linear(emb_dim, h)
        elif cfg.mode == "retrieval":
            self.bank_proj = nn.Linear(emb_dim, h)
        else:
            self.attr_head = None

    def forward(
        self,
        pooled: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}

        if self.cfg.mode == "template":
            attr_logits = self.attr_head(pooled)  # (B, vocab_size)
            result["attr_logits"] = attr_logits

            if target_tokens is not None:
                # teacher-forced decoding
                B, L = target_tokens.shape
                tgt = self.token_embed(target_tokens)  # (B, L, h)
                memory = pooled.unsqueeze(1)  # (B, 1, emb_dim) - project
                # simple: project memory to decoder dim
                memory = self.mem_proj(memory)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
                dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)
                result["token_logits"] = self.output_proj(dec_out)  # (B, L, vocab)
            else:
                result["token_logits"] = attr_logits.unsqueeze(1)

        elif self.cfg.mode == "retrieval":
            result["query_embedding"] = F.normalize(self.bank_proj(pooled), dim=-1)

        return result


# ---------------------------------------------------------------------------
# Topology-reduction recommendation head
# ---------------------------------------------------------------------------

class TopologyReductionHead(nn.Module):
    """Produce structured simulation-reduction recommendations.

    Output:
        recommended_reduction: classification logit
        confidence: scalar per recommendation
        detected features aggregated from other heads
    """

    REDUCTIONS = [
        "none",
        "mirror_half",
        "mirror_quarter",
        "axisymmetric_2d",
        "cyclic_sector",
        "extrusion_2d",
    ]

    def __init__(self, emb_dim: int, cfg: ReductionHeadConfig):
        super().__init__()
        h = cfg.hidden_dim
        num_reductions = len(self.REDUCTIONS)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, num_reductions),
        )

        self.confidence = nn.Sequential(
            nn.Linear(emb_dim, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        # reasoning: predict which geometric features support the recommendation
        self.reason_head = nn.Sequential(
            nn.Linear(emb_dim, h),
            nn.GELU(),
            nn.Linear(h, 8),  # 8 binary reason flags
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.classifier(pooled)
        return {
            "reduction_logits": logits,
            "reduction_probs": logits.softmax(dim=-1),
            "confidence": self.confidence(pooled).squeeze(-1),
            "reason_flags": self.reason_head(pooled),
        }
