"""GAOTBackbone: Geometry-Aware Operator Transformer backbone.

Combines the MAGNO encoder (tokenizer) and transformer processor into
a single backbone that exposes token-level, feature-level, and task-level
forward passes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_foundation.configs.default import ShapeConfig
from shape_foundation.models.tokenizer_magno import MAGNOEncoder
from shape_foundation.models.processor_transformer import TransformerProcessor
from shape_foundation.models.heads import (
    GeometryEmbeddingHead,
    SymmetryHead,
    PrimitiveTopologyHead,
    PartRegionHead,
    CaptionHead,
    TopologyReductionHead,
)


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Learned attention pooling over token sequence.

    Computes scalar attention logits per token, softmax-weights tokens,
    and returns the weighted sum projected to emb_dim.
    """

    def __init__(self, token_dim: int, emb_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, emb_dim),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (B, T, C)
        Returns:
            pooled: (B, emb_dim)
        """
        # (B, T, 1) -> softmax over T
        attn_logits = self.score(token_embeddings)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)
        # weighted sum
        pooled = (token_embeddings * attn_weights).sum(dim=1)  # (B, C)
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class GAOTBackbone(nn.Module):
    """Geometry-Aware Operator Transformer backbone.

    Architecture:
        MAGNO Encoder -> (optional token pos enc) -> Transformer Processor -> Task Heads

    Forward modes:
        forward_tokens: raw token + pooled embeddings
        forward_features: tokens + head outputs
        forward_tasks: full pipeline including task-specific postprocessing
    """

    def __init__(self, cfg: ShapeConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder (tokenizer): mesh -> latent tokens
        self.encoder = MAGNOEncoder(cfg.tokenizer, cfg.input)

        # Processor: latent tokens -> processed tokens
        self.processor = TransformerProcessor(
            cfg.processor,
            latent_shape=cfg.tokenizer.latent.latent_shape,
            token_dim=cfg.tokenizer.latent.token_dim,
        )

        token_dim = cfg.tokenizer.latent.token_dim
        emb_dim = cfg.heads.embedding_dim

        # Optional token-level positional encoding from latent grid 3D coords
        self.token_pos_proj = None
        if cfg.tokenizer.token_pos_encoding:
            self.token_pos_proj = nn.Linear(3, token_dim)

        # Pooling: mean or learned attention
        self.pooling_mode = cfg.heads.pooling
        if self.pooling_mode == "attention":
            self.pool = AttentionPooling(token_dim, emb_dim)
        else:
            # mean pooling with projection
            self.pool = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, emb_dim),
            )

        # Task heads
        self.heads = nn.ModuleDict()
        self.heads["embedding"] = GeometryEmbeddingHead(token_dim, emb_dim)

        if cfg.heads.symmetry.enabled:
            self.heads["symmetry"] = SymmetryHead(emb_dim, cfg.heads.symmetry)
        if cfg.heads.primitive.enabled:
            self.heads["primitive"] = PrimitiveTopologyHead(token_dim, emb_dim, cfg.heads.primitive)
        if cfg.heads.part.enabled:
            self.heads["part"] = PartRegionHead(token_dim, cfg.heads.part)
        if cfg.heads.caption.enabled:
            self.heads["caption"] = CaptionHead(emb_dim, cfg.heads.caption)
        if cfg.heads.reduction.enabled:
            self.heads["reduction"] = TopologyReductionHead(emb_dim, cfg.heads.reduction)

    def _encode(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None = None,
        curvature: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.encoder(points, features, normals, curvature)

    def _add_token_pos(self, token_embeddings: torch.Tensor, enc_out: dict) -> torch.Tensor:
        """Add latent grid positional encoding to token embeddings if enabled."""
        if self.token_pos_proj is not None and "grid_positions" in enc_out:
            grid_pos = enc_out["grid_positions"]  # (T, 3)
            B = token_embeddings.shape[0]
            # broadcast grid positions across batch
            pos_enc = self.token_pos_proj(grid_pos)  # (T, C)
            token_embeddings = token_embeddings + pos_enc.unsqueeze(0).expand(B, -1, -1)
        return token_embeddings

    def _process(
        self,
        token_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.processor(token_embeddings, mask)

    def _pool(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "attention":
            return self.pool(token_embeddings)
        else:
            pooled = token_embeddings.mean(dim=1)
            return self.pool(pooled)

    def forward_tokens(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None = None,
        curvature: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode + process, return raw token and pooled embeddings."""
        enc = self._encode(points, features, normals, curvature)
        tokens = self._add_token_pos(enc["token_embeddings"], enc)
        proc = self._process(tokens, mask)
        pooled = self._pool(proc["token_embeddings"])

        result = {
            "token_embeddings": proc["token_embeddings"],
            "pooled_embedding": pooled,
            "pyramid": proc.get("pyramid", []),
        }
        if "raw_geo_stats" in enc:
            result["raw_geo_stats"] = enc["raw_geo_stats"]
        return result

    def forward_features(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None = None,
        curvature: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Like forward_tokens but also runs task heads."""
        backbone_out = self.forward_tokens(points, features, normals, curvature, mask)
        token_emb = backbone_out["token_embeddings"]
        pooled = backbone_out["pooled_embedding"]

        head_outputs = {}
        for name, head in self.heads.items():
            if name == "embedding":
                head_outputs[name] = head(token_emb, pooled)
            elif name in ("symmetry", "caption", "reduction"):
                head_outputs[name] = head(pooled)
            elif name == "primitive":
                head_outputs[name] = head(token_emb, pooled)
            elif name == "part":
                head_outputs[name] = head(token_emb)
            else:
                head_outputs[name] = head(token_emb, pooled)

        return {**backbone_out, "heads": head_outputs}

    def forward_tasks(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None = None,
        curvature: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full pipeline for inference: encode, process, run all heads."""
        return self.forward_features(points, features, normals, curvature)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None = None,
        curvature: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Default forward = forward_features."""
        return self.forward_features(points, features, normals, curvature, mask)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
