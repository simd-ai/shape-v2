"""MAGNO (Multi-scale Attention Graph Neural Operator) encoder.

Tokenizes mesh/point-cloud inputs into structured latent tokens via
cross-attention from a latent grid to physical surface points, with
explicit geometric embeddings and multi-scale neighbor aggregation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from shape_foundation.configs.default import TokenizerConfig, InputConfig


# ---------------------------------------------------------------------------
# Neighbor search
# ---------------------------------------------------------------------------

def _radius_search_torch_cluster(
    query: torch.Tensor,
    support: torch.Tensor,
    radius: float,
    max_neighbors: int,
    batch_q: torch.Tensor | None = None,
    batch_s: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radius neighbor search via torch_cluster.

    Returns:
        row: indices into query  (edges: query[row] <-- support[col])
        col: indices into support
    """
    from torch_cluster import radius as tc_radius

    edge_index = tc_radius(
        support, query, r=radius,
        batch_x=batch_s, batch_y=batch_q,
        max_num_neighbors=max_neighbors,
    )
    # edge_index[0] = query indices (into y), edge_index[1] = support indices (into x)
    return edge_index[0], edge_index[1]  # (query_idx, support_idx)


def _knn_search_torch_cluster(
    query: torch.Tensor,
    support: torch.Tensor,
    k: int,
    batch_q: torch.Tensor | None = None,
    batch_s: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch_cluster import knn

    edge_index = knn(support, query, k=k, batch_x=batch_s, batch_y=batch_q)
    return edge_index[0], edge_index[1]  # (query_idx, support_idx)


def _radius_search_native(
    query: torch.Tensor,
    support: torch.Tensor,
    radius: float,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch fallback radius search (no batching, for small inputs)."""
    dists = torch.cdist(query, support)  # (Q, S)
    mask = dists < radius
    rows, cols = [], []
    for i in range(query.shape[0]):
        js = mask[i].nonzero(as_tuple=True)[0]
        if js.numel() > max_neighbors:
            perm = torch.randperm(js.numel(), device=js.device)[:max_neighbors]
            js = js[perm]
        rows.append(torch.full((js.numel(),), i, device=query.device, dtype=torch.long))
        cols.append(js)
    return torch.cat(rows), torch.cat(cols)


def neighbor_search(
    query: torch.Tensor,
    support: torch.Tensor,
    cfg: "TokenizerConfig",
    radius: float,
    batch_q: torch.Tensor | None = None,
    batch_s: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch neighbor search to configured backend."""
    backend = cfg.neighbor.backend
    strategy = cfg.neighbor.strategy

    if backend == "auto":
        try:
            import torch_cluster  # noqa: F401
            backend = "torch_cluster"
        except ImportError:
            backend = "native"

    if strategy == "knn":
        if backend == "torch_cluster":
            return _knn_search_torch_cluster(
                query, support, cfg.neighbor.knn_k, batch_q, batch_s,
            )
        # native knn fallback
        dists = torch.cdist(query, support)
        _, idx = dists.topk(cfg.neighbor.knn_k, dim=-1, largest=False)
        q_idx = torch.arange(query.shape[0], device=query.device).unsqueeze(1).expand_as(idx)
        return q_idx.reshape(-1), idx.reshape(-1)

    # radius search
    if backend == "torch_cluster":
        return _radius_search_torch_cluster(
            query, support, radius, cfg.neighbor.max_neighbors, batch_q, batch_s,
        )
    return _radius_search_native(query, support, radius, cfg.neighbor.max_neighbors)


# ---------------------------------------------------------------------------
# Geometric embedding
# ---------------------------------------------------------------------------

class StatisticalGeoEmbed(nn.Module):
    """Compute geometric embedding from neighborhood statistics.

    For each query token and its neighbors, compute mean/std/min/max of
    relative position vectors, then pass through an MLP.
    """

    def __init__(self, cfg: "TokenizerConfig"):
        super().__init__()
        self.cfg = cfg.geo_embed
        n_stats = len(self.cfg.stat_features)  # e.g. 4: mean, std, min, max
        coord_dim = 3
        raw_dim = n_stats * coord_dim  # 12

        # extra features: normals (3), curvature (1)
        extra = 0
        if self.cfg.augment_normals:
            extra += 3 * n_stats
        if self.cfg.augment_curvature:
            extra += 1 * n_stats
        raw_dim += extra

        self.raw_dim = raw_dim
        layers = []
        in_d = raw_dim
        for _ in range(self.cfg.mlp_layers):
            layers.extend([nn.Linear(in_d, self.cfg.mlp_hidden), nn.GELU()])
            in_d = self.cfg.mlp_hidden
        self.mlp = nn.Sequential(*layers)
        self.out_dim = self.cfg.mlp_hidden

    def forward(
        self,
        query_pos: torch.Tensor,       # (Q, 3)
        support_pos: torch.Tensor,      # (S, 3)
        q_idx: torch.Tensor,            # (E,) edge query indices
        s_idx: torch.Tensor,            # (E,) edge support indices
        support_normals: torch.Tensor | None = None,  # (S, 3)
        support_curvature: torch.Tensor | None = None,  # (S, 1)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (embedding, raw_stats) where raw_stats is optional."""
        # relative positions per edge
        rel = support_pos[s_idx] - query_pos[q_idx]  # (E, 3)

        # aggregate per query
        Q = query_pos.shape[0]
        stats_parts = []

        for stat_name in self.cfg.stat_features:
            s = self._scatter_stat(rel, q_idx, Q, stat_name)  # (Q, 3)
            stats_parts.append(s)

        if self.cfg.augment_normals and support_normals is not None:
            rel_n = support_normals[s_idx]  # (E, 3)
            for stat_name in self.cfg.stat_features:
                s = self._scatter_stat(rel_n, q_idx, Q, stat_name)
                stats_parts.append(s)

        if self.cfg.augment_curvature and support_curvature is not None:
            rel_c = support_curvature[s_idx]  # (E, 1)
            for stat_name in self.cfg.stat_features:
                s = self._scatter_stat(rel_c, q_idx, Q, stat_name)
                stats_parts.append(s)

        raw_stats = torch.cat(stats_parts, dim=-1)  # (Q, raw_dim)
        embedding = self.mlp(raw_stats)  # (Q, mlp_hidden)

        return embedding, raw_stats if self.cfg.return_raw_stats else None

    @staticmethod
    def _scatter_stat(
        values: torch.Tensor, idx: torch.Tensor, num_groups: int, stat: str,
    ) -> torch.Tensor:
        D = values.shape[-1]
        if stat == "mean":
            out = torch.zeros(num_groups, D, device=values.device, dtype=values.dtype)
            count = torch.zeros(num_groups, 1, device=values.device, dtype=values.dtype)
            out.scatter_add_(0, idx.unsqueeze(-1).expand_as(values), values)
            count.scatter_add_(0, idx.unsqueeze(-1), torch.ones_like(idx, dtype=values.dtype).unsqueeze(-1))
            return out / count.clamp(min=1)
        elif stat == "std":
            mean = StatisticalGeoEmbed._scatter_stat(values, idx, num_groups, "mean")
            diff_sq = (values - mean[idx]) ** 2
            var = torch.zeros(num_groups, D, device=values.device, dtype=values.dtype)
            count = torch.zeros(num_groups, 1, device=values.device, dtype=values.dtype)
            var.scatter_add_(0, idx.unsqueeze(-1).expand_as(diff_sq), diff_sq)
            count.scatter_add_(0, idx.unsqueeze(-1), torch.ones_like(idx, dtype=values.dtype).unsqueeze(-1))
            return (var / count.clamp(min=1)).sqrt()
        elif stat == "min":
            out = torch.full((num_groups, D), float("inf"), device=values.device, dtype=values.dtype)
            out.scatter_reduce_(0, idx.unsqueeze(-1).expand_as(values), values, reduce="amin")
            out = out.clamp(min=-100, max=100)
            return out
        elif stat == "max":
            out = torch.full((num_groups, D), float("-inf"), device=values.device, dtype=values.dtype)
            out.scatter_reduce_(0, idx.unsqueeze(-1).expand_as(values), values, reduce="amax")
            out = out.clamp(min=-100, max=100)
            return out
        else:
            raise ValueError(f"Unknown stat: {stat}")


class PointNetGeoEmbed(nn.Module):
    """PointNet-style per-neighbor MLP + max-pool geometric embedding."""

    def __init__(self, cfg: "TokenizerConfig"):
        super().__init__()
        in_dim = 3  # relative position
        if cfg.geo_embed.augment_normals:
            in_dim += 3
        if cfg.geo_embed.augment_curvature:
            in_dim += 1
        hidden = cfg.geo_embed.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out_dim = hidden

    def forward(
        self,
        query_pos: torch.Tensor,
        support_pos: torch.Tensor,
        q_idx: torch.Tensor,
        s_idx: torch.Tensor,
        support_normals: torch.Tensor | None = None,
        support_curvature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        rel = support_pos[s_idx] - query_pos[q_idx]
        parts = [rel]
        if support_normals is not None:
            parts.append(support_normals[s_idx])
        if support_curvature is not None:
            parts.append(support_curvature[s_idx])
        x = torch.cat(parts, dim=-1)
        x = self.mlp(x)  # (E, hidden)
        Q = query_pos.shape[0]
        out = torch.zeros(Q, x.shape[-1], device=x.device, dtype=x.dtype)
        out.scatter_reduce_(0, q_idx.unsqueeze(-1).expand_as(x), x, reduce="amax")
        return out, None


def build_geo_embed(cfg: "TokenizerConfig") -> nn.Module:
    if cfg.geo_embed.mode == "statistical":
        return StatisticalGeoEmbed(cfg)
    elif cfg.geo_embed.mode == "pointnet":
        return PointNetGeoEmbed(cfg)
    else:
        return None


# ---------------------------------------------------------------------------
# AGNO attention layer
# ---------------------------------------------------------------------------

class AGNOCrossAttention(nn.Module):
    """Attentional Graph Neural Operator cross-attention.

    Query tokens attend to physical support points weighted by
    geometric embeddings and cosine/dot similarity.
    """

    def __init__(self, token_dim: int, geo_dim: int, cfg: "TokenizerConfig"):
        super().__init__()
        agno = cfg.agno
        self.num_heads = agno.num_heads
        self.head_dim = token_dim // agno.num_heads
        assert token_dim % agno.num_heads == 0

        self.W_q = nn.Linear(token_dim, token_dim, bias=False)
        self.W_k = nn.Linear(token_dim, token_dim, bias=False)
        self.W_v = nn.Linear(token_dim, token_dim, bias=False)
        self.W_g = nn.Linear(geo_dim, token_dim, bias=False)  # project geo embed to value space
        self.out_proj = nn.Linear(token_dim, token_dim)

        self.similarity = agno.similarity
        if agno.learned_temperature:
            self.log_tau = nn.Parameter(torch.tensor(math.log(agno.initial_temperature)))
        else:
            self.register_buffer("log_tau", torch.tensor(math.log(agno.initial_temperature)))

        self.dropout = nn.Dropout(agno.dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,  # (Q, C)
        support_feats: torch.Tensor,  # (S, C)
        geo_embed: torch.Tensor,      # (Q, G)
        q_idx: torch.Tensor,          # (E,)
        s_idx: torch.Tensor,          # (E,)
        num_queries: int,
    ) -> torch.Tensor:
        """Cross-attend from query tokens to support features."""
        Q_feat = self.W_q(query_tokens)  # (Q, C)
        K_feat = self.W_k(support_feats)  # (S, C)
        V_feat = self.W_v(support_feats)  # (S, C)
        G_feat = self.W_g(geo_embed)      # (Q, C)

        # gather along edges
        q_h = Q_feat[q_idx]  # (E, C)
        k_h = K_feat[s_idx]  # (E, C)
        v_h = V_feat[s_idx]  # (E, C)
        g_h = G_feat[q_idx]  # (E, C)

        # reshape to heads
        E = q_h.shape[0]
        q_h = q_h.view(E, self.num_heads, self.head_dim)
        k_h = k_h.view(E, self.num_heads, self.head_dim)
        v_h = v_h.view(E, self.num_heads, self.head_dim)
        g_h = g_h.view(E, self.num_heads, self.head_dim)

        # attention scores
        tau = self.log_tau.exp()
        if self.similarity == "cosine":
            q_norm = F.normalize(q_h, dim=-1)
            k_norm = F.normalize(k_h, dim=-1)
            scores = (q_norm * k_norm).sum(dim=-1) / tau  # (E, H)
        else:
            scores = (q_h * k_h).sum(dim=-1) / (math.sqrt(self.head_dim) * tau)

        # softmax per query (scatter)
        scores_max = torch.zeros(num_queries, self.num_heads, device=scores.device, dtype=scores.dtype)
        scores_max.scatter_reduce_(
            0, q_idx.unsqueeze(-1).expand_as(scores), scores, reduce="amax",
        )
        scores = scores - scores_max[q_idx]
        scores_exp = scores.exp()
        scores_sum = torch.zeros(num_queries, self.num_heads, device=scores.device, dtype=scores.dtype)
        scores_sum.scatter_add_(0, q_idx.unsqueeze(-1).expand_as(scores_exp), scores_exp)
        alpha = scores_exp / scores_sum[q_idx].clamp(min=1e-8)  # (E, H)
        alpha = self.dropout(alpha)

        # weighted sum of (value + geo bias)
        weighted = alpha.unsqueeze(-1) * (v_h + g_h)  # (E, H, D)
        out = torch.zeros(num_queries, self.num_heads, self.head_dim, device=weighted.device, dtype=weighted.dtype)
        out.scatter_add_(
            0,
            q_idx.unsqueeze(-1).unsqueeze(-1).expand_as(weighted),
            weighted,
        )
        out = out.view(num_queries, -1)  # (Q, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Single-scale MAGNO layer
# ---------------------------------------------------------------------------

class MAGNOLayer(nn.Module):
    """One scale of MAGNO: geo-embed + AGNO cross-attention + FFN."""

    def __init__(self, token_dim: int, cfg: "TokenizerConfig"):
        super().__init__()
        self.geo_embed_mod = build_geo_embed(cfg)
        geo_dim = self.geo_embed_mod.out_dim if self.geo_embed_mod is not None else token_dim
        self.cross_attn = AGNOCrossAttention(token_dim, geo_dim, cfg)
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,    # (Q, C)
        query_pos: torch.Tensor,        # (Q, 3)
        support_feats: torch.Tensor,    # (S, C)
        support_pos: torch.Tensor,      # (S, 3)
        q_idx: torch.Tensor,
        s_idx: torch.Tensor,
        support_normals: torch.Tensor | None = None,
        support_curvature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        Q = query_tokens.shape[0]

        # geometric embedding
        raw_stats = None
        if self.geo_embed_mod is not None:
            geo, raw_stats = self.geo_embed_mod(
                query_pos, support_pos, q_idx, s_idx,
                support_normals, support_curvature,
            )
        else:
            geo = torch.zeros(Q, self.cross_attn.W_g.in_features, device=query_tokens.device)

        # cross-attention + residual
        h = self.norm1(query_tokens)
        h = self.cross_attn(h, support_feats, geo, q_idx, s_idx, Q)
        query_tokens = query_tokens + h

        # FFN + residual
        h = self.norm2(query_tokens)
        query_tokens = query_tokens + self.ffn(h)

        return query_tokens, raw_stats


# ---------------------------------------------------------------------------
# MAGNO Encoder (full tokenizer)
# ---------------------------------------------------------------------------

class MAGNOEncoder(nn.Module):
    """Multi-scale MAGNO encoder: mesh/point-cloud -> latent tokens.

    Creates a structured 3D latent grid, then cross-attends from grid
    tokens to physical surface points at multiple spatial scales.
    """

    def __init__(self, cfg: TokenizerConfig, input_cfg: InputConfig):
        super().__init__()
        self.cfg = cfg
        self.input_cfg = input_cfg
        self.token_dim = cfg.latent.token_dim
        self.latent_shape = cfg.latent.latent_shape

        # Input feature projection
        in_feat_dim = 3  # xyz
        if input_cfg.compute_normals:
            in_feat_dim += 3
        if input_cfg.compute_curvature:
            in_feat_dim += 1
        if input_cfg.append_constant_channel:
            in_feat_dim += 1
        self.input_proj = nn.Linear(in_feat_dim, self.token_dim)

        # Learnable latent token embeddings
        if cfg.latent.strategy == "structured_latent_grid":
            num_tokens = cfg.latent.latent_shape[0] * cfg.latent.latent_shape[1] * cfg.latent.latent_shape[2]
        elif cfg.latent.strategy == "surface_adaptive_queries":
            num_tokens = cfg.latent.adaptive_num_tokens
        elif cfg.latent.strategy == "multiresolution_latent_grid":
            num_tokens = sum(s[0] * s[1] * s[2] for s in cfg.latent.multires_shapes)
        else:
            raise ValueError(f"Unknown latent strategy: {cfg.latent.strategy}")

        self.num_tokens = num_tokens
        self.latent_embed = nn.Parameter(torch.randn(num_tokens, self.token_dim) * 0.02)

        # Build latent grid positions (registered as buffer)
        grid_pos = self._build_grid_positions()
        self.register_buffer("grid_pos", grid_pos)  # (num_tokens, 3)

        # Multi-scale MAGNO layers
        radii = cfg.neighbor.multiscale_radii[:cfg.num_scales]
        self.scales = nn.ModuleList([MAGNOLayer(self.token_dim, cfg) for _ in radii])
        self.radii = radii

        # Scale fusion
        if cfg.scale_fusion == "concat_project" and len(radii) > 1:
            self.fusion = nn.Linear(self.token_dim * len(radii), self.token_dim)
        elif cfg.scale_fusion == "gated" and len(radii) > 1:
            self.gate = nn.Linear(self.token_dim * len(radii), len(radii))
            self.fusion = None
        else:
            self.fusion = None

        self.scale_fusion_mode = cfg.scale_fusion
        self.out_norm = nn.LayerNorm(self.token_dim)

    def _build_grid_positions(self) -> torch.Tensor:
        """Build structured 3D grid in [-1, 1]^3."""
        if self.cfg.latent.strategy in ("structured_latent_grid", "multiresolution_latent_grid"):
            shapes = (
                [self.cfg.latent.latent_shape]
                if self.cfg.latent.strategy == "structured_latent_grid"
                else self.cfg.latent.multires_shapes
            )
            all_pos = []
            for shape in shapes:
                gx = torch.linspace(-1, 1, shape[0])
                gy = torch.linspace(-1, 1, shape[1])
                gz = torch.linspace(-1, 1, shape[2])
                grid = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), dim=-1)
                all_pos.append(grid.reshape(-1, 3))
            return torch.cat(all_pos, dim=0)
        else:
            # adaptive: initial positions uniformly sampled, will be overridden
            return torch.rand(self.cfg.latent.adaptive_num_tokens, 3) * 2 - 1

    def forward(
        self,
        points: torch.Tensor,           # (B, N, 3) surface point positions
        features: torch.Tensor,          # (B, N, F) point features
        normals: torch.Tensor | None = None,    # (B, N, 3)
        curvature: torch.Tensor | None = None,  # (B, N, 1)
        batch_offsets: torch.Tensor | None = None,  # for variable-size batches
    ) -> dict[str, torch.Tensor]:
        """Encode point cloud to latent tokens.

        Returns dict with:
            token_embeddings: (B, T, C)
            grid_positions: (T, 3)
            raw_geo_stats: optional (B, T, raw_dim)
        """
        B, N, _ = points.shape
        device = points.device

        # Project input features
        support_feats = self.input_proj(features)  # (B, N, C)

        # Flatten batch for neighbor search
        flat_support_pos = points.reshape(-1, 3)        # (B*N, 3)
        flat_support_feats = support_feats.reshape(-1, self.token_dim)
        # Ensure normals/curvature are present if config expects them (pad with zeros)
        if normals is None and self.cfg.geo_embed.augment_normals:
            normals = torch.zeros(B, N, 3, device=device, dtype=points.dtype)
        if curvature is None and self.cfg.geo_embed.augment_curvature:
            curvature = torch.zeros(B, N, 1, device=device, dtype=points.dtype)
        flat_normals = normals.reshape(-1, 3) if normals is not None else None
        flat_curvature = curvature.reshape(-1, 1) if curvature is not None else None

        # Expand grid positions per batch
        grid = self.grid_pos.unsqueeze(0).expand(B, -1, -1)  # (B, T, 3)
        flat_grid = grid.reshape(-1, 3)  # (B*T, 3)

        T = self.num_tokens

        # Batch indices for torch_cluster
        batch_q = torch.arange(B, device=device).repeat_interleave(T)
        batch_s = torch.arange(B, device=device).repeat_interleave(N)

        # Initial latent tokens
        flat_tokens = self.latent_embed.unsqueeze(0).expand(B, -1, -1).reshape(-1, self.token_dim)

        # Multi-scale encoding
        scale_outputs = []
        last_raw_stats = None
        for scale_layer, r in zip(self.scales, self.radii):
            q_idx, s_idx = neighbor_search(
                flat_grid, flat_support_pos, self.cfg, r, batch_q, batch_s,
            )

            if q_idx.numel() == 0:
                # no neighbors found at this scale; skip
                scale_outputs.append(flat_tokens)
                continue

            out, raw_stats = scale_layer(
                flat_tokens, flat_grid, flat_support_feats, flat_support_pos,
                q_idx, s_idx, flat_normals, flat_curvature,
            )
            scale_outputs.append(out)
            if raw_stats is not None:
                last_raw_stats = raw_stats

        # Fuse scales
        if len(scale_outputs) == 1:
            flat_tokens = scale_outputs[0]
        elif self.scale_fusion_mode == "concat_project":
            flat_tokens = self.fusion(torch.cat(scale_outputs, dim=-1))
        elif self.scale_fusion_mode == "gated":
            stacked = torch.stack(scale_outputs, dim=-1)  # (B*T, C, S)
            gates = self.gate(torch.cat(scale_outputs, dim=-1)).softmax(dim=-1)  # (B*T, S)
            flat_tokens = (stacked * gates.unsqueeze(1)).sum(dim=-1)
        else:  # sum
            flat_tokens = sum(scale_outputs)

        flat_tokens = self.out_norm(flat_tokens)
        token_embeddings = flat_tokens.view(B, T, self.token_dim)

        result = {
            "token_embeddings": token_embeddings,
            "grid_positions": self.grid_pos,
        }
        if last_raw_stats is not None:
            result["raw_geo_stats"] = last_raw_stats.view(B, T, -1)
        return result
