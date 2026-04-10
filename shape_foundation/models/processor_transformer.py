"""Transformer processor with 3D patchification.

Takes latent tokens from the MAGNO encoder, patchifies them into a
3D grid, processes with transformer layers (GQA, RMSNorm), and
unpatchifies back to token space.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from shape_foundation.configs.default import ProcessorConfig


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def build_norm(dim: int, norm_type: str) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embed[:, :x.shape[1]]


class RotaryPositionalEmbedding3D(nn.Module):
    """3D RoPE: splits hidden dim into 3 groups, applies 1D RoPE per axis."""

    def __init__(self, dim: int, grid_shape: tuple[int, int, int], theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.grid_shape = grid_shape
        assert dim % 6 == 0, f"dim={dim} must be divisible by 6 for 3D RoPE"
        self.dim_per_axis = dim // 3

        # precompute freqs for each axis
        freqs_list = []
        for axis_size in grid_shape:
            half = self.dim_per_axis // 2
            freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
            positions = torch.arange(axis_size, dtype=torch.float32)
            angles = torch.outer(positions, freqs)  # (axis_size, half)
            freqs_list.append(angles)
        self.register_buffer("freqs_x", freqs_list[0])
        self.register_buffer("freqs_y", freqs_list[1])
        self.register_buffer("freqs_z", freqs_list[2])

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to q and k. Shape: (B, H, T, D)."""
        B, H, T, D = q.shape
        gx, gy, gz = self.grid_shape

        # build full rotation for flattened grid
        cos_parts, sin_parts = [], []
        half = self.dim_per_axis // 2
        for freqs, gs in [(self.freqs_x, gx), (self.freqs_y, gy), (self.freqs_z, gz)]:
            c = freqs.cos()  # (gs, half)
            s = freqs.sin()
            cos_parts.append(c)
            sin_parts.append(s)

        # expand to 3D grid then flatten
        cx = cos_parts[0].unsqueeze(1).unsqueeze(1).expand(gx, gy, gz, half).reshape(T, half)
        sx = sin_parts[0].unsqueeze(1).unsqueeze(1).expand(gx, gy, gz, half).reshape(T, half)
        cy = cos_parts[1].unsqueeze(0).unsqueeze(2).expand(gx, gy, gz, half).reshape(T, half)
        sy = sin_parts[1].unsqueeze(0).unsqueeze(2).expand(gx, gy, gz, half).reshape(T, half)
        cz = cos_parts[2].unsqueeze(0).unsqueeze(0).expand(gx, gy, gz, half).reshape(T, half)
        sz = sin_parts[2].unsqueeze(0).unsqueeze(0).expand(gx, gy, gz, half).reshape(T, half)

        full_cos = torch.cat([cx, cy, cz], dim=-1)  # (T, D//2)
        full_sin = torch.cat([sx, sy, sz], dim=-1)

        # truncate to actual T if padded
        full_cos = full_cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
        full_sin = full_sin[:T].unsqueeze(0).unsqueeze(0)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :D // 2], x[..., D // 2:]
            return torch.cat([
                x1 * full_cos - x2 * full_sin,
                x1 * full_sin + x2 * full_cos,
            ], dim=-1)

        return rotate(q), rotate(k)


# ---------------------------------------------------------------------------
# Grouped-query attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding3D | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            # expand kv to full heads before RoPE, then contract
            k_exp = k.repeat_interleave(self.kv_groups, dim=1)
            q, k_exp = rope(q, k_exp)
            # contract back (take every kv_groups-th head)
            k = k_exp[:, ::self.kv_groups]
        else:
            pass  # no RoPE

        # repeat kv heads for GQA
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)

        # scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ProcessorConfig):
        super().__init__()
        dim = cfg.hidden_size
        self.norm1 = build_norm(dim, cfg.norm_type)
        self.attn = GroupedQueryAttention(dim, cfg.num_heads, cfg.num_kv_heads, cfg.dropout)
        self.norm2 = build_norm(dim, cfg.norm_type)
        mlp_hidden = int(dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding3D | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 3D Patchify / Unpatchify
# ---------------------------------------------------------------------------

class Patchify3D(nn.Module):
    """Reshape (B, H, W, D, C) -> (B, nH*nW*nD, p^3 * C) and project."""

    def __init__(self, patch_size: int, in_dim: int, out_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size ** 3 * in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, D, C)
        p = self.patch_size
        B, H, W, D, C = x.shape
        assert H % p == 0 and W % p == 0 and D % p == 0, \
            f"Grid {H}x{W}x{D} not divisible by patch_size={p}"
        x = rearrange(x, "b (nh p1) (nw p2) (nd p3) c -> b (nh nw nd) (p1 p2 p3 c)",
                       p1=p, p2=p, p3=p)
        return self.proj(x)


class Unpatchify3D(nn.Module):
    """Reverse: (B, T, hidden) -> (B, H, W, D, C)."""

    def __init__(self, patch_size: int, hidden_dim: int, out_dim: int, grid_shape: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.grid_shape = grid_shape
        self.proj = nn.Linear(hidden_dim, patch_size ** 3 * out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        nH = self.grid_shape[0] // p
        nW = self.grid_shape[1] // p
        nD = self.grid_shape[2] // p
        x = self.proj(x)  # (B, T, p^3 * C)
        x = rearrange(x, "b (nh nw nd) (p1 p2 p3 c) -> b (nh p1) (nw p2) (nd p3) c",
                       nh=nH, nw=nW, nd=nD, p1=p, p2=p, p3=p)
        return x


# ---------------------------------------------------------------------------
# Transformer processor
# ---------------------------------------------------------------------------

class TransformerProcessor(nn.Module):
    """Process latent tokens with patchified 3D transformer.

    1. Reshape flat tokens to 3D grid
    2. Patchify -> patch tokens
    3. Add positional embeddings
    4. Process through transformer layers
    5. Unpatchify back to latent token grid
    """

    def __init__(self, cfg: ProcessorConfig, latent_shape: tuple[int, int, int], token_dim: int):
        super().__init__()
        self.cfg = cfg
        self.latent_shape = latent_shape
        self.token_dim = token_dim
        p = cfg.patch_size

        grid_h, grid_w, grid_d = latent_shape
        assert grid_h % p == 0 and grid_w % p == 0 and grid_d % p == 0, \
            f"latent_shape {latent_shape} must be divisible by patch_size {p}"

        self.num_patches = (grid_h // p) * (grid_w // p) * (grid_d // p)
        patch_grid = (grid_h // p, grid_w // p, grid_d // p)

        self.patchify = Patchify3D(p, token_dim, cfg.hidden_size)
        self.unpatchify = Unpatchify3D(p, cfg.hidden_size, token_dim, latent_shape)

        # Positional embedding
        self.rope = None
        if cfg.positional_embedding == "absolute":
            self.pos_embed = AbsolutePositionalEmbedding(self.num_patches, cfg.hidden_size)
        elif cfg.positional_embedding == "rope":
            self.pos_embed = None
            self.rope = RotaryPositionalEmbedding3D(
                cfg.hidden_size // cfg.num_heads,
                patch_grid,
                cfg.rope_theta,
            )
        else:
            raise ValueError(f"Unknown positional embedding: {cfg.positional_embedding}")

        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = build_norm(cfg.hidden_size, cfg.norm_type)

        # Learnable mask token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))

        # Optional UViT skip connections
        self.uvit_skip = cfg.uvit_skip
        if self.uvit_skip and cfg.num_layers > 1:
            n_skip = cfg.num_layers // 2
            self.skip_projs = nn.ModuleList([
                nn.Linear(cfg.hidden_size * 2, cfg.hidden_size) for _ in range(n_skip)
            ])

    def forward(
        self,
        token_embeddings: torch.Tensor,  # (B, T, C) where T = H*W*D
        mask: torch.Tensor | None = None,  # (B, T) bool, True = masked
    ) -> dict[str, torch.Tensor]:
        """Process latent tokens.

        Returns dict with:
            token_embeddings: (B, T, C) processed tokens
            patch_embeddings: (B, num_patches, hidden) intermediate
            pyramid: list of (B, num_patches, hidden) per layer (optional)
        """
        B, T, C = token_embeddings.shape
        H, W, D = self.latent_shape

        # Reshape to 3D grid
        x_3d = token_embeddings.view(B, H, W, D, C)

        # Patchify
        x = self.patchify(x_3d)  # (B, num_patches, hidden)

        # Apply mask if provided (replace masked patch tokens with learnable mask)
        if mask is not None:
            # mask is per-token (B, T); convert to per-patch using configurable threshold
            mask_3d = mask.view(B, H, W, D)
            p = self.cfg.patch_size
            # compute fraction of masked tokens per patch
            patch_mask_counts = rearrange(
                mask_3d.float(), "b (nh p1) (nw p2) (nd p3) -> b (nh nw nd) (p1 p2 p3)",
                p1=p, p2=p, p3=p,
            )
            patch_mask_ratio = patch_mask_counts.mean(dim=-1)  # (B, num_patches)
            mask_patches = patch_mask_ratio >= self.cfg.mask_patch_threshold
            x = torch.where(mask_patches.unsqueeze(-1), self.mask_token.expand_as(x), x)

        # Positional embedding
        if self.pos_embed is not None:
            x = self.pos_embed(x)

        # Transformer
        pyramid = []
        skip_cache = []

        n_layers = len(self.layers)
        half = n_layers // 2

        for i, layer in enumerate(self.layers):
            if self.uvit_skip:
                if i < half:
                    skip_cache.append(x)
                elif i >= n_layers - half:
                    skip_idx = n_layers - 1 - i
                    if skip_idx < len(skip_cache) and skip_idx < len(self.skip_projs):
                        skip_in = skip_cache[skip_idx]
                        x = self.skip_projs[skip_idx](torch.cat([x, skip_in], dim=-1))

            x = layer(x, self.rope)
            pyramid.append(x)

        x = self.norm(x)

        # Unpatchify
        x_3d_out = self.unpatchify(x)  # (B, H, W, D, C)
        out_tokens = x_3d_out.view(B, T, C)

        return {
            "token_embeddings": out_tokens,
            "patch_embeddings": x,
            "pyramid": pyramid,
        }
