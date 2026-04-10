"""Mesh preprocessing: canonicalization, feature computation, augmentation."""

from __future__ import annotations

import numpy as np
import torch

from shape_foundation.configs.default import InputConfig


class MeshPreprocessor:
    """Preprocess raw mesh vertices/faces into model-ready point features.

    Pipeline:
        1. Canonicalize to centered unit box [-1, 1]^3
        2. Sample surface points (delegated to SurfaceSampler)
        3. Compute normals
        4. Compute curvature proxies
        5. Build feature tensor
    """

    def __init__(self, cfg: InputConfig):
        self.cfg = cfg

    def canonicalize(
        self, vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Center and scale vertices to [-1, 1]^3.

        Returns:
            vertices: canonicalized
            center: original center
            scale: original scale
        """
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
        vertices = vertices - center
        scale = np.abs(vertices).max()
        if scale > 1e-10:
            vertices = vertices / scale
        return vertices, center, float(scale)

    def compute_vertex_normals(
        self, vertices: np.ndarray, faces: np.ndarray,
    ) -> np.ndarray:
        """Compute per-vertex normals by area-weighted face normal averaging."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3)

        vertex_normals = np.zeros_like(vertices)
        for i in range(3):
            np.add.at(vertex_normals, faces[:, i], face_normals)

        norms = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        vertex_normals /= norms
        return vertex_normals

    def compute_curvature_proxy(
        self, vertices: np.ndarray, faces: np.ndarray,
    ) -> np.ndarray:
        """Estimate mean curvature via Laplacian-Beltrami approximation.

        Fully vectorized — no Python loops. For each triangle we compute
        cotangent weights at all three corners at once and scatter them
        into the sparse Laplacian.
        """
        from scipy.sparse import coo_matrix

        V = vertices.shape[0]
        F = faces.shape[0]
        if F == 0:
            return np.zeros(V, dtype=np.float32)

        v0 = vertices[faces[:, 0]]  # (F, 3)
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # For each triangle, compute cotangent of each angle
        # cot at vertex 0 is the weight of edge (1, 2)
        def cot(a, b, c):
            # angle at vertex a between edges ab and ac
            ab = b - a
            ac = c - a
            # cot = (ab . ac) / |ab x ac|
            dot = np.einsum("ij,ij->i", ab, ac)
            cross = np.cross(ab, ac)
            norm = np.linalg.norm(cross, axis=-1) + 1e-12
            return dot / norm

        cot0 = cot(v0, v1, v2)  # weight for edge (1, 2)
        cot1 = cot(v1, v2, v0)  # weight for edge (2, 0)
        cot2 = cot(v2, v0, v1)  # weight for edge (0, 1)

        # Build rows, cols, weights vectors
        rows = np.concatenate([
            faces[:, 1], faces[:, 2],  # edge (1,2) contributions
            faces[:, 2], faces[:, 0],  # edge (2,0)
            faces[:, 0], faces[:, 1],  # edge (0,1)
        ])
        cols = np.concatenate([
            faces[:, 2], faces[:, 1],
            faces[:, 0], faces[:, 2],
            faces[:, 1], faces[:, 0],
        ])
        weights = np.concatenate([
            cot0 / 2, cot0 / 2,
            cot1 / 2, cot1 / 2,
            cot2 / 2, cot2 / 2,
        ])

        L = coo_matrix((weights, (rows, cols)), shape=(V, V)).tocsr()
        diag = np.array(L.sum(axis=1)).flatten()
        # Laplacian = D - L (where D is diagonal of row sums)
        L_diag = coo_matrix((diag, (np.arange(V), np.arange(V))), shape=(V, V)).tocsr()
        laplacian = L_diag - L

        delta = laplacian.dot(vertices)  # (V, 3)
        curvature = np.linalg.norm(delta, axis=-1) / 2.0  # (V,)
        return curvature.astype(np.float32)

    def build_features(
        self,
        points: np.ndarray,
        normals: np.ndarray | None = None,
        curvature: np.ndarray | None = None,
    ) -> np.ndarray:
        """Concatenate point features: [xyz, normals?, curvature?, constant?]."""
        parts = [points]
        if self.cfg.compute_normals and normals is not None:
            parts.append(normals)
        if self.cfg.compute_curvature and curvature is not None:
            parts.append(curvature[:, None] if curvature.ndim == 1 else curvature)
        if self.cfg.append_constant_channel:
            const = np.full((points.shape[0], 1), self.cfg.constant_channel_value, dtype=np.float32)
            parts.append(const)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def __call__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals_in: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Full preprocessing pipeline.

        Returns dict with:
            vertices: (V, 3) canonicalized
            faces: (F, 3)
            normals: (V, 3)
            curvature: (V,)
            center, scale: canonicalization params
        """
        if self.cfg.canonicalize:
            vertices, center, scale = self.canonicalize(vertices)
        else:
            center = np.zeros(3)
            scale = 1.0

        if normals_in is not None:
            normals = normals_in
        else:
            normals = self.compute_vertex_normals(vertices, faces) if faces.shape[0] > 0 else np.zeros_like(vertices)

        curvature = None
        if self.cfg.compute_curvature and faces.shape[0] > 0:
            try:
                curvature = self.compute_curvature_proxy(vertices, faces)
            except Exception:
                curvature = np.zeros(vertices.shape[0], dtype=np.float32)

        return {
            "vertices": vertices.astype(np.float32),
            "faces": faces,
            "normals": normals.astype(np.float32),
            "curvature": curvature,
            "center": center.astype(np.float32) if isinstance(center, np.ndarray) else np.array(center, dtype=np.float32),
            "scale": np.float32(scale),
        }
