"""Surface sampling from triangle meshes."""

from __future__ import annotations

import numpy as np

from shape_foundation.configs.default import InputConfig


class SurfaceSampler:
    """Sample points from mesh surface with various strategies."""

    def __init__(self, cfg: InputConfig):
        self.cfg = cfg

    def sample(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray | None = None,
        curvature: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample surface points according to configured mode.

        Returns:
            points: (N, 3)
            normals: (N, 3)
            curvature: (N, 1) or None
            features: (N, F) concatenated features
        """
        mode = self.cfg.mode
        n = self.cfg.num_surface_points

        if mode == "vertices_only":
            return self._sample_vertices(vertices, normals, curvature, n)
        elif mode == "surface_sampled_points":
            return self._sample_surface(vertices, faces, normals, curvature, n)
        elif mode == "hybrid_vertices_plus_surface":
            return self._sample_hybrid(vertices, faces, normals, curvature, n)
        elif mode == "feature_aware_sampling_near_sharp_edges":
            return self._sample_feature_aware(vertices, faces, normals, curvature, n)
        else:
            raise ValueError(f"Unknown input mode: {mode}")

    def _sample_vertices(
        self, vertices: np.ndarray, normals: np.ndarray | None,
        curvature: np.ndarray | None, n: int,
    ) -> dict[str, np.ndarray]:
        V = vertices.shape[0]
        if V >= n:
            idx = np.random.choice(V, n, replace=False)
        else:
            idx = np.random.choice(V, n, replace=True)
        pts = vertices[idx]
        nrm = normals[idx] if normals is not None else np.zeros_like(pts)
        crv = curvature[idx] if curvature is not None else None
        return {"points": pts, "normals": nrm, "curvature": crv}

    def _sample_surface(
        self, vertices: np.ndarray, faces: np.ndarray,
        normals: np.ndarray | None, curvature: np.ndarray | None, n: int,
    ) -> dict[str, np.ndarray]:
        """Sample points uniformly on triangle surface, proportional to area."""
        if faces.shape[0] == 0:
            return self._sample_vertices(vertices, normals, curvature, n)

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # face areas
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=-1) / 2.0
        total_area = areas.sum()
        if total_area < 1e-12:
            return self._sample_vertices(vertices, normals, curvature, n)

        probs = areas / total_area

        # sample faces
        face_idx = np.random.choice(faces.shape[0], n, replace=True, p=probs)

        # random barycentric coordinates
        r1 = np.random.rand(n, 1).astype(np.float32)
        r2 = np.random.rand(n, 1).astype(np.float32)
        sqrt_r1 = np.sqrt(r1)
        bary = np.concatenate([1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2], axis=-1)  # (n, 3)

        # interpolate positions
        f = faces[face_idx]
        pts = (
            bary[:, 0:1] * vertices[f[:, 0]] +
            bary[:, 1:2] * vertices[f[:, 1]] +
            bary[:, 2:3] * vertices[f[:, 2]]
        )

        # interpolate normals
        if normals is not None:
            nrm = (
                bary[:, 0:1] * normals[f[:, 0]] +
                bary[:, 1:2] * normals[f[:, 1]] +
                bary[:, 2:3] * normals[f[:, 2]]
            )
            nrm_len = np.linalg.norm(nrm, axis=-1, keepdims=True)
            nrm = nrm / np.clip(nrm_len, 1e-10, None)
        else:
            # use face normals
            face_nrm = cross[face_idx]
            face_nrm_len = np.linalg.norm(face_nrm, axis=-1, keepdims=True)
            nrm = face_nrm / np.clip(face_nrm_len, 1e-10, None)

        # interpolate curvature
        crv = None
        if curvature is not None:
            crv = (
                bary[:, 0] * curvature[f[:, 0]] +
                bary[:, 1] * curvature[f[:, 1]] +
                bary[:, 2] * curvature[f[:, 2]]
            )

        return {"points": pts.astype(np.float32), "normals": nrm.astype(np.float32), "curvature": crv}

    def _sample_hybrid(
        self, vertices: np.ndarray, faces: np.ndarray,
        normals: np.ndarray | None, curvature: np.ndarray | None, n: int,
    ) -> dict[str, np.ndarray]:
        """Half from vertices, half from surface."""
        n_vert = n // 2
        n_surf = n - n_vert
        vert_sample = self._sample_vertices(vertices, normals, curvature, n_vert)
        surf_sample = self._sample_surface(vertices, faces, normals, curvature, n_surf)
        return {
            "points": np.concatenate([vert_sample["points"], surf_sample["points"]], axis=0),
            "normals": np.concatenate([vert_sample["normals"], surf_sample["normals"]], axis=0),
            "curvature": (
                np.concatenate([vert_sample["curvature"], surf_sample["curvature"]], axis=0)
                if vert_sample["curvature"] is not None and surf_sample["curvature"] is not None
                else None
            ),
        }

    def _sample_feature_aware(
        self, vertices: np.ndarray, faces: np.ndarray,
        normals: np.ndarray | None, curvature: np.ndarray | None, n: int,
    ) -> dict[str, np.ndarray]:
        """Oversample near sharp edges (high dihedral angle changes)."""
        if faces.shape[0] == 0 or normals is None:
            return self._sample_surface(vertices, faces, normals, curvature, n)

        # compute edge sharpness from dihedral angles
        from collections import defaultdict
        edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        for fi, f in enumerate(faces):
            for j in range(3):
                e = tuple(sorted([f[j], f[(j + 1) % 3]]))
                edge_faces[e].append(fi)

        # face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_norm = np.linalg.norm(fn, axis=-1, keepdims=True)
        fn = fn / np.clip(fn_norm, 1e-10, None)

        # vertex sharpness = max dihedral angle at adjacent edges
        vertex_sharp = np.zeros(vertices.shape[0], dtype=np.float32)
        for (vi, vj), flist in edge_faces.items():
            if len(flist) == 2:
                cos_angle = np.dot(fn[flist[0]], fn[flist[1]])
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                vertex_sharp[vi] = max(vertex_sharp[vi], angle)
                vertex_sharp[vj] = max(vertex_sharp[vj], angle)

        # weight surface sampling by sharpness
        face_sharpness = (
            vertex_sharp[faces[:, 0]] +
            vertex_sharp[faces[:, 1]] +
            vertex_sharp[faces[:, 2]]
        ) / 3.0

        # blend: (1 - ratio) * area + ratio * sharpness
        areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1) / 2.0
        total_area = areas.sum()
        if total_area < 1e-12:
            return self._sample_vertices(vertices, normals, curvature, n)

        ratio = self.cfg.sharp_edge_ratio
        area_weight = areas / total_area
        sharp_total = face_sharpness.sum()
        sharp_weight = face_sharpness / sharp_total if sharp_total > 1e-12 else area_weight
        combined = (1 - ratio) * area_weight + ratio * sharp_weight
        combined = combined / combined.sum()

        # sample
        face_idx = np.random.choice(faces.shape[0], n, replace=True, p=combined)
        r1 = np.random.rand(n, 1).astype(np.float32)
        r2 = np.random.rand(n, 1).astype(np.float32)
        sqrt_r1 = np.sqrt(r1)
        bary = np.concatenate([1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2], axis=-1)

        f = faces[face_idx]
        pts = bary[:, 0:1] * vertices[f[:, 0]] + bary[:, 1:2] * vertices[f[:, 1]] + bary[:, 2:3] * vertices[f[:, 2]]
        nrm = bary[:, 0:1] * normals[f[:, 0]] + bary[:, 1:2] * normals[f[:, 1]] + bary[:, 2:3] * normals[f[:, 2]]
        nrm = nrm / np.clip(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-10, None)

        crv = None
        if curvature is not None:
            crv = bary[:, 0] * curvature[f[:, 0]] + bary[:, 1] * curvature[f[:, 1]] + bary[:, 2] * curvature[f[:, 2]]

        return {"points": pts.astype(np.float32), "normals": nrm.astype(np.float32), "curvature": crv}
