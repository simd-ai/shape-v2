"""Synthetic label generation for symmetry, primitives, and reduction recommendations.

Generates training labels from CAD/B-rep/procedural geometry via
geometric heuristics: reflection symmetry, rotational symmetry,
axisymmetry, primitive fitting, repeated sectors, and reduction types.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from shape_foundation.configs.default import SyntheticLabelConfig


class SyntheticLabelGenerator:
    """Generate symmetry, primitive, and reduction labels for meshes."""

    def __init__(self, cfg: SyntheticLabelConfig):
        self.cfg = cfg

    def generate_all(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray | None = None,
    ) -> dict:
        """Generate all synthetic labels for a mesh.

        Returns dict with keys matching the label schema expected by MeshDataset.
        """
        labels: dict = {}

        sym = self.detect_symmetry(vertices)
        labels["symmetry_type"] = sym["type"]
        labels["symmetry_planes"] = sym.get("planes", [])
        labels["symmetry_axes"] = sym.get("axes", [])

        prims = self.detect_primitives(vertices, faces, normals)
        labels["primitive_labels"] = prims["labels"]

        topo = self.detect_topology(vertices, faces)
        labels["repeated_sectors"] = topo["repeated_sectors"]
        labels["constant_cross_section"] = topo["constant_cross_section"]

        reduction = self.recommend_reduction(sym, prims, topo)
        labels["reduction_type"] = reduction

        return labels

    # ------------------------------------------------------------------
    # Symmetry detection
    # ------------------------------------------------------------------

    def detect_symmetry(self, vertices: np.ndarray) -> dict:
        """Detect reflection and rotational symmetry via point cloud matching."""
        result = {"type": "none", "planes": [], "axes": []}

        # test reflection symmetry for 3 cardinal planes + 6 diagonals
        candidates = self._symmetry_plane_candidates()
        best_plane = None
        best_score = float("inf")

        tree = KDTree(vertices)
        for normal, offset in candidates:
            reflected = self._reflect_points(vertices, normal, offset)
            dists, _ = tree.query(reflected)
            mean_dist = dists.mean()
            if mean_dist < self.cfg.symmetry_plane_threshold:
                result["planes"].append(normal.tolist() + [offset])
                if mean_dist < best_score:
                    best_score = mean_dist
                    best_plane = (normal, offset)

        # count mirror planes
        n_planes = len(result["planes"])
        if n_planes >= 2:
            result["type"] = "mirror_quarter"
        elif n_planes == 1:
            result["type"] = "mirror_half"

        # test rotational symmetry
        axis_candidates = self._axis_candidates()
        for axis, point in axis_candidates:
            is_axi, n_sectors = self._test_rotational_symmetry(vertices, axis, point, tree)
            if is_axi:
                result["axes"].append(axis.tolist() + point.tolist())
                if n_sectors >= 360:  # continuous
                    result["type"] = "axisymmetric"
                elif n_sectors > 1 and result["type"] not in ("axisymmetric",):
                    result["type"] = "cyclic_sector"

        return result

    def _symmetry_plane_candidates(self) -> list[tuple[np.ndarray, float]]:
        """Generate candidate symmetry planes to test."""
        planes = []
        # cardinal planes through origin
        for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
            planes.append((axis.astype(np.float64), 0.0))
        # diagonal planes
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            n = np.zeros(3)
            n[a] = 1.0
            n[b] = 1.0
            n = n / np.linalg.norm(n)
            planes.append((n, 0.0))
            n2 = np.zeros(3)
            n2[a] = 1.0
            n2[b] = -1.0
            n2 = n2 / np.linalg.norm(n2)
            planes.append((n2, 0.0))
        return planes

    def _axis_candidates(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate candidate rotation axes."""
        origin = np.zeros(3)
        axes = [
            (np.array([1, 0, 0], dtype=np.float64), origin),
            (np.array([0, 1, 0], dtype=np.float64), origin),
            (np.array([0, 0, 1], dtype=np.float64), origin),
        ]
        return axes

    @staticmethod
    def _reflect_points(
        points: np.ndarray, normal: np.ndarray, offset: float,
    ) -> np.ndarray:
        """Reflect points across a plane defined by normal and offset."""
        d = points @ normal - offset
        return points - 2 * d[:, None] * normal[None, :]

    def _test_rotational_symmetry(
        self,
        vertices: np.ndarray,
        axis: np.ndarray,
        point: np.ndarray,
        tree: KDTree,
    ) -> tuple[bool, int]:
        """Test if the point cloud has rotational symmetry about an axis.

        Returns (is_symmetric, num_sectors) where num_sectors=360 means
        continuous (axisymmetric).
        """
        n_steps = self.cfg.rotational_angle_steps
        threshold = self.cfg.axisymmetry_threshold

        best_n = 0
        # test discrete angles: 360/n for n in [2, 3, 4, 5, 6, 8, 10, 12, ...]
        test_n_values = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36]

        for n in test_n_values:
            angle = 2 * np.pi / n
            rotated = self._rotate_points(vertices, axis, point, angle)
            dists, _ = tree.query(rotated)
            if dists.mean() < threshold:
                best_n = max(best_n, n)

        # test near-continuous (many fine angles)
        continuous = True
        step = 2 * np.pi / n_steps
        for i in range(1, min(n_steps, 12)):
            angle = i * step
            rotated = self._rotate_points(vertices, axis, point, angle)
            dists, _ = tree.query(rotated)
            if dists.mean() >= threshold:
                continuous = False
                break

        if continuous:
            return True, 360
        if best_n > 1:
            return True, best_n
        return False, 0

    @staticmethod
    def _rotate_points(
        points: np.ndarray, axis: np.ndarray, center: np.ndarray, angle: float,
    ) -> np.ndarray:
        """Rotate points about axis through center by angle (radians)."""
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        pts = points - center
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # Rodrigues' rotation
        rotated = (
            pts * cos_a +
            np.cross(axis, pts) * sin_a +
            axis * (pts @ axis)[:, None] * (1 - cos_a)
        )
        return rotated + center

    # ------------------------------------------------------------------
    # Primitive detection
    # ------------------------------------------------------------------

    def detect_primitives(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray | None = None,
    ) -> dict:
        """Per-vertex primitive type classification via local geometry.

        Returns dict with 'labels': (V,) int array.
        Mapping: 0=plane, 1=cylinder, 2=cone, 3=sphere, 4=torus, 5=freeform
        """
        V = vertices.shape[0]
        labels = np.full(V, 5, dtype=np.int64)  # default: freeform

        if normals is None or faces.shape[0] == 0:
            return {"labels": labels.tolist()}

        # build adjacency
        from collections import defaultdict
        adj: dict[int, set[int]] = defaultdict(set)
        for f in faces:
            for i in range(3):
                adj[f[i]].add(f[(i + 1) % 3])
                adj[f[i]].add(f[(i + 2) % 3])

        threshold = self.cfg.primitive_fit_threshold

        for vi in range(V):
            neighbors = list(adj.get(vi, set()))
            if len(neighbors) < 3:
                continue

            n_pts = vertices[neighbors]
            n_nrm = normals[neighbors]
            center_n = normals[vi]

            # check planar: all normals similar
            dots = n_nrm @ center_n
            if np.all(np.abs(dots) > 1 - threshold * 10):
                labels[vi] = 0  # plane
                continue

            # check spherical: normals point away from a common center
            # estimate center via least-squares
            diffs = n_pts - vertices[vi]
            dists = np.linalg.norm(diffs, axis=-1)
            if dists.std() < threshold * 5 and dists.mean() > 0.01:
                # roughly equidistant -> sphere candidate
                radial_dirs = diffs / np.clip(dists[:, None], 1e-10, None)
                alignment = np.abs((radial_dirs * n_nrm).sum(axis=-1))
                if alignment.mean() > 0.9:
                    labels[vi] = 3  # sphere
                    continue

            # check cylindrical: normals perpendicular to a common axis
            # use PCA on normals
            nrm_mean = n_nrm.mean(axis=0)
            nrm_centered = n_nrm - nrm_mean
            if nrm_centered.shape[0] >= 3:
                cov = nrm_centered.T @ nrm_centered
                eigvals = np.linalg.eigvalsh(cov)
                ratio = eigvals[0] / (eigvals[-1] + 1e-10)
                if ratio < 0.1:
                    labels[vi] = 1  # cylinder
                    continue

            # remaining: cone or torus heuristics
            # cone: normals converge to a point
            # torus: normals rotate in a plane
            # simplified: leave as freeform for now
            labels[vi] = 5

        return {"labels": labels.tolist()}

    # ------------------------------------------------------------------
    # Topology detection
    # ------------------------------------------------------------------

    def detect_topology(
        self, vertices: np.ndarray, faces: np.ndarray,
    ) -> dict:
        """Detect repeated sectors and constant cross-section regions."""
        result = {"repeated_sectors": 0, "constant_cross_section": 0.0}

        if faces.shape[0] == 0:
            return result

        # repeated sectors: check cross-section similarity at different angles
        # project to cylindrical coordinates around each axis
        for axis_idx in range(3):
            other = [i for i in range(3) if i != axis_idx]
            xy = vertices[:, other]
            angles = np.arctan2(xy[:, 1], xy[:, 0])  # (-pi, pi)
            radii = np.linalg.norm(xy, axis=-1)

            # bin by angle and compare radial profiles
            n_bins = 36
            bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
            profiles = []
            for i in range(n_bins):
                mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
                if mask.sum() > 5:
                    r = radii[mask]
                    z = vertices[mask, axis_idx]
                    profiles.append((np.sort(r), np.sort(z)))

            if len(profiles) >= n_bins * 0.8:
                # compare adjacent profiles
                similarities = []
                for i in range(len(profiles) - 1):
                    r1, r2 = profiles[i][0], profiles[i + 1][0]
                    min_len = min(len(r1), len(r2))
                    if min_len > 3:
                        sim = 1 - np.mean(np.abs(r1[:min_len] - r2[:min_len]))
                        similarities.append(sim)
                if similarities and np.mean(similarities) > 0.95:
                    result["repeated_sectors"] = n_bins
                    break

        # constant cross-section: check if slices along each axis are similar
        for axis_idx in range(3):
            z_vals = vertices[:, axis_idx]
            z_min, z_max = z_vals.min(), z_vals.max()
            if z_max - z_min < 0.1:
                continue

            n_slices = 10
            slice_edges = np.linspace(z_min, z_max, n_slices + 1)
            slice_areas = []
            for i in range(n_slices):
                mask = (z_vals >= slice_edges[i]) & (z_vals < slice_edges[i + 1])
                if mask.sum() > 3:
                    other = [j for j in range(3) if j != axis_idx]
                    xy = vertices[mask][:, other]
                    # approximate area via convex hull
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(xy)
                        slice_areas.append(hull.volume)  # 2D: volume = area
                    except Exception:
                        pass

            if len(slice_areas) >= n_slices * 0.7:
                areas = np.array(slice_areas)
                cv = areas.std() / (areas.mean() + 1e-10)
                if cv < 0.05:
                    result["constant_cross_section"] = 1.0
                    break

        return result

    # ------------------------------------------------------------------
    # Reduction recommendation
    # ------------------------------------------------------------------

    def recommend_reduction(
        self, symmetry: dict, primitives: dict, topology: dict,
    ) -> str:
        """Recommend simulation reduction based on detected features."""
        sym_type = symmetry["type"]
        has_sectors = topology["repeated_sectors"] > 1
        has_const_xs = topology["constant_cross_section"] > 0.5

        if sym_type == "axisymmetric":
            return "axisymmetric_2d"
        if has_sectors and symmetry.get("axes"):
            return "cyclic_sector"
        if sym_type == "mirror_quarter":
            return "mirror_quarter"
        if sym_type == "mirror_half":
            return "mirror_half"
        if has_const_xs:
            return "extrusion_2d"
        return "none"

    # ------------------------------------------------------------------
    # Procedural shape generation for training data
    # ------------------------------------------------------------------

    @staticmethod
    def generate_procedural_shapes(
        n_per_type: int = 100, noise_scale: float = 0.0,
    ) -> list[dict]:
        """Generate procedural shapes with known labels.

        Types: mirrored solids, revolved solids, repeated sectors,
        extrusions, perturbed near-symmetric, noisy/decimated.
        """
        shapes = []
        rng = np.random.default_rng(42)

        for _ in range(n_per_type):
            # --- Mirrored box ---
            w, h, d = rng.uniform(0.3, 1.0, 3)
            verts = np.array([
                [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
                [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d],
            ], dtype=np.float32)
            faces = np.array([
                [0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                [2,6,7],[2,7,3],[0,3,7],[0,7,4],[1,5,6],[1,6,2],
            ], dtype=np.int64)
            if noise_scale > 0:
                verts += rng.normal(0, noise_scale, verts.shape).astype(np.float32)
            shapes.append({
                "vertices": verts, "faces": faces,
                "symmetry_type": "mirror_quarter",
                "reduction_type": "mirror_quarter",
                "symmetry_planes": [[1,0,0,0],[0,1,0,0],[0,0,1,0]],
            })

        for _ in range(n_per_type):
            # --- Revolved solid (axisymmetric) ---
            n_ring = 32
            n_stack = 16
            profile_r = rng.uniform(0.3, 0.8, n_stack)
            profile_z = np.linspace(-1, 1, n_stack)
            angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            verts = []
            for si in range(n_stack):
                for ai in range(n_ring):
                    x = profile_r[si] * np.cos(angles[ai])
                    y = profile_r[si] * np.sin(angles[ai])
                    z = profile_z[si]
                    verts.append([x, y, z])
            verts = np.array(verts, dtype=np.float32)
            faces_list = []
            for si in range(n_stack - 1):
                for ai in range(n_ring):
                    i0 = si * n_ring + ai
                    i1 = si * n_ring + (ai + 1) % n_ring
                    i2 = (si + 1) * n_ring + (ai + 1) % n_ring
                    i3 = (si + 1) * n_ring + ai
                    faces_list.append([i0, i1, i2])
                    faces_list.append([i0, i2, i3])
            faces = np.array(faces_list, dtype=np.int64)
            if noise_scale > 0:
                verts += rng.normal(0, noise_scale, verts.shape).astype(np.float32)
            shapes.append({
                "vertices": verts, "faces": faces,
                "symmetry_type": "axisymmetric",
                "reduction_type": "axisymmetric_2d",
                "symmetry_axes": [[0,0,1,0,0,0]],
            })

        for _ in range(n_per_type):
            # --- Cyclic sector (n-fold symmetry) ---
            n_fold = rng.choice([3, 4, 5, 6, 8])
            n_ring = n_fold * 4
            r_base = rng.uniform(0.5, 0.9)
            modulation = rng.uniform(0.05, 0.2)
            verts = []
            n_layers = 8
            for zi in range(n_layers):
                z = -1 + 2 * zi / (n_layers - 1)
                for ai in range(n_ring):
                    angle = 2 * np.pi * ai / n_ring
                    r = r_base + modulation * np.cos(n_fold * angle)
                    verts.append([r * np.cos(angle), r * np.sin(angle), z])
            verts = np.array(verts, dtype=np.float32)
            faces_list = []
            for zi in range(n_layers - 1):
                for ai in range(n_ring):
                    i0 = zi * n_ring + ai
                    i1 = zi * n_ring + (ai + 1) % n_ring
                    i2 = (zi + 1) * n_ring + (ai + 1) % n_ring
                    i3 = (zi + 1) * n_ring + ai
                    faces_list.append([i0, i1, i2])
                    faces_list.append([i0, i2, i3])
            faces = np.array(faces_list, dtype=np.int64)
            shapes.append({
                "vertices": verts, "faces": faces,
                "symmetry_type": "cyclic_sector",
                "reduction_type": "cyclic_sector",
                "repeated_sectors": int(n_fold),
            })

        for _ in range(n_per_type):
            # --- Extrusion (constant cross-section) ---
            n_profile = rng.integers(4, 12)
            angles = np.linspace(0, 2 * np.pi, n_profile, endpoint=False)
            radii = rng.uniform(0.3, 0.8, n_profile)
            profile = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)
            n_layers = 8
            verts = []
            for zi in range(n_layers):
                z = -1 + 2 * zi / (n_layers - 1)
                for p in profile:
                    verts.append([p[0], p[1], z])
            verts = np.array(verts, dtype=np.float32)
            faces_list = []
            for zi in range(n_layers - 1):
                for pi in range(n_profile):
                    i0 = zi * n_profile + pi
                    i1 = zi * n_profile + (pi + 1) % n_profile
                    i2 = (zi + 1) * n_profile + (pi + 1) % n_profile
                    i3 = (zi + 1) * n_profile + pi
                    faces_list.append([i0, i1, i2])
                    faces_list.append([i0, i2, i3])
            faces = np.array(faces_list, dtype=np.int64)
            shapes.append({
                "vertices": verts, "faces": faces,
                "symmetry_type": "none",
                "reduction_type": "extrusion_2d",
                "constant_cross_section": 1.0,
            })

        for _ in range(n_per_type):
            # --- Asymmetric (no reduction) ---
            n_pts = rng.integers(20, 60)
            verts_raw = rng.uniform(-1, 1, (n_pts, 3)).astype(np.float32)
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(verts_raw)
                verts = verts_raw[hull.vertices]
                faces = hull.simplices.astype(np.int64)
                # reindex
                old_to_new = {old: new for new, old in enumerate(hull.vertices)}
                faces = np.vectorize(old_to_new.get)(faces)
            except Exception:
                verts = verts_raw[:8]
                faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]], dtype=np.int64)

            shapes.append({
                "vertices": verts, "faces": faces,
                "symmetry_type": "none",
                "reduction_type": "none",
            })

        return shapes
