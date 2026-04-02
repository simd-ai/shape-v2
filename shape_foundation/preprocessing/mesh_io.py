"""Mesh loading and saving utilities supporting STEP, STL, OBJ, PLY, MSH formats."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class RawMesh(NamedTuple):
    """Minimal mesh representation."""
    vertices: np.ndarray   # (V, 3) float64
    faces: np.ndarray      # (F, 3) int64
    normals: np.ndarray | None = None  # (V, 3) or None


def load_mesh(path: str | Path) -> RawMesh:
    """Load a mesh from file. Supports STL, OBJ, PLY, OFF, MSH, STEP."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".stl", ".obj", ".ply", ".off", ".glb", ".gltf"):
        return _load_trimesh(path)
    elif suffix in (".msh",):
        return _load_meshio(path)
    elif suffix in (".step", ".stp"):
        return _load_step(path)
    else:
        # Fallback: try trimesh
        return _load_trimesh(path)


def _load_trimesh(path: Path) -> RawMesh:
    import trimesh
    mesh = trimesh.load(str(path), force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64) if mesh.vertex_normals is not None else None
    return RawMesh(vertices=verts, faces=faces, normals=normals)


def _load_meshio(path: Path) -> RawMesh:
    import meshio
    m = meshio.read(str(path))
    verts = np.asarray(m.points[:, :3], dtype=np.float64)
    # extract triangles from cell blocks
    faces_list = []
    for cell_block in m.cells:
        if cell_block.type == "triangle":
            faces_list.append(cell_block.data)
    if faces_list:
        faces = np.concatenate(faces_list, axis=0).astype(np.int64)
    else:
        # try to extract surface triangles from tetrahedra
        for cell_block in m.cells:
            if cell_block.type == "tetra":
                tets = cell_block.data
                # extract all 4 faces per tet
                f0 = tets[:, [0, 1, 2]]
                f1 = tets[:, [0, 1, 3]]
                f2 = tets[:, [0, 2, 3]]
                f3 = tets[:, [1, 2, 3]]
                all_faces = np.concatenate([f0, f1, f2, f3], axis=0)
                # keep boundary faces (appear once)
                sorted_faces = np.sort(all_faces, axis=1)
                _, idx, counts = np.unique(
                    sorted_faces, axis=0, return_index=True, return_counts=True
                )
                boundary = all_faces[idx[counts == 1]]
                faces_list.append(boundary)
        faces = np.concatenate(faces_list, axis=0).astype(np.int64) if faces_list else np.zeros((0, 3), dtype=np.int64)
    return RawMesh(vertices=verts, faces=faces)


def _load_step(path: Path) -> RawMesh:
    """Load STEP/STP via trimesh's built-in or gmsh fallback."""
    try:
        import trimesh
        mesh = trimesh.load(str(path), force="mesh")
        return RawMesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            normals=np.asarray(mesh.vertex_normals, dtype=np.float64),
        )
    except Exception:
        pass
    # gmsh fallback
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(str(path))
    gmsh.model.mesh.generate(2)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    verts = np.array(coords, dtype=np.float64).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=2)
    faces_list = []
    for et, nodes in zip(elem_types, elem_nodes):
        if et == 2:  # triangle
            tri_nodes = np.array(nodes, dtype=np.int64).reshape(-1, 3)
            mapped = np.vectorize(tag_to_idx.get)(tri_nodes)
            faces_list.append(mapped)
    gmsh.finalize()
    faces = np.concatenate(faces_list, axis=0) if faces_list else np.zeros((0, 3), dtype=np.int64)
    return RawMesh(vertices=verts, faces=faces)


def save_mesh(mesh: RawMesh, path: str | Path) -> None:
    """Save a mesh to STL/OBJ/PLY via trimesh."""
    import trimesh
    t = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    t.export(str(path))
