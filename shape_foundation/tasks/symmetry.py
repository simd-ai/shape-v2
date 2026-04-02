"""Symmetry detection task: detect and classify simulation-relevant symmetries."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.models.heads import SymmetryHead


class SymmetryDetector:
    """High-level symmetry detection from a trained backbone.

    Runs the backbone + symmetry head, then post-processes
    predictions into structured outputs.
    """

    LABELS = SymmetryHead.LABELS

    def __init__(self, model: GAOTBackbone, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None = None,
        curvature: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Detect symmetry in a mesh.

        Args:
            points: (1, N, 3) or (N, 3) surface points
            features: (1, N, F) or (N, F) point features

        Returns dict with:
            type: str (e.g., "mirror_half", "axisymmetric")
            confidence: float
            planes: list of [nx, ny, nz, d]
            axes: list of [dx, dy, dz, px, py, pz]
            all_probs: dict mapping type -> probability
        """
        if points.ndim == 2:
            points = points.unsqueeze(0)
            features = features.unsqueeze(0)
            if normals is not None:
                normals = normals.unsqueeze(0)
            if curvature is not None:
                curvature = curvature.unsqueeze(0)

        points = points.to(self.device)
        features = features.to(self.device)
        if normals is not None:
            normals = normals.to(self.device)
        if curvature is not None:
            curvature = curvature.to(self.device)

        out = self.model.forward_features(points, features, normals, curvature)
        heads = out["heads"]

        if "symmetry" not in heads:
            return {"type": "none", "confidence": 0.0, "planes": [], "axes": []}

        sym_out = heads["symmetry"]
        probs = sym_out["probs"][0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_type = self.LABELS[pred_idx]
        confidence = float(sym_out["confidence"][0].max().cpu().numpy())

        planes = []
        if "planes" in sym_out:
            p = sym_out["planes"][0].cpu().numpy()
            for i in range(p.shape[0]):
                if sym_out["confidence"][0, i].item() > 0.3:
                    planes.append(p[i].tolist())

        axes = []
        if "axes" in sym_out:
            a = sym_out["axes"][0].cpu().numpy()
            for i in range(a.shape[0]):
                if sym_out["confidence"][0, i].item() > 0.3:
                    axes.append(a[i].tolist())

        return {
            "type": pred_type,
            "confidence": confidence,
            "planes": planes,
            "axes": axes,
            "all_probs": {label: float(probs[i]) for i, label in enumerate(self.LABELS)},
        }
