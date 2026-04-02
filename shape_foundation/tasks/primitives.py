"""Primitive detection task: identify geometric primitives in mesh regions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.models.heads import PrimitiveTopologyHead


class PrimitiveDetector:
    """High-level primitive detection from a trained backbone."""

    PRIMITIVES = PrimitiveTopologyHead.PRIMITIVES

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
        """Detect primitives in a mesh.

        Returns:
            primitives: list of {type, ratio, token_indices}
            topology: {repeated_sectors, constant_cross_section}
            per_token_labels: (T,) int array of predicted primitive types
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

        if "primitive" not in heads:
            return {"primitives": [], "topology": {}, "per_token_labels": []}

        prim_out = heads["primitive"]
        probs = prim_out["primitive_probs"][0].cpu().numpy()  # (T, C)
        labels = probs.argmax(axis=-1)  # (T,)

        # aggregate: count each primitive type
        primitives = []
        T = labels.shape[0]
        for prim_idx, prim_name in enumerate(self.PRIMITIVES):
            mask = labels == prim_idx
            count = int(mask.sum())
            if count > 0:
                primitives.append({
                    "type": prim_name,
                    "ratio": count / T,
                    "token_count": count,
                    "token_indices": np.where(mask)[0].tolist(),
                })

        topology = {}
        if "repeated_sector_count" in prim_out:
            topology["repeated_sectors"] = float(prim_out["repeated_sector_count"][0].cpu().numpy())
        if "constant_cross_section" in prim_out:
            topology["constant_cross_section"] = float(prim_out["constant_cross_section"][0].cpu().numpy())

        return {
            "primitives": primitives,
            "topology": topology,
            "per_token_labels": labels.tolist(),
        }
