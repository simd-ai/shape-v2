"""Topology-reduction recommendation task.

Aggregates symmetry, primitive, and topology predictions into
a structured simulation-reduction recommendation.
"""

from __future__ import annotations

from typing import Any

import torch

from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.models.heads import SymmetryHead, TopologyReductionHead
from shape_foundation.tasks.symmetry import SymmetryDetector
from shape_foundation.tasks.primitives import PrimitiveDetector


class ReductionRecommender:
    """Produce structured simulation-reduction recommendations.

    Output schema:
    {
        "description": "...",
        "symmetry": {type, confidence, planes, axes},
        "primitives": [...],
        "topology": {repeated_sectors, constant_cross_section, thin_regions},
        "simulation_hints": {
            "recommended_reduction": "...",
            "reasoning": [...],
            "confidence": float
        }
    }
    """

    def __init__(self, model: GAOTBackbone, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.sym_detector = SymmetryDetector(model, device)
        self.prim_detector = PrimitiveDetector(model, device)

    @torch.no_grad()
    def recommend(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None = None,
        curvature: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full inference pipeline producing structured recommendation."""
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

        # Run full forward
        out = self.model.forward_features(points, features, normals, curvature)
        heads = out["heads"]

        # Symmetry
        symmetry = self._extract_symmetry(heads)

        # Primitives + topology
        primitives, topology = self._extract_primitives(heads)

        # Reduction
        reduction, reasoning, confidence = self._extract_reduction(heads, symmetry, topology)

        # Description
        from shape_foundation.tasks.captioning import GeometryCaptioner
        captioner = GeometryCaptioner.__new__(GeometryCaptioner)
        description = captioner._template_caption(heads)

        return {
            "description": description,
            "symmetry": symmetry,
            "primitives": primitives,
            "topology": topology,
            "simulation_hints": {
                "recommended_reduction": reduction,
                "reasoning": reasoning,
                "confidence": confidence,
            },
        }

    def _extract_symmetry(self, heads: dict) -> dict[str, Any]:
        if "symmetry" not in heads:
            return {"type": "none", "confidence": 0.0, "planes": [], "axes": []}

        sym = heads["symmetry"]
        probs = sym["probs"][0].cpu().numpy()
        pred_idx = int(probs.argmax())
        confidence = float(sym["confidence"][0].max().cpu().numpy())

        planes = []
        if "planes" in sym:
            p = sym["planes"][0].cpu().numpy()
            for i in range(p.shape[0]):
                if sym["confidence"][0, i].item() > 0.3:
                    planes.append(p[i].tolist())

        axes = []
        if "axes" in sym:
            a = sym["axes"][0].cpu().numpy()
            for i in range(a.shape[0]):
                if sym["confidence"][0, i].item() > 0.3:
                    axes.append(a[i].tolist())

        return {
            "type": SymmetryHead.LABELS[pred_idx],
            "confidence": confidence,
            "planes": planes,
            "axes": axes,
        }

    def _extract_primitives(self, heads: dict) -> tuple[list, dict]:
        from shape_foundation.models.heads import PrimitiveTopologyHead

        primitives = []
        topology: dict[str, Any] = {
            "repeated_sectors": 0,
            "constant_cross_section": False,
            "thin_regions": False,
        }

        if "primitive" not in heads:
            return primitives, topology

        prim = heads["primitive"]
        probs = prim["primitive_probs"][0].cpu().numpy()
        labels = probs.argmax(axis=-1)
        T = labels.shape[0]

        for i, name in enumerate(PrimitiveTopologyHead.PRIMITIVES):
            count = int((labels == i).sum())
            if count > 0:
                primitives.append({
                    "type": name,
                    "ratio": count / T,
                })

        if "repeated_sector_count" in prim:
            n = prim["repeated_sector_count"][0].item()
            topology["repeated_sectors"] = int(round(n)) if n > 1.5 else 0

        if "constant_cross_section" in prim:
            topology["constant_cross_section"] = prim["constant_cross_section"][0].item() > 0.5

        return primitives, topology

    def _extract_reduction(
        self, heads: dict, symmetry: dict, topology: dict,
    ) -> tuple[str, list[str], float]:
        """Extract reduction recommendation with reasoning."""
        if "reduction" not in heads:
            return "none", [], 0.0

        red = heads["reduction"]
        probs = red["reduction_probs"][0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_type = TopologyReductionHead.REDUCTIONS[pred_idx]
        confidence = float(red["confidence"][0].cpu().numpy())

        # Build reasoning from detected features
        reasoning = []
        reason_flags = red["reason_flags"][0].cpu().numpy()

        reason_descriptions = [
            "Mirror symmetry plane detected",
            "Multiple symmetry planes detected",
            "Rotational symmetry axis detected",
            "Continuous rotational symmetry (axisymmetry)",
            "Repeated angular sectors detected",
            "Constant cross-section along an axis",
            "Dominant planar/cylindrical surfaces",
            "Geometry is compact and bounded",
        ]

        for i, (flag, desc) in enumerate(zip(reason_flags, reason_descriptions)):
            if flag > 0.5:
                reasoning.append(desc)

        # Add symmetry-based reasoning
        if symmetry["type"] != "none":
            reasoning.append(f"Symmetry type: {symmetry['type']}")
        if topology.get("repeated_sectors", 0) > 1:
            reasoning.append(f"Repeated sectors: {topology['repeated_sectors']}")
        if topology.get("constant_cross_section"):
            reasoning.append("Constant cross-section detected")

        return pred_type, reasoning, confidence
