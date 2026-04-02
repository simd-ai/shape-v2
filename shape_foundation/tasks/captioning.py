"""Geometry captioning task: generate structural descriptions of meshes."""

from __future__ import annotations

from typing import Any

import torch

from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.models.heads import SymmetryHead, PrimitiveTopologyHead, TopologyReductionHead


class GeometryCaptioner:
    """Generate natural-language descriptions from model predictions.

    Uses template-based captioning by default: fills structured templates
    from predicted attributes (symmetry, primitives, topology).
    """

    def __init__(self, model: GAOTBackbone, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def caption(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: torch.Tensor | None = None,
        curvature: torch.Tensor | None = None,
    ) -> str:
        """Generate a structural description of the geometry."""
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

        return self._template_caption(heads)

    def _template_caption(self, heads: dict[str, Any]) -> str:
        """Build a description from template rules applied to head outputs."""
        parts = []

        # Overall shape description
        parts.append("This is a 3D geometry")

        # Symmetry
        if "symmetry" in heads:
            sym = heads["symmetry"]
            probs = sym["probs"][0].cpu()
            pred_idx = int(probs.argmax())
            pred_type = SymmetryHead.LABELS[pred_idx]
            conf = float(probs[pred_idx])

            sym_descriptions = {
                "none": "with no detected symmetry",
                "mirror_half": "with mirror symmetry (half-domain reduction possible)",
                "mirror_quarter": "with two planes of mirror symmetry (quarter-domain reduction possible)",
                "axisymmetric": "that is axisymmetric (2D axisymmetric reduction possible)",
                "cyclic_sector": "with cyclic/sector symmetry (sector reduction possible)",
            }
            desc = sym_descriptions.get(pred_type, "")
            if desc and conf > 0.5:
                parts.append(desc)

            if "planes" in sym and pred_type in ("mirror_half", "mirror_quarter"):
                planes = sym["planes"][0].cpu().numpy()
                n_planes = min(2, planes.shape[0])
                plane_strs = []
                for i in range(n_planes):
                    n = planes[i, :3]
                    axis_name = self._axis_name(n)
                    if axis_name:
                        plane_strs.append(f"the {axis_name} plane")
                if plane_strs:
                    parts.append(f"Symmetry plane(s): {', '.join(plane_strs)}.")

        # Primitives
        if "primitive" in heads:
            prim = heads["primitive"]
            probs = prim["primitive_probs"][0].cpu()  # (T, C)
            labels = probs.argmax(dim=-1)
            T = labels.shape[0]
            prim_names = PrimitiveTopologyHead.PRIMITIVES
            dominant = []
            for i, name in enumerate(prim_names):
                ratio = (labels == i).float().mean().item()
                if ratio > 0.1:
                    dominant.append(f"{name} ({ratio:.0%})")
            if dominant:
                parts.append(f"Dominant surface types: {', '.join(dominant)}.")

            if "repeated_sector_count" in prim:
                count = prim["repeated_sector_count"][0].item()
                if count > 1.5:
                    parts.append(f"Detected {int(round(count))} repeated sectors.")
            if "constant_cross_section" in prim:
                ccs = prim["constant_cross_section"][0].item()
                if ccs > 0.5:
                    parts.append("The geometry has a constant cross-section along one axis.")

        # Reduction recommendation
        if "reduction" in heads:
            red = heads["reduction"]
            probs = red["reduction_probs"][0].cpu()
            pred_idx = int(probs.argmax())
            pred_type = TopologyReductionHead.REDUCTIONS[pred_idx]
            conf = float(red["confidence"][0].cpu())

            if pred_type != "none" and conf > 0.5:
                reduction_descriptions = {
                    "mirror_half": "Recommended: solve only half the domain using mirror symmetry.",
                    "mirror_quarter": "Recommended: solve only a quarter of the domain using two symmetry planes.",
                    "axisymmetric_2d": "Recommended: reduce to a 2D axisymmetric simulation.",
                    "cyclic_sector": "Recommended: solve only one periodic sector.",
                    "extrusion_2d": "Recommended: reduce to a 2D cross-section simulation.",
                }
                desc = reduction_descriptions.get(pred_type, "")
                if desc:
                    parts.append(desc)

        return " ".join(parts)

    @staticmethod
    def _axis_name(normal: Any) -> str:
        """Map a normal vector to a human-readable axis name."""
        import numpy as np
        n = np.abs(np.array(normal))
        idx = int(n.argmax())
        if n[idx] > 0.9:
            return ["YZ (x-normal)", "XZ (y-normal)", "XY (z-normal)"][idx]
        return ""
