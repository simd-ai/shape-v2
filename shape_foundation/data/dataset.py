"""Dataset classes for loading and serving preprocessed mesh data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from shape_foundation.configs.default import DataConfig, InputConfig, DatasetSourceConfig
from shape_foundation.data.preprocessing import MeshPreprocessor
from shape_foundation.data.sampling import SurfaceSampler
from shape_foundation.preprocessing.mesh_io import load_mesh


class MeshDataset(Dataset):
    """Unified mesh dataset that loads from multiple sources.

    Each sample is preprocessed, sampled, and returned as tensors
    ready for the backbone.
    """

    # Hash-split bucket count. 10_000 buckets give 0.01% granularity on
    # val_fraction and uniform distribution across thousands of files.
    _HASH_SPLIT_BUCKETS = 10_000

    def __init__(
        self,
        cfg: DataConfig,
        input_cfg: InputConfig,
        split: str = "train",
        transform: Any = None,
    ):
        self.cfg = cfg
        self.input_cfg = input_cfg
        self.split = split
        self.transform = transform
        self.preprocessor = MeshPreprocessor(input_cfg)
        self.sampler = SurfaceSampler(input_cfg)

        use_hash_split = (
            cfg.val_fraction > 0.0
            and split in ("train", "val")
        )

        # Collect all file paths from configured sources
        self.samples: list[dict[str, Any]] = []
        for source_cfg in cfg.sources:
            # Legacy per-source split field is honored only when hash-split
            # is disabled. When hash-split is active, every source contributes
            # to both train and val via the deterministic path-hash bucket.
            if not use_hash_split:
                if source_cfg.split != split and source_cfg.split != "all":
                    continue
            source_samples = self._discover_source(source_cfg)
            self.samples.extend(source_samples)

        if use_hash_split:
            self.samples = self._apply_hash_split(
                self.samples, cfg.val_fraction, split,
            )

    @classmethod
    def _apply_hash_split(
        cls,
        samples: list[dict[str, Any]],
        val_fraction: float,
        split: str,
    ) -> list[dict[str, Any]]:
        """Deterministic train/val split by md5 hash of the sample path.

        Each sample is placed in exactly one split based on a stable hash
        of its path, so a given file always lands in the same split across
        runs, ranks, and machines — no RNG seed required. Python's builtin
        `hash()` is salted per process (PYTHONHASHSEED), which is why we
        use hashlib.md5 here instead.
        """
        threshold = int(round(val_fraction * cls._HASH_SPLIT_BUCKETS))
        threshold = max(1, min(cls._HASH_SPLIT_BUCKETS - 1, threshold))
        want_val = split == "val"
        kept: list[dict[str, Any]] = []
        for s in samples:
            h = hashlib.md5(s["path"].encode("utf-8")).digest()
            bucket = int.from_bytes(h[:8], byteorder="big", signed=False) % cls._HASH_SPLIT_BUCKETS
            in_val = bucket < threshold
            if in_val == want_val:
                kept.append(s)
        return kept

    def _discover_source(self, src: DatasetSourceConfig) -> list[dict[str, Any]]:
        """Find mesh files for a dataset source."""
        root = Path(src.root) if src.root else Path(self.cfg.cache_dir) / src.name
        samples = []

        if not root.exists():
            return samples

        # look for preprocessed .pt files first
        pt_files = sorted(root.glob("**/*.pt"))
        if pt_files:
            for f in pt_files:
                samples.append({"path": str(f), "format": "pt", "source": src.name})
        else:
            # look for mesh files
            mesh_exts = ("*.obj", "*.stl", "*.ply", "*.off", "*.msh", "*.step", "*.stp", "*.glb", "*.gltf")
            for ext in mesh_exts:
                for f in sorted(root.glob(f"**/{ext}")):
                    samples.append({"path": str(f), "format": "mesh", "source": src.name})

        # look for labels JSON
        label_file = root / "labels.json"
        if label_file.exists():
            with open(label_file) as f:
                labels = json.load(f)
            label_map = {item["filename"]: item for item in labels if "filename" in item}
            for s in samples:
                fname = Path(s["path"]).stem
                if fname in label_map:
                    s["labels"] = label_map[fname]

        if src.max_samples > 0:
            samples = samples[:src.max_samples]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if sample["format"] == "pt":
            data = torch.load(sample["path"], map_location="cpu", weights_only=True)
            if self.transform is not None:
                data = self.transform(data)
            return data

        # Load and preprocess raw mesh
        mesh = load_mesh(sample["path"])
        processed = self.preprocessor(mesh.vertices, mesh.faces, mesh.normals)

        # Sample surface points
        sampled = self.sampler.sample(
            processed["vertices"],
            processed["faces"],
            processed["normals"],
            processed.get("curvature"),
        )

        # Build feature tensor
        features = self.preprocessor.build_features(
            sampled["points"],
            sampled["normals"],
            sampled.get("curvature"),
        )

        result = {
            "points": torch.from_numpy(sampled["points"]),  # (N, 3)
            "features": torch.from_numpy(features),          # (N, F)
            "normals": torch.from_numpy(sampled["normals"]),  # (N, 3)
            "source": sample["source"],
            "path": sample["path"],
        }

        if sampled.get("curvature") is not None:
            result["curvature"] = torch.from_numpy(
                sampled["curvature"][:, None] if sampled["curvature"].ndim == 1
                else sampled["curvature"]
            )

        # Attach labels if present
        if "labels" in sample:
            labels = sample["labels"]
            if "symmetry_type" in labels:
                sym_map = {"none": 0, "mirror_half": 1, "mirror_quarter": 2, "axisymmetric": 3, "cyclic_sector": 4}
                result["symmetry_label"] = torch.tensor(sym_map.get(labels["symmetry_type"], 0))
            if "symmetry_planes" in labels:
                planes = torch.tensor(labels["symmetry_planes"], dtype=torch.float32)
                result["symmetry_planes"] = planes
            if "symmetry_axes" in labels:
                axes = torch.tensor(labels["symmetry_axes"], dtype=torch.float32)
                result["symmetry_axes"] = axes
            if "primitive_labels" in labels:
                result["primitive_labels"] = torch.tensor(labels["primitive_labels"], dtype=torch.long)
            if "part_labels" in labels:
                result["part_labels"] = torch.tensor(labels["part_labels"], dtype=torch.long)
            if "reduction_type" in labels:
                red_map = {"none": 0, "mirror_half": 1, "mirror_quarter": 2, "axisymmetric_2d": 3, "cyclic_sector": 4, "extrusion_2d": 5}
                result["reduction_label"] = torch.tensor(red_map.get(labels["reduction_type"], 0))
            if "repeated_sectors" in labels:
                result["repeated_sectors"] = torch.tensor(float(labels["repeated_sectors"]))
            if "constant_cross_section" in labels:
                result["constant_cross_section"] = torch.tensor(float(labels["constant_cross_section"]))

        if self.transform is not None:
            result = self.transform(result)

        return result


class CollateFunction:
    """Custom collate function that handles variable-size data."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Tensor fields: stack if same shape, else pad
        result: dict[str, Any] = {}
        keys = batch[0].keys()

        for key in keys:
            values = [b[key] for b in batch if key in b]
            if not values:
                continue

            if isinstance(values[0], torch.Tensor):
                if all(v.shape == values[0].shape for v in values):
                    result[key] = torch.stack(values)
                else:
                    # pad to max size
                    max_shape = [max(v.shape[d] for v in values) for d in range(values[0].ndim)]
                    padded = []
                    for v in values:
                        pad_widths = []
                        for d in range(v.ndim - 1, -1, -1):
                            pad_widths.extend([0, max_shape[d] - v.shape[d]])
                        padded.append(torch.nn.functional.pad(v, pad_widths))
                    result[key] = torch.stack(padded)
            elif isinstance(values[0], str):
                result[key] = values
            else:
                result[key] = values

        return result


def build_dataloader(
    cfg: DataConfig,
    input_cfg: InputConfig,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader for the given config and split."""
    dataset = MeshDataset(cfg, input_cfg, split=split)
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=CollateFunction(),
        drop_last=shuffle,
    )
