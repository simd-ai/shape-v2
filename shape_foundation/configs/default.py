"""Dataclass-based configuration with all options from spec."""

from __future__ import annotations

import copy
import dataclasses as dc
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dc.dataclass
class InputConfig:
    """How raw meshes are consumed."""
    mode: str = "surface_sampled_points"  # vertices_only | surface_sampled_points | hybrid_vertices_plus_surface | feature_aware_sampling_near_sharp_edges
    num_surface_points: int = 8192
    canonicalize: bool = True
    compute_normals: bool = True
    compute_curvature: bool = True
    append_constant_channel: bool = True
    constant_channel_value: float = 1.0
    sharp_edge_ratio: float = 0.3  # fraction of samples near sharp edges (feature_aware mode)


@dc.dataclass
class NeighborConfig:
    """Neighbor search configuration."""
    backend: str = "torch_cluster"  # auto | torch_cluster | open3d | chunked | native
    strategy: str = "radius"  # radius | knn
    knn_k: int = 32
    multiscale_radii: list[float] = dc.field(default_factory=lambda: [0.05, 0.1, 0.2])
    max_neighbors: int = 64
    enable_caching: bool = True
    edge_drop_rate: float = 0.0
    ratio_sampling: float = 1.0  # subsample ratio for large point clouds


@dc.dataclass
class LatentGridConfig:
    """Latent tokenization."""
    strategy: str = "structured_latent_grid"  # structured_latent_grid | surface_adaptive_queries | multiresolution_latent_grid
    latent_shape: tuple[int, int, int] = (48, 48, 48)
    token_dim: int = 256
    adaptive_num_tokens: int = 4096  # for surface_adaptive_queries
    multires_shapes: list[tuple[int, int, int]] = dc.field(
        default_factory=lambda: [(24, 24, 24), (48, 48, 48)]
    )


@dc.dataclass
class GeoEmbedConfig:
    """Geometric embedding of neighborhoods."""
    mode: str = "statistical"  # statistical | pointnet | off
    stat_features: list[str] = dc.field(
        default_factory=lambda: ["mean", "std", "min", "max"]
    )
    mlp_hidden: int = 128
    mlp_layers: int = 2
    return_raw_stats: bool = True  # required for masked token and inpainting pretraining losses
    augment_normals: bool = True
    augment_curvature: bool = True
    augment_thickness: bool = False
    sdf_auxiliary_target: bool = False  # signed distance as auxiliary supervision only


@dc.dataclass
class AGNOConfig:
    """AGNO (Attentional Graph Neural Operator) attention."""
    similarity: str = "cosine"  # cosine | dot
    learned_temperature: bool = True
    initial_temperature: float = 0.1
    num_heads: int = 4
    dropout: float = 0.0


@dc.dataclass
class TokenizerConfig:
    """MAGNO encoder configuration."""
    latent: LatentGridConfig = dc.field(default_factory=LatentGridConfig)
    neighbor: NeighborConfig = dc.field(default_factory=NeighborConfig)
    geo_embed: GeoEmbedConfig = dc.field(default_factory=GeoEmbedConfig)
    agno: AGNOConfig = dc.field(default_factory=AGNOConfig)
    num_scales: int = 3  # number of multi-scale MAGNO layers
    scale_fusion: str = "concat_project"  # concat_project | sum | gated
    token_pos_encoding: bool = True  # add latent grid 3D positional encoding to tokens before processor


@dc.dataclass
class ProcessorConfig:
    """Transformer processor configuration."""
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 2  # for grouped-query attention; set == num_heads for MHA
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    patch_size: int = 6
    positional_embedding: str = "absolute"  # absolute | rope
    rope_theta: float = 10000.0
    norm_type: str = "rmsnorm"  # rmsnorm | layernorm
    uvit_skip: bool = False  # UViT-style long-range skip connections
    spatial_dims: int = 3  # 2 or 3
    mask_patch_threshold: float = 0.5  # fraction of masked tokens in patch to mark whole patch masked


@dc.dataclass
class SymmetryHeadConfig:
    enabled: bool = True
    num_classes: int = 5  # none, mirror_half, mirror_quarter, axisymmetric, cyclic_sector
    max_candidates: int = 3
    hidden_dim: int = 256
    regress_plane: bool = True
    regress_axis: bool = True


@dc.dataclass
class PrimitiveHeadConfig:
    enabled: bool = True
    num_primitive_types: int = 6  # plane, cylinder, cone, sphere, torus, generic_freeform
    detect_repeated_sectors: bool = True
    detect_constant_cross_section: bool = True
    hidden_dim: int = 256


@dc.dataclass
class PartHeadConfig:
    enabled: bool = True
    max_parts: int = 64
    hidden_dim: int = 256
    hierarchical: bool = True


@dc.dataclass
class CaptionHeadConfig:
    enabled: bool = True
    mode: str = "template"  # template | retrieval | llm
    vocab_size: int = 512  # for template token vocabulary
    max_length: int = 64
    hidden_dim: int = 256


@dc.dataclass
class ReductionHeadConfig:
    enabled: bool = True
    hidden_dim: int = 256


@dc.dataclass
class HeadsConfig:
    embedding_dim: int = 256
    pooling: str = "attention"  # mean | attention
    symmetry: SymmetryHeadConfig = dc.field(default_factory=SymmetryHeadConfig)
    primitive: PrimitiveHeadConfig = dc.field(default_factory=PrimitiveHeadConfig)
    part: PartHeadConfig = dc.field(default_factory=PartHeadConfig)
    caption: CaptionHeadConfig = dc.field(default_factory=CaptionHeadConfig)
    reduction: ReductionHeadConfig = dc.field(default_factory=ReductionHeadConfig)


# ---------------------------------------------------------------------------
# Loss / training
# ---------------------------------------------------------------------------

@dc.dataclass
class MaskedTokenLossConfig:
    enabled: bool = True
    # 0.5 forces the model to reconstruct from distant context rather than
    # local neighbors (which are too informative at 0.25). MAE uses 0.75 in
    # 2D — we stay at 0.5 for 3D because each latent-grid neighbor carries
    # less context than a 2D image patch.
    mask_ratio: float = 0.5
    mask_strategy: str = "random"  # random | block | hybrid | spatial_3d
    block_size: int = 4
    target: str = "geo_stats"  # geo_stats | coordinates


@dc.dataclass
class ContrastiveLossConfig:
    enabled: bool = True
    temperature: float = 0.07
    num_augmentations: int = 2
    # When True (default, safer), the clean and augmented views are
    # concatenated along the batch dim and run through a single DDP forward;
    # outputs are split afterwards. Avoids two separate forward passes before
    # a single backward, which is fragile with DDP + find_unused_parameters.
    # Set False to fall back to the legacy two-forward path for comparison.
    concat_forward: bool = True
    # Augmentation strength for the second view. Previously hardcoded to
    # jitter_std=0.005 and point_dropout=0.10, which made InfoNCE trivial —
    # contrastive_raw collapsed to ~0.003 by epoch 15 and contributed no
    # gradient afterwards. 0.02 / 0.30 keeps the two views meaningfully
    # different for the full 50 epochs without losing the "same mesh" signal.
    jitter_std: float = 0.02
    point_dropout: float = 0.30


@dc.dataclass
class InpaintingLossConfig:
    enabled: bool = True
    crop_ratio: float = 0.25
    crop_strategy: str = "spatial"  # spatial | random_region


@dc.dataclass
class SupervisedLossConfig:
    symmetry_weight: float = 1.0
    primitive_weight: float = 1.0
    part_weight: float = 1.0
    plane_regression_weight: float = 0.5
    axis_regression_weight: float = 0.5


@dc.dataclass
class RegressionLossConfig:
    """Controls the regression-style loss used by geometry reconstruction
    (masked-token, inpainting) and geometric regression (symmetry plane,
    symmetry axis). Classification losses are unaffected.
    """
    kind: str = "smooth_l1"  # "mse" | "smooth_l1"
    beta: float = 1.0        # SmoothL1 transition point; ignored for mse


@dc.dataclass
class ReconTargetNormConfig:
    """Per-dimension normalization of raw_geo_stats reconstruction targets.

    Computes (mean, std) per-dim from the training split only, over a pool
    of training batches collected at the start of training, then normalizes
    target = (target - mean) / (std + eps) before feeding into the
    reconstruction loss (masked-token, inpainting). Only the target is
    normalized — the predictor learns to output values in the normalized
    space directly.

    The computed stats are stored as registered buffers on LossComputer so
    they round-trip through checkpoint save/load and are inspectable via
    `trainer.loss_computer.recon_target_mean` / `.recon_target_std`.
    """
    enabled: bool = True
    n_calib_batches: int = 32
    eps: float = 1e-6


@dc.dataclass
class TextAlignLossConfig:
    enabled: bool = False
    temperature: float = 0.07


@dc.dataclass
class LossWeightsConfig:
    """Per-loss scalar weights applied before summing into total loss.

    Supervised defaults are 0.0 because the stock synthetic labels in
    data/synthetic_labels.py are memorizable per-file and overfit
    catastrophically (val CE ~2.5 vs train CE ~1e-4). Re-enable these
    per-config only once you are training against labels that genuinely
    generalize across unseen meshes.
    """
    masked_token: float = 1.0
    contrastive: float = 0.2
    inpainting: float = 0.5
    symmetry: float = 0.0
    primitive: float = 0.0
    part: float = 0.0
    reduction: float = 0.0
    text_align: float = 0.2


@dc.dataclass
class LossConfig:
    masked_token: MaskedTokenLossConfig = dc.field(default_factory=MaskedTokenLossConfig)
    contrastive: ContrastiveLossConfig = dc.field(default_factory=ContrastiveLossConfig)
    inpainting: InpaintingLossConfig = dc.field(default_factory=InpaintingLossConfig)
    supervised: SupervisedLossConfig = dc.field(default_factory=SupervisedLossConfig)
    regression: RegressionLossConfig = dc.field(default_factory=RegressionLossConfig)
    recon_target_norm: ReconTargetNormConfig = dc.field(default_factory=ReconTargetNormConfig)
    text_align: TextAlignLossConfig = dc.field(default_factory=TextAlignLossConfig)
    weights: LossWeightsConfig = dc.field(default_factory=LossWeightsConfig)


@dc.dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"  # cosine | linear | constant


@dc.dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "shape-foundation"
    entity: str = ""  # W&B team/user name, empty = default
    name: str = ""  # run name, empty = auto-generated
    tags: list[str] = dc.field(default_factory=list)


@dc.dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # bf16 | fp16 | fp32
    compile_model: bool = False  # torch.compile
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 2000
    # Heavy outputs default to /data/... to avoid filling the root disk.
    # Both paths can be overridden per-run from YAML (train.checkpoint_dir,
    # train.log_dir) or via the SHAPE_CHECKPOINT_DIR / SHAPE_LOG_DIR env vars.
    checkpoint_dir: str = "/data/shape-v2/checkpoints"
    log_dir: str = "/data/shape-v2/runs"
    optimizer: OptimizerConfig = dc.field(default_factory=OptimizerConfig)
    loss: LossConfig = dc.field(default_factory=LossConfig)
    wandb: WandbConfig = dc.field(default_factory=WandbConfig)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dc.dataclass
class DatasetSourceConfig:
    name: str = "abc"  # abc | shapenet | partnet | objaverse | objaverse_xl | thingi10k | shrec2022 | shrec2023 | scan2cad | fusion360 | mfcad | custom
    root: str = ""
    split: str = "train"
    max_samples: int = -1  # -1 = use all
    weight: float = 1.0  # sampling weight in mixed dataset


@dc.dataclass
class SyntheticLabelConfig:
    enabled: bool = True
    symmetry_plane_threshold: float = 0.01
    rotational_angle_steps: int = 36
    axisymmetry_threshold: float = 0.02
    primitive_fit_threshold: float = 0.005
    generate_perturbed: bool = True
    perturbation_scale: float = 0.01


@dc.dataclass
class DataConfig:
    sources: list[DatasetSourceConfig] = dc.field(
        default_factory=lambda: [DatasetSourceConfig(name="abc")]
    )
    cache_dir: str = "data_cache"
    # Deterministic hash-based train/val split applied to every source when > 0.
    # Each sample is assigned to val iff md5(path) % 10000 < val_fraction*10000,
    # so the same file always ends up in the same split across runs/ranks.
    # Set to 0.0 to opt out and fall back to the per-source `split:` field.
    val_fraction: float = 0.05
    synthetic_labels: SyntheticLabelConfig = dc.field(default_factory=SyntheticLabelConfig)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dc.dataclass
class ShapeConfig:
    """Top-level configuration for the Shape foundation model."""
    input: InputConfig = dc.field(default_factory=InputConfig)
    tokenizer: TokenizerConfig = dc.field(default_factory=TokenizerConfig)
    processor: ProcessorConfig = dc.field(default_factory=ProcessorConfig)
    heads: HeadsConfig = dc.field(default_factory=HeadsConfig)
    train: TrainConfig = dc.field(default_factory=TrainConfig)
    data: DataConfig = dc.field(default_factory=DataConfig)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _dataclass_to_dict(obj: Any) -> Any:
    if dc.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dataclass_to_dict(v) for k, v in dc.asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    if not dc.is_dataclass(cls):
        return data
    field_types = {f.name: f.type for f in dc.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        ft = field_types[key]
        # resolve string annotations
        if isinstance(ft, str):
            ft = eval(ft, {**globals(), **{cls.__name__: cls}})
        if dc.is_dataclass(ft) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(ft, value)
        elif hasattr(ft, "__origin__") and ft.__origin__ is list and isinstance(value, list):
            args = getattr(ft, "__args__", ())
            if args and dc.is_dataclass(args[0]):
                kwargs[key] = [_dict_to_dataclass(args[0], v) if isinstance(v, dict) else v for v in value]
            else:
                kwargs[key] = value
        elif isinstance(value, list) and hasattr(ft, "__origin__") and ft.__origin__ is tuple:
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ShapeConfig:
    """Load a ShapeConfig from a YAML file, merging with defaults."""
    cfg = ShapeConfig()
    p = Path(path)
    if p.exists():
        with open(p) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _dict_to_dataclass(ShapeConfig, {**_dataclass_to_dict(cfg), **_deep_merge(_dataclass_to_dict(cfg), overrides)})
    return cfg


def save_config(cfg: ShapeConfig, path: str | Path) -> None:
    """Save config to YAML."""
    with open(path, "w") as f:
        yaml.dump(_dataclass_to_dict(cfg), f, default_flow_style=False, sort_keys=False)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
