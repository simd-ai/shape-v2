# Shape Foundation Model

3D geometry foundation model for simulation-relevant mesh analysis. Given a mesh of a physical domain, the model produces geometry embeddings, detects symmetry/primitives/topology, and recommends simulation reductions (mirror symmetry, axisymmetry, cyclic sectors, extrusions).

## Architecture

**GAOTBackbone** = MAGNO Encoder + Transformer Processor + Task Heads

- **MAGNO Encoder**: Multi-scale cross-attention from a structured 3D latent grid to physical surface points, with explicit geometric embeddings (statistical neighborhood features) and AGNO (cosine-similarity) attention.
- **Transformer Processor**: 3D patchification → grouped-query attention with RMSNorm → unpatchification. Supports absolute and RoPE positional embeddings, optional UViT skip connections.
- **Task Heads**: Geometry embedding, symmetry detection (classification + plane/axis regression), primitive recognition (per-token), part segmentation, template captioning, topology-reduction recommendation.

## Quick Start

```bash
# Install
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Generate synthetic training data
python -m shape_foundation.scripts.prepare_dataset --generate-synthetic --n-per-type 500

# Smoke test (tiny model, 2 epochs)
python -m shape_foundation.scripts.train_pretrain --config configs/smoke_test.yaml --dry-run
python -m shape_foundation.scripts.train_pretrain --config configs/smoke_test.yaml

# Inference
python -m shape_foundation.scripts.infer_mesh --checkpoint checkpoints/checkpoint_final.pt --mesh path/to/mesh.stl --verbose
```

## Training

### Stage 1: Self-Supervised Pretraining

Train on ABC + ShapeNet + Objaverse + Thingi10K with masked token modeling, multi-resolution contrastive learning, and partial inpainting.

```bash
# Prepare real datasets (requires downloading to data_cache/<name>/)
python -m shape_foundation.scripts.prepare_dataset --source abc --root /data/abc --output data_cache/abc
python -m shape_foundation.scripts.prepare_dataset --source shapenet --root /data/ShapeNet --output data_cache/shapenet

# Single GPU — medium config (recommended default)
python -m shape_foundation.scripts.train_pretrain --config configs/medium.yaml

# Multi-GPU DDP
torchrun --nproc_per_node=4 -m shape_foundation.scripts.train_pretrain --config configs/large.yaml

# Override specific settings
python -m shape_foundation.scripts.train_pretrain --config configs/medium.yaml --epochs 50 --batch-size 64 --lr 1e-4
```

### Stage 2: Supervised Fine-tuning

Add PartNet + Fusion360 + MFCAD++ + SHREC 2022/2023 for supervised heads.

```bash
python -m shape_foundation.scripts.train_finetune \
    --config configs/medium.yaml \
    --pretrained checkpoints/checkpoint_final.pt \
    --freeze-backbone-epochs 5

# Multi-GPU
torchrun --nproc_per_node=4 -m shape_foundation.scripts.train_finetune \
    --config configs/large.yaml \
    --pretrained checkpoints/checkpoint_final.pt
```

### Stage 3: Symmetry/Reduction Fine-tuning

Fine-tune on SHREC 2023, Scan2CAD, and synthetic reduction labels.

```bash
# Generate synthetic labels with perturbations
python -m shape_foundation.scripts.prepare_dataset --generate-synthetic --n-per-type 1000

python -m shape_foundation.scripts.train_finetune \
    --pretrained checkpoints_finetune/checkpoint_final.pt \
    --lr 1e-5 --epochs 20
```

## Evaluation

```bash
# Full evaluation suite
python -m shape_foundation.scripts.eval_backbone \
    --checkpoint checkpoints/checkpoint_final.pt \
    --config configs/medium.yaml \
    --output eval_results.json

# With robustness tests (noise + decimation)
python -m shape_foundation.scripts.eval_backbone \
    --checkpoint checkpoints/checkpoint_final.pt \
    --robustness

# TensorBoard
tensorboard --logdir runs/
```

### Metrics

| Metric | Description |
|--------|-------------|
| `retrieval_recall@K` | Recall@K for embedding retrieval |
| `mean_cosine_sim` | Cosine agreement across remeshings |
| `symmetry_accuracy` | Symmetry type classification accuracy |
| `symmetry_f1_macro` | Macro F1 across symmetry classes |
| `symmetry_plane_angular_error_deg` | Angular error on predicted symmetry planes |
| `primitive_accuracy` | Per-token primitive classification accuracy |
| `part_mIoU` | Part segmentation mean IoU |
| `reduction_accuracy` | Reduction recommendation accuracy |
| `reduction_ece` | Expected calibration error |
| `robustness_noise_*` | Cosine similarity under noise |
| `robustness_decimate_*` | Cosine similarity under decimation |

## Inference

```bash
python -m shape_foundation.scripts.infer_mesh \
    --checkpoint checkpoints/checkpoint_final.pt \
    --mesh path/to/mesh.stl \
    --output result.json \
    --verbose
```

Output schema:
```json
{
  "description": "This is a 3D geometry with mirror symmetry...",
  "symmetry": {
    "type": "mirror_half",
    "confidence": 0.95,
    "planes": [[1.0, 0.0, 0.0, 0.0]],
    "axes": []
  },
  "primitives": [
    {"type": "plane", "ratio": 0.45},
    {"type": "cylinder", "ratio": 0.30}
  ],
  "topology": {
    "repeated_sectors": 0,
    "constant_cross_section": false,
    "thin_regions": false
  },
  "simulation_hints": {
    "recommended_reduction": "mirror_half",
    "reasoning": ["Mirror symmetry plane detected", "Symmetry type: mirror_half"],
    "confidence": 0.92
  }
}
```

## Ablations

All ablation configs are in `configs/ablations/`. Run with the same train scripts:

```bash
# Input modes: vertices_only vs surface_sampled_points vs hybrid
python -m shape_foundation.scripts.train_pretrain --config configs/ablations/input_modes.yaml

# Latent grid: structured vs adaptive
python -m shape_foundation.scripts.train_pretrain --config configs/ablations/latent_grid.yaml

# Positional embedding: absolute vs RoPE
python -m shape_foundation.scripts.train_pretrain --config configs/ablations/positional.yaml

# Patch sizes: 4, 6, 8 (override from CLI)
python -m shape_foundation.scripts.train_pretrain --config configs/ablations/patch_sizes.yaml
# Edit patch_sizes.yaml to change processor.patch_size

# Transformer depth: 3, 6, 12
python -m shape_foundation.scripts.train_pretrain --config configs/ablations/transformer_depth.yaml

# Token dims: 128, 256, 512
python -m shape_foundation.scripts.train_pretrain --config configs/ablations/token_dims.yaml

# Loss combinations: masked-only, +contrastive, +inpainting
python -m shape_foundation.scripts.train_pretrain --config configs/ablations/loss_combos.yaml
```

## Configs

| Config | Latent Grid | Token Dim | Layers | Heads | Patch | Use Case |
|--------|-------------|-----------|--------|-------|-------|----------|
| `smoke_test.yaml` | 8³ | 64 | 2 | 2 | 4 | Verify setup |
| `small.yaml` | 24³ | 128 | 3 | 4 | 6 | Single GPU experiments |
| `medium.yaml` | 48³ | 256 | 6 | 8 | 6 | **Recommended default** |
| `large.yaml` | 48³ | 512 | 12 | 16 | 6 | Multi-GPU full training |

## Datasets

| Stage | Datasets | Purpose |
|-------|----------|---------|
| 1 | ABC, ShapeNet, Objaverse, Thingi10K | Broad geometry diversity |
| 2 | + PartNet, Fusion360, MFCAD++ | Engineering parts, topology |
| 3 | + SHREC 2022/2023, Scan2CAD | Symmetry/primitive supervision |
| 4 | + Custom/synthetic | CFD reduction labels |

## Project Structure

```
training/
├── configs/                    # YAML configs
│   ├── smoke_test.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   ├── large.yaml
│   └── ablations/
├── shape_foundation/
│   ├── configs/                # Dataclass config definitions
│   │   └── default.py
│   ├── data/                   # Data pipeline
│   │   ├── dataset.py          # MeshDataset, CollateFunction
│   │   ├── preprocessing.py    # Canonicalization, normals, curvature
│   │   ├── sampling.py         # Surface sampling strategies
│   │   └── synthetic_labels.py # Symmetry/primitive label generation
│   ├── preprocessing/
│   │   └── mesh_io.py          # Mesh loading (STL/OBJ/PLY/MSH/STEP)
│   ├── models/
│   │   ├── gaot_backbone.py    # GAOTBackbone (main model)
│   │   ├── tokenizer_magno.py  # MAGNO encoder (tokenizer)
│   │   ├── processor_transformer.py  # Transformer processor
│   │   └── heads.py            # All task heads
│   ├── training/
│   │   ├── losses.py           # All loss functions
│   │   ├── trainer.py          # DDP training loop
│   │   └── eval.py             # Evaluation suite
│   ├── tasks/                  # High-level task modules
│   │   ├── symmetry.py
│   │   ├── primitives.py
│   │   ├── captioning.py
│   │   └── topology_reduction.py
│   └── scripts/                # CLI entry points
│       ├── prepare_dataset.py
│       ├── train_pretrain.py
│       ├── train_finetune.py
│       ├── eval_backbone.py
│       └── infer_mesh.py
├── notebooks/
├── pyproject.toml
├── requirements.txt
└── README.md
```
