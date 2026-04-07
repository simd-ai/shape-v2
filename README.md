# Shape Foundation Model

3D geometry foundation model for simulation-relevant mesh analysis. Given a mesh of a physical domain, the model produces geometry embeddings, detects symmetry/primitives/topology, and recommends simulation reductions (mirror symmetry, axisymmetry, cyclic sectors, extrusions).

## Architecture

**GAOTBackbone** = MAGNO Encoder + Transformer Processor + Task Heads

- **MAGNO Encoder**: Multi-scale cross-attention from a structured 3D latent grid to physical surface points, with explicit geometric embeddings (statistical neighborhood features) and AGNO (cosine-similarity) attention.
- **Transformer Processor**: 3D patchification → grouped-query attention with RMSNorm → unpatchification. Supports absolute and RoPE positional embeddings, optional UViT skip connections.
- **Task Heads**: Geometry embedding, symmetry detection (classification + plane/axis regression), primitive recognition (per-token), part segmentation, template captioning, topology-reduction recommendation.

## Quick Start

```bash
# Install (tested on CUDA 12.8+ / V100/H100)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Also install aria2 for fast parallel downloads
sudo apt-get install -y aria2 p7zip-full
```

## Step 1: Download Datasets

```bash
# Download datasets — choose a config size:
#   small  = abc, thingi10k
#   medium = abc, objaverse, thingi10k
#   large  = abc, objaverse, objaverse_xl, thingi10k, partnet, fusion360, mfcad
./scripts/download_datasets.sh small

# Download more ABC chunks (default is 5, max 100, ~10k meshes each):
ABC_CHUNKS=20 ./scripts/download_datasets.sh small

# aria2c uses 16 parallel connections per file, chunks download simultaneously.
# Downloads are resumable — re-run the same command if interrupted.
# Raw data goes to data_raw/<dataset_name>/
```

## Step 2: Extract Archives

```bash
# Extract ABC 7z archives (if not auto-extracted by download script)
for f in data_raw/abc/abc_*_obj_v00.7z; do 7z x -odata_raw/abc "$f" -y; done
```

## Step 3: Preprocess into .pt Files

```bash
# Preprocess a specific dataset
python -m shape_foundation.scripts.prepare_dataset --source abc --root data_raw/abc --output data_cache/abc

# Preprocess thingi10k
python -m shape_foundation.scripts.prepare_dataset --source thingi10k --root data_raw/thingi10k --output data_cache/thingi10k

# Or generate synthetic data (no download needed)
python -m shape_foundation.scripts.prepare_dataset --generate-synthetic --n-per-type 500

# Limit samples for faster testing
python -m shape_foundation.scripts.prepare_dataset --source abc --root data_raw/abc --output data_cache/abc --max-samples 1000
```

## Step 4: Train (Pretrain)

```bash
# Smoke test (tiny model, verifies setup)
python -m shape_foundation.scripts.train_pretrain --config configs/smoke_test.yaml

# Single GPU — small config
python -m shape_foundation.scripts.train_pretrain --config configs/small.yaml

# Multi-GPU DDP — 8x V100
torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain --config configs/small.yaml

# Multi-GPU DDP — medium config (recommended)
torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain --config configs/medium.yaml

# Multi-GPU DDP — large config
torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain --config configs/large.yaml

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

## Cloud Storage (GCS)

Save and restore data across instances using Google Cloud Storage.

```bash
# --- Setup (one-time) ---
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud storage buckets create gs://shape-foundation-data --location=us-central1

# --- Upload to GCS ---
# Raw datasets (~1TB+)
gsutil -m rsync -r data_raw/ gs://shape-foundation-data/data_raw/
# Preprocessed .pt files
gsutil -m rsync -r data_cache/ gs://shape-foundation-data/data_cache/
# Checkpoints
gsutil -m rsync -r checkpoints/ gs://shape-foundation-data/checkpoints/

# --- Download from GCS (on a new instance) ---
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gsutil -m rsync -r gs://shape-foundation-data/data_raw/ data_raw/
gsutil -m rsync -r gs://shape-foundation-data/data_cache/ data_cache/
gsutil -m rsync -r gs://shape-foundation-data/checkpoints/ checkpoints/

# Then resume training
pip install -e .
torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain --config configs/small.yaml
```

Notes:
- `gsutil -m rsync` uses parallel transfers and only syncs new/changed files (resumable).
- To skip raw data and only sync preprocessed files: just sync `data_cache/` and `checkpoints/`.

## Datasets

| Dataset | Download Method | Status |
|---------|----------------|--------|
| ABC | `./scripts/download_datasets.sh` (auto) | Available |
| Thingi10K | `./scripts/download_datasets.sh` (auto) | Available |
| Objaverse | `./scripts/download_datasets.sh` (auto, via Python) | Available |
| Objaverse XL | `./scripts/download_datasets.sh` (auto, via Python) | Available |
| ShapeNet | Manual — requires shapenet.org / HuggingFace access | Pending approval |
| PartNet | Manual — requires Stanford access | Manual |
| Fusion360 | `./scripts/download_datasets.sh` (semi-auto) | Available |
| MFCAD++ | `./scripts/download_datasets.sh` (auto) | Available |

| Stage | Datasets | Purpose |
|-------|----------|---------|
| 1 | ABC, Objaverse, Thingi10K | Broad geometry diversity |
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
