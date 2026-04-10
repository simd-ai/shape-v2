# Shape Foundation Model

3D geometry foundation model for simulation-relevant mesh analysis. Given a mesh of a physical domain, the model produces geometry embeddings, detects symmetry/primitives/topology, and recommends simulation reductions (mirror symmetry, axisymmetry, cyclic sectors, extrusions).

## Architecture

**GAOTBackbone** = MAGNO Encoder + Transformer Processor + Task Heads

- **MAGNO Encoder**: Multi-scale cross-attention from a structured 3D latent grid to physical surface points, with explicit geometric embeddings (statistical neighborhood features) and AGNO (cosine-similarity) attention.
- **Transformer Processor**: 3D patchification → grouped-query attention with RMSNorm → unpatchification. Supports absolute and RoPE positional embeddings, optional UViT skip connections.
- **Task Heads**: Geometry embedding, symmetry detection (classification + plane/axis regression), primitive recognition (per-token), part segmentation, template captioning, topology-reduction recommendation.

## Quick Start

### Option A: From Preprocessed Data (Recommended)

If someone has already preprocessed the data and uploaded to GCS, skip straight to training:

```bash
# 1. Install
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install -e .

# 2. Pull preprocessed .pt files from GCS
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gsutil -m rsync -r gs://shape-foundation-data/data_cache/ data_cache/

# 3. Train
torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain --config configs/small.yaml
```

### Option B: From Scratch

Full pipeline: download raw meshes → preprocess → train.

#### Step 1: Install

```bash
# System dependencies (needed for STEP file processing via gmsh)
sudo apt-get install -y aria2 p7zip-full libxcursor1 libxinerama1 libxft2 libxmu6 libxi6 libglu1-mesa libgl1

# Python dependencies (install torch first — PyG extensions build against it)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install -e .
```

#### Step 2: Download Datasets

```bash
# Choose a tier:
#   small  = thingi10k, mfcad, fusion360           (~68 GB raw, ~15 GB preprocessed)
#   medium = + objaverse, partnet                   (~50-60 GB preprocessed)
#   large  = + abc, objaverse_xl                    (~500 GB+ preprocessed)
./scripts/download_datasets.sh small

# Check what's downloaded
./scripts/check_datasets.sh
```

Fusion360 segmentation subset needs a separate download (not automated):
```bash
aria2c -x 16 -s 16 -d data_raw/fusion360 -o s2.0.1.zip \
  "https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip"
unzip -q -o data_raw/fusion360/s2.0.1.zip -d data_raw/fusion360
```

#### Step 3: Preprocess into .pt Files

Each raw mesh is converted to a self-contained `.pt` file with surface points, normals, features, and curvature. Preprocessing is parallelized across CPU cores and resumable (safe to cancel and rerun).

```bash
# Auto-parallelized (uses 75% of CPU cores by default)
python -m shape_foundation.scripts.prepare_dataset --source thingi10k --root data_raw/thingi10k --output data_cache/thingi10k
python -m shape_foundation.scripts.prepare_dataset --source mfcad --root data_raw/mfcad --output data_cache/mfcad
python -m shape_foundation.scripts.prepare_dataset --source fusion360 --root data_raw/fusion360 --output data_cache/fusion360

# Control parallelism manually
python -m shape_foundation.scripts.prepare_dataset --source thingi10k --root data_raw/thingi10k --output data_cache/thingi10k --workers 64

# Generate synthetic data (no download needed)
python -m shape_foundation.scripts.prepare_dataset --generate-synthetic --n-per-type 500

# Limit samples for quick testing
python -m shape_foundation.scripts.prepare_dataset --source mfcad --root data_raw/mfcad --output data_cache/mfcad --max-samples 1000
```

Note: STEP files (MFCAD, Fusion360) are slower to preprocess than STL/OBJ (Thingi10K) because gmsh must tessellate CAD geometry.

#### Step 4: Upload to GCS (for team sharing)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create bucket (one-time)
gsutil mb -l us-central1 gs://shape-foundation-data

# Upload preprocessed data (resumable — safe to rerun if interrupted)
gsutil -m rsync -r data_cache/thingi10k/ gs://shape-foundation-data/data_cache/thingi10k/
gsutil -m rsync -r data_cache/mfcad/ gs://shape-foundation-data/data_cache/mfcad/
gsutil -m rsync -r data_cache/fusion360/ gs://shape-foundation-data/data_cache/fusion360/

# Upload checkpoints after training
gsutil -m rsync -r checkpoints/ gs://shape-foundation-data/checkpoints/
```

No need to upload `data_raw/` — the `.pt` cache is all you need for training.

#### Step 5: Train (Pretrain)

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

Only the preprocessed `.pt` files and checkpoints need to be shared — raw data is not needed.

```bash
# --- Upload (after preprocessing/training) ---
gsutil -m rsync -r data_cache/thingi10k/ gs://shape-foundation-data/data_cache/thingi10k/
gsutil -m rsync -r data_cache/mfcad/ gs://shape-foundation-data/data_cache/mfcad/
gsutil -m rsync -r data_cache/fusion360/ gs://shape-foundation-data/data_cache/fusion360/
gsutil -m rsync -r checkpoints/ gs://shape-foundation-data/checkpoints/

# --- Download (on a new instance) ---
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gsutil -m rsync -r gs://shape-foundation-data/data_cache/ data_cache/
gsutil -m rsync -r gs://shape-foundation-data/checkpoints/ checkpoints/

# Then train directly — no preprocessing needed
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install -e .
torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain --config configs/small.yaml
```

Notes:
- `gsutil -m rsync` uses parallel transfers and only syncs new/changed files (resumable).
- If upload gets interrupted, just rerun — it picks up where it left off.
- Upload datasets one at a time if the full rsync times out.

## Datasets

| Dataset | Files | Raw Size | Format | Source |
|---------|-------|----------|--------|--------|
| Thingi10K | 9,999 | 46 GB | STL/OBJ | [HuggingFace](https://huggingface.co/datasets/Thingi10K/Thingi10K) |
| MFCAD++ | 30,976 | 2.5 GB | STEP | [GitHub](https://github.com/hducg/MFCAD) |
| Fusion360 | 71,362 | 20 GB | STEP/BREP | [S3](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip) |
| Objaverse | 100K+ | ~50 GB | GLB/OBJ | `objaverse` Python package |
| PartNet | varies | varies | OBJ | Manual — requires Stanford access |
| ABC | ~1M | ~1 TB | OBJ | [ABC Dataset](https://deep-geometry.github.io/abc-dataset/) |
| Objaverse XL | 500K+ | ~200 GB | GLB/OBJ | `objaverse` Python package |

| Tier | Datasets | ~Preprocessed Size | Use Case |
|------|----------|-------------------|----------|
| **small** | Thingi10K, MFCAD++, Fusion360 | ~10-15 GB | Initial experiments |
| **medium** | + Objaverse, PartNet | ~50-60 GB | Broader diversity |
| **large** | + ABC, Objaverse XL | ~500 GB+ | Full-scale training |

Check dataset status: `./scripts/check_datasets.sh`

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
