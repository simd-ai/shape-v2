# Shape Foundation Model

A 3D geometry foundation model for industrial CAD analysis. Takes a mesh of a physical domain and produces dense geometric embeddings with a self-supervised reconstruction prior that enables per-token attribution for explainable predictions.

## Current Release

**Small v3** — self-supervised backbone, trained and validated.

| | |
|---|---|
| Parameters | 10,913,297 |
| Training data | 61,052 CAD meshes |
| Datasets | Fusion360 (58.4%), MFCAD (25.4%), Thingi10K (16.2%) |
| Train / val split | 58,069 / 2,983 (deterministic hash-based) |
| Compute | 8 × H100 80GB, 50 epochs, ~2h 30min |
| Val reconstruction loss | 0.030 (matches train within 1%) |
| Precision | bf16 + DDP + torch.compile |
| Model hub | [`simd-ai/shape-foundation-small-v3`](https://huggingface.co/simd-ai/shape-foundation-small-v3) |

The self-supervised backbone generalizes to unseen meshes. Supervised task heads (symmetry, primitive, part, reduction) are present in the architecture but their weights are currently `0.0` — the stock synthetic labels do not generalize and would overfit the model. See [Known Limitations](#known-limitations) below.

## Architecture

**GAOTBackbone** = MAGNO Encoder → Transformer Processor → Task Heads

- **MAGNO Encoder** — cross-attends from a structured 3D latent grid (24³ = 13,824 tokens) to 8,192 sampled surface points using cosine-similarity attention with learned temperature. Each token encodes local geometric statistics (mean / std / min / max of relative positions, normals, curvature) in a 28-dim signature.

- **Transformer Processor** — 3D patchification (patch_size=6), grouped-query attention with RMSNorm, optional RoPE positional embeddings, unpatchification back to dense token embeddings.

- **Task Heads** — geometry embedding (global pooled), reconstruction projection (for pretraining), plus symmetry, primitive, part, caption, and topology-reduction heads (currently disabled pending label rework).

### Pretraining objectives

Self-supervised only — no labels required.

1. **Masked Token Reconstruction** (weight 1.0) — 50% of latent tokens are masked; the model predicts their geometry statistics from surrounding context. Analogous to masked-language-modeling in LLMs but in 3D latent space. SmoothL1 loss (β=1.0) in normalized target space.

2. **Multi-resolution Contrastive Consistency** (weight 0.2) — the same mesh under two augmentations (position jitter σ=0.02, 30% point dropout) must embed similarly. InfoNCE with temperature 0.07.

### Key engineering decisions

- **SmoothL1 regression** instead of MSE — curvature-dim outliers had std up to 711; MSE was dominated by a handful of extreme values. SmoothL1's linear regime above β=1 makes the loss outlier-robust.
- **Per-dimension target normalization** — `raw_geo_stats` has 28 dimensions with std spanning `[0.036, 711]`. Per-dim mean and std are calibrated once on the training split at startup, stored as registered buffers, and used to normalize reconstruction targets to unit variance.
- **Deterministic hash-based train/val split** — each file's split assignment is `md5(path) mod 10000 < val_fraction × 10000`. Identical across runs, ranks, and machines.
- **Atomic checkpoint saves** — writes to `<path>.tmp`, fsyncs, then `os.replace` to the final name. A failed save never leaves a corrupt checkpoint.

## Installation

```bash
# System dependencies
sudo apt-get install -y aria2 p7zip-full libxcursor1 libxinerama1 libxft2 libxmu6 libxi6 libglu1-mesa libgl1

# Python dependencies — install torch first, PyG builds against it
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install -e .
```

Requires Python ≥3.10, CUDA 12.4+, PyTorch ≥2.6.

## Quick Start

The fastest path — download the preprocessed dataset and pretrained checkpoint, then train or evaluate.

```bash
# 1. Download preprocessed dataset from GCS (~33 GB)
gcloud auth login
gcloud storage cp -r gs://<YOUR_BUCKET>/shape-v2/data_cache/ data_cache/

# 2. Download the small v3 checkpoint from HuggingFace
hf download simd-ai/shape-foundation-small-v3 --local-dir checkpoints/

# 3. Train (or resume from the downloaded checkpoint)
torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain --config configs/small.yaml
```

## Training from Scratch

If you want to preprocess the raw meshes yourself:

```bash
# 1. Download raw meshes
./scripts/download_datasets.sh small

# Fusion360 segmentation subset (manual step)
aria2c -x 16 -s 16 -d data_raw/fusion360 -o s2.0.1.zip \
  "https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip"
unzip -q -o data_raw/fusion360/s2.0.1.zip -d data_raw/fusion360

# 2. Preprocess each source into .pt files (parallelized, resumable)
python -m shape_foundation.scripts.prepare_dataset --source thingi10k --root data_raw/thingi10k --output data_cache/thingi10k
python -m shape_foundation.scripts.prepare_dataset --source mfcad    --root data_raw/mfcad    --output data_cache/mfcad
python -m shape_foundation.scripts.prepare_dataset --source fusion360 --root data_raw/fusion360 --output data_cache/fusion360

# 3. Train
torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain --config configs/small.yaml
```

Preprocessing takes roughly 1 hour on a 32-core machine for the small tier. STEP files (MFCAD, Fusion360) are slower than STL/OBJ because gmsh has to tessellate the CAD geometry.

## Evaluation

```bash
python -m shape_foundation.scripts.eval_backbone \
    --checkpoint checkpoints/checkpoint_final.pt
```

Loads the checkpoint, runs the evaluator on the validation split, and writes `checkpoint_final_eval_val.json` next to the checkpoint. Use `--split test` or `--robustness` to change behavior.

## Inference

```bash
python -m shape_foundation.scripts.infer_mesh \
    --checkpoint checkpoints/checkpoint_final.pt \
    --mesh path/to/mesh.stl
```

For an interactive 3D demo with masked-token reconstruction heatmaps and shape retrieval, see the separate [`inference/`](https://github.com/simd-ai/shape-backend-v2) repo (FastAPI backend + Next.js frontend).

## Configuration

All configuration is dataclass-based in `shape_foundation/configs/default.py`. YAML files in `configs/` override the defaults via deep merge.

| Config | Parameters | Latent Grid | Token Dim | Layers × Heads | Status |
|---|---|---|---|---|---|
| `small.yaml` | 10.9M | 24³ | 128 | 3 × 4 | Trained ✅ |
| `medium.yaml` | ~150M | 48³ | 256 | 6 × 8 | Next |
| `large.yaml` | ~600M | 48³ | 512 | 12 × 16 | Planned |

### Tuning common parameters

Edit the YAML or override on the command line:

```bash
python -m shape_foundation.scripts.train_pretrain --config configs/small.yaml --epochs 100 --batch-size 32 --lr 1e-4
```

## Scaling Strategy

All scaling is config-only — no architectural changes needed.

| Axis | Small (v1) | Medium (v2) | Large (v3) |
|---|---|---|---|
| Parameters | 10M | 300M | 1B |
| Latent grid | 24³ | 48³ | 64³ |
| Hidden dim | 128 | 256 | 512 |
| Transformer layers | 3 | 6 | 12 |
| Training meshes | 61k | 500k | 2M+ |
| Data sources | Thingi10K + MFCAD + Fusion360 | + Objaverse + PartNet | + ABC + Objaverse-XL |
| Mask ratio | 0.5 | 0.5 | 0.75 |
| Compute budget | ~20 GPU-h | ~2,000 GPU-h | ~40,000 GPU-h |

What stays constant: MAGNO cross-attention, grouped-query attention, self-supervised objective, per-dim target normalization, SmoothL1 regression.

## Datasets

| Dataset | Meshes | Format | Source |
|---|---|---|---|
| Thingi10K | 9,883 | STL / OBJ | [HuggingFace](https://huggingface.co/datasets/Thingi10K/Thingi10K) |
| MFCAD++ | 15,488 | STEP | [GitHub](https://github.com/hducg/MFCAD) |
| Fusion360 | 35,681 | STEP / BREP | [AWS S3](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip) |

Check dataset status at any time with `./scripts/check_datasets.sh`.

## Known Limitations

**Supervised task heads are disabled.** The symmetry, primitive, part, and reduction heads are architecturally present but trained with weight `0.0`. Under previous runs with stock synthetic labels enabled, training cross-entropy collapsed to `~1e-4` while validation CE stayed at chance level (`~2.5`) — classic memorization of per-file label noise. Re-enabling these heads requires rewriting `shape_foundation/data/synthetic_labels.py` so the labels are recoverable from the sampled point cloud the model actually sees, not from full-mesh properties.

**Contrastive loss saturates early.** With per-rank batch size 16, InfoNCE has only 15 negatives per anchor, which makes the objective too easy after a few epochs. Cross-rank negative sampling (via `dist.all_gather` on pooled embeddings) is the natural fix when scaling up the batch size becomes too expensive.

## License

See `LICENSE`. Training data is used under the respective licenses of each upstream dataset.
