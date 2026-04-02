# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

3D geometry foundation model ("Shape Foundation") for simulation-relevant mesh analysis. Takes a mesh, produces geometry embeddings, detects symmetry/primitives/topology, and recommends simulation reductions.

## Commands

```bash
# Install (requires CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Dev dependencies
pip install -e ".[dev]"

# Generate synthetic training data
python -m shape_foundation.scripts.prepare_dataset --generate-synthetic --n-per-type 500

# Smoke test (tiny model, verifies setup)
python -m shape_foundation.scripts.train_pretrain --config configs/smoke_test.yaml --dry-run
python -m shape_foundation.scripts.train_pretrain --config configs/smoke_test.yaml

# Pretrain (single GPU)
python -m shape_foundation.scripts.train_pretrain --config configs/medium.yaml

# Pretrain (multi-GPU DDP)
torchrun --nproc_per_node=4 -m shape_foundation.scripts.train_pretrain --config configs/large.yaml

# Fine-tune
python -m shape_foundation.scripts.train_finetune --pretrained checkpoints/checkpoint_final.pt --config configs/medium.yaml

# Evaluate
python -m shape_foundation.scripts.eval_backbone --checkpoint checkpoints/checkpoint_final.pt --config configs/medium.yaml

# Inference on a single mesh
python -m shape_foundation.scripts.infer_mesh --checkpoint checkpoints/checkpoint_final.pt --mesh path/to/mesh.stl

# Lint
ruff check shape_foundation/

# Tests
pytest
```

## Architecture

**GAOTBackbone** = MAGNO Encoder → Transformer Processor → Task Heads

- **MAGNO Encoder** (`models/tokenizer_magno.py`): Multi-scale cross-attention from a structured 3D latent grid to physical surface points. Uses geometric embeddings (statistical neighborhood features) and AGNO (cosine-similarity) attention. Configurable scales, grid strategies (structured/adaptive/multiresolution), and geo-embed modes.

- **Transformer Processor** (`models/processor_transformer.py`): 3D patchification → grouped-query attention (GQA) with RMSNorm → unpatchification. Supports absolute and RoPE positional embeddings, optional UViT skip connections.

- **Task Heads** (`models/heads.py`): Six heads fed from token-level or pooled embeddings:
  - `GeometryEmbeddingHead` — global geometry embedding
  - `SymmetryHead` — classification + plane/axis regression (pooled)
  - `PrimitiveTopologyHead` — per-token primitive type + topology flags (token-level)
  - `PartRegionHead` — part segmentation (token-level)
  - `CaptionHead` — template-based text description (pooled)
  - `TopologyReductionHead` — simulation reduction recommendation (pooled)

- **GAOTBackbone** (`models/gaot_backbone.py`): Orchestrates the pipeline. Three forward modes: `forward_tokens` (raw embeddings), `forward_features` (+ head outputs), `forward_tasks` (full inference).

## Configuration System

All config is dataclass-based in `shape_foundation/configs/default.py`. Top-level `ShapeConfig` nests: `InputConfig`, `TokenizerConfig`, `ProcessorConfig`, `HeadsConfig`, `TrainConfig`, `DataConfig`. YAML configs in `configs/` override defaults via deep merge. CLI args can further override specific fields.

Config hierarchy: `configs/smoke_test.yaml` < `small.yaml` < `medium.yaml` (recommended default) < `large.yaml`.

## Training Pipeline

Three-stage training: (1) self-supervised pretraining with masked token modeling + contrastive + inpainting losses, (2) supervised fine-tuning with task head losses, (3) symmetry/reduction-specific fine-tuning. The `Trainer` class in `training/trainer.py` handles DDP, mixed precision (bf16/fp16/fp32), gradient accumulation, `torch.compile`, checkpointing with resume, and TensorBoard logging.

## Data Pipeline

`data/dataset.py` provides `MeshDataset` and `CollateFunction`. Preprocessing in `data/preprocessing.py` (canonicalization, normals, curvature). Surface sampling strategies in `data/sampling.py`. Synthetic label generation for symmetry/primitives in `data/synthetic_labels.py`. Mesh I/O supports STL/OBJ/PLY/MSH/STEP via `preprocessing/mesh_io.py`.

## Key Dependencies

PyTorch ≥2.2, torch-geometric ≥2.5, trimesh ≥4.0, einops, meshio, scipy. Python ≥3.10.
