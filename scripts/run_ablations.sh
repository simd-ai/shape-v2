#!/usr/bin/env bash
# Run the 4 loss/normalization ablation variants sequentially on 8 × H100.
#
# Each variant:
#   1. Trains for 20 epochs with its own config
#   2. Saves checkpoint + TB logs to a variant-specific directory
#   3. Runs evaluation on the final checkpoint
#   4. Appends results to a shared CSV for cross-run plotting
#
# Total wall-clock: ~4 hours on 8 × H100 80GB (4 runs × ~1h at 20 epochs).
#
# Usage:
#     ./scripts/run_ablations.sh
#
# To re-run only one variant, invoke the corresponding torchrun + eval pair
# from the table below manually.

set -euo pipefail

RESULTS_DIR=/home/nvidia/shape-v2/results/ablations
HISTORY_CSV=$RESULTS_DIR/ablation_history.csv
mkdir -p "$RESULTS_DIR"

run_one() {
    local name=$1
    local config=$2
    local ckpt_dir=$3

    echo ""
    echo "============================================================"
    echo " Ablation: $name"
    echo " Config:   $config"
    echo " Ckpt dir: $ckpt_dir"
    echo "============================================================"
    echo ""

    torchrun --nproc_per_node=8 -m shape_foundation.scripts.train_pretrain \
        --config "$config"

    python -m shape_foundation.scripts.eval_backbone \
        --checkpoint "$ckpt_dir/checkpoint_final.pt" \
        --output "$RESULTS_DIR/${name}_eval_val.json" \
        --history "$HISTORY_CSV"
}

cd /home/nvidia/shape-v2

run_one "run1_mse_raw" \
        "configs/ablations/small_mse_raw.yaml" \
        "/data/shape-v2/ablations/run1_mse_raw/checkpoints"

run_one "run2_mse_norm" \
        "configs/ablations/small_mse_norm.yaml" \
        "/data/shape-v2/ablations/run2_mse_norm/checkpoints"

run_one "run3_smoothl1_raw" \
        "configs/ablations/small_smoothl1_raw.yaml" \
        "/data/shape-v2/ablations/run3_smoothl1_raw/checkpoints"

run_one "run4_smoothl1_norm" \
        "configs/ablations/small_smoothl1_norm.yaml" \
        "/data/shape-v2/ablations/run4_smoothl1_norm/checkpoints"

echo ""
echo "============================================================"
echo " All 4 ablations complete."
echo " History CSV: $HISTORY_CSV"
echo ""
echo " Generating comparison plots..."
echo "============================================================"

python -m shape_foundation.scripts.plot_eval_metrics \
    --history "$HISTORY_CSV" \
    --output "$RESULTS_DIR/plots/"

echo ""
echo "Done. Artifacts:"
echo "  $RESULTS_DIR/"
echo "  $RESULTS_DIR/plots/"
echo ""
