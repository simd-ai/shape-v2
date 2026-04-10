#!/usr/bin/env bash
# Check which datasets are downloaded and their status.
# Usage: ./scripts/check_datasets.sh

set -euo pipefail

DATA_RAW="${DATA_RAW:-$(pwd)/data_raw}"
DATA_CACHE="${DATA_CACHE:-$(pwd)/data_cache}"

# ---------------------------------------------------------------------------
# Dataset definitions: name, extensions to search, tier (small/medium/large)
# ---------------------------------------------------------------------------
DATASETS=(
    "thingi10k|*.stl,*.obj,*.ply|small"
    "mfcad|*.step,*.stp,*.stl,*.obj|small"
    "fusion360|*.step,*.stp,*.stl,*.obj,*.brep|small"
    "objaverse|*.glb,*.obj|medium"
    "partnet|*.obj|medium"
    "abc|*.obj,*.stl,*.ply|large"
    "objaverse_xl|*.glb,*.obj|large"
)

# ---------------------------------------------------------------------------
# Count mesh files for a dataset
# ---------------------------------------------------------------------------
count_files() {
    local dir="$1"
    local exts="$2"
    [ -d "$dir" ] || { echo 0; return; }
    local count=0
    IFS=',' read -ra ext_arr <<< "$exts"
    for ext in "${ext_arr[@]}"; do
        local n
        n=$(find "$dir" -type f -name "$ext" 2>/dev/null | wc -l)
        count=$((count + n))
    done
    echo "$count"
}

# ---------------------------------------------------------------------------
# Human-readable size
# ---------------------------------------------------------------------------
dir_size() {
    local dir="$1"
    [ -d "$dir" ] || { echo "—"; return; }
    du -sh "$dir" 2>/dev/null | cut -f1
}

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
printf "\n"
printf "  %-15s  %-8s  %10s  %10s  %10s  %10s\n" "Dataset" "Tier" "Raw Files" "Raw Size" "Cached .pt" "Cache Size"
printf "  %-15s  %-8s  %10s  %10s  %10s  %10s\n" "───────────────" "────────" "──────────" "──────────" "──────────" "──────────"

small_ok=true
medium_ok=true
large_ok=true

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name exts tier <<< "$entry"

    raw_dir="$DATA_RAW/$name"
    cache_dir="$DATA_CACHE/$name"

    raw_count=$(count_files "$raw_dir" "$exts")
    raw_size=$(dir_size "$raw_dir")

    # Count preprocessed .pt files
    if [ -d "$cache_dir" ]; then
        pt_count=$(find "$cache_dir" -type f -name "*.pt" 2>/dev/null | wc -l)
        pt_size=$(dir_size "$cache_dir")
    else
        pt_count=0
        pt_size="—"
    fi

    # Status indicator
    if [ "$raw_count" -gt 0 ]; then
        status="✓"
    else
        status="✗"
        case "$tier" in
            small)  small_ok=false; medium_ok=false; large_ok=false ;;
            medium) medium_ok=false; large_ok=false ;;
            large)  large_ok=false ;;
        esac
    fi

    printf "  %-15s  %-8s  %10s  %10s  %10s  %10s  %s\n" \
        "$name" "$tier" "$raw_count" "$raw_size" "$pt_count" "$pt_size" "$status"
done

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
syn_dir="$DATA_CACHE/synthetic"
if [ -d "$syn_dir" ]; then
    syn_count=$(find "$syn_dir" -type f -name "*.pt" 2>/dev/null | wc -l)
    syn_size=$(dir_size "$syn_dir")
else
    syn_count=0
    syn_size="—"
fi
printf "  %-15s  %-8s  %10s  %10s  %10s  %10s\n" \
    "synthetic" "any" "—" "—" "$syn_count" "$syn_size"

# ---------------------------------------------------------------------------
# Tier summary
# ---------------------------------------------------------------------------
printf "\n"
printf "  Tier readiness:\n"
if $small_ok; then
    printf "    small  : ✓ ready (thingi10k, mfcad, fusion360)\n"
else
    printf "    small  : ✗ missing datasets — run: ./scripts/download_datasets.sh small\n"
fi
if $medium_ok; then
    printf "    medium : ✓ ready (+ objaverse, partnet)\n"
else
    printf "    medium : ✗ missing datasets — run: ./scripts/download_datasets.sh medium\n"
fi
if $large_ok; then
    printf "    large  : ✓ ready (+ abc, objaverse_xl)\n"
else
    printf "    large  : ✗ missing datasets — run: ./scripts/download_datasets.sh large\n"
fi
printf "\n"
