#!/usr/bin/env bash
# Download datasets for Shape Foundation model training.
# Usage: ./scripts/download_datasets.sh [small|medium|large]
#
# Datasets are downloaded to data_raw/<name>/ then preprocessed into data_cache/<name>/
# After downloading, run:
#   python -m shape_foundation.scripts.prepare_dataset --config configs/<size>.yaml
#
# Environment variables:
#   CONNECTIONS_PER_FILE=16  — number of parallel connections per file (default: 16)
#   OBJAVERSE_MAX=100000    — max objaverse objects to download

set -euo pipefail

SIZE="${1:-small}"
CONNECTIONS="${CONNECTIONS_PER_FILE:-16}"
DATA_RAW="$(pwd)/data_raw"
DATA_CACHE="$(pwd)/data_cache"
mkdir -p "$DATA_RAW" "$DATA_CACHE"

# ---------------------------------------------------------------------------
# Downloader: uses aria2c (multi-connection) with fallback to wget
# ---------------------------------------------------------------------------
download_file() {
    local url="$1"
    local output="$2"
    if command -v aria2c &>/dev/null; then
        aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -c -d "$(dirname "$output")" -o "$(basename "$output")" "$url"
    else
        wget -c --show-progress -O "$output" "$url" || \
        curl -L -C - -o "$output" --progress-bar "$url"
    fi
}

# ---------------------------------------------------------------------------
# Dataset download functions
# ---------------------------------------------------------------------------

download_abc() {
    local dest="$DATA_RAW/abc"
    if [ -d "$dest" ] && [ "$(find "$dest" -name '*.obj' -o -name '*.stl' -o -name '*.ply' 2>/dev/null | head -1)" ]; then
        echo "[abc] Already downloaded, skipping."
        return
    fi
    echo "[abc] Downloading ABC dataset (A Big CAD Model Dataset)..."
    mkdir -p "$dest"
    # ABC dataset from https://deep-geometry.github.io/abc-dataset/
    # Download the URL index, then fetch OBJ chunks in parallel
    local URL_INDEX="$dest/obj_v00.txt"
    if [ ! -f "$URL_INDEX" ]; then
        echo "  [abc] Fetching chunk URL index..."
        download_file "https://deep-geometry.github.io/abc-dataset/data/obj_v00.txt" "$URL_INDEX"
    fi
    local NUM_CHUNKS="${ABC_CHUNKS:-5}"  # default 5 chunks, override with ABC_CHUNKS=100
    local pids=()
    local chunks_to_download=()
    local logdir
    logdir=$(mktemp -d)

    local line_num=0
    while IFS=' ' read -r url filename; do
        [ -z "$url" ] && continue
        line_num=$((line_num + 1))
        [ "$line_num" -gt "$NUM_CHUNKS" ] && break
        local chunk
        chunk=$(printf "%04d" "$((line_num - 1))")
        local archive="$dest/${filename}"
        if [ -f "$archive" ]; then
            echo "  [abc] Chunk $chunk already downloaded."
            continue
        fi
        chunks_to_download+=("$chunk")
        (
            download_file "$url" "$archive" > "$logdir/$chunk.log" 2>&1 || \
            echo "FAILED" > "$logdir/$chunk.log"
        ) &
        pids+=($!)
    done < "$URL_INDEX"

    # Show live progress table for all parallel downloads
    if [ ${#pids[@]} -gt 0 ]; then
        local n_chunks=${#chunks_to_download[@]}
        echo ""
        echo "  ┌────────────┬────────────┬────────────┐"
        echo "  │ Chunk      │ Downloaded │ Status     │"
        echo "  ├────────────┼────────────┼────────────┤"
        for chunk in "${chunks_to_download[@]}"; do
            echo "  │ $chunk       │ ---        │ starting   │"
        done
        echo "  └────────────┴────────────┴────────────┘"

        while true; do
            local all_done=true
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    all_done=false
                    break
                fi
            done

            printf "\033[%dA" "$((n_chunks + 1))"

            for i in "${!chunks_to_download[@]}"; do
                local chunk="${chunks_to_download[$i]}"
                local archive="$dest/abc_${chunk}_obj_v00.7z"
                local size="---"
                local status="waiting"
                if [ -f "$archive" ]; then
                    size=$(du -h "$archive" 2>/dev/null | cut -f1)
                    if kill -0 "${pids[$i]}" 2>/dev/null; then
                        status="downloading"
                    else
                        if wait "${pids[$i]}" 2>/dev/null; then
                            status="done ✓"
                        else
                            status="FAILED ✗"
                        fi
                    fi
                fi
                printf "  │ %-10s │ %-10s │ %-10s │\n" "$chunk" "$size" "$status"
            done
            echo "  └────────────┴────────────┴────────────┘"

            if $all_done; then
                break
            fi
            sleep 2
        done
    fi

    # Collect exit statuses
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done
    rm -rf "$logdir"
    if [ "$failed" -gt 0 ]; then
        echo "  [abc] WARNING: $failed chunk(s) failed to download."
        echo "  [abc] If URLs failed, the ABC dataset may need manual download from:"
        echo "  [abc] https://deep-geometry.github.io/abc-dataset/"
        echo "  [abc] Download the OBJ chunks and place them in: $dest"
    else
        echo "  [abc] All chunks downloaded."
    fi
    # Extract
    if command -v 7z &>/dev/null; then
        for f in "$dest"/abc_*_obj_v00.7z; do
            [ -f "$f" ] || continue
            echo "  [abc] Extracting $(basename "$f")..."
            7z x -o"$dest" "$f" -y > /dev/null
        done
    else
        echo "  [abc] Install p7zip to extract: sudo apt-get install p7zip-full"
        echo "  [abc] Then run: for f in $dest/abc_*_obj.7z; do 7z x -o\"$dest\" \"\$f\"; done"
    fi
}

download_shapenet() {
    local dest="$DATA_RAW/shapenet"
    if [ -d "$dest" ] && [ "$(find "$dest" -name '*.obj' 2>/dev/null | head -1)" ]; then
        echo "[shapenet] Already downloaded, skipping."
        return
    fi
    echo "[shapenet] ShapeNet requires manual registration at https://shapenet.org/"
    echo "  1. Go to https://shapenet.org/ and create an account"
    echo "  2. Download ShapeNetCore.v2.zip"
    echo "  3. Extract to: $dest"
    echo ""
    echo "  If you already have it elsewhere, symlink it:"
    echo "    ln -s /path/to/ShapeNetCore.v2 $dest"
    echo ""
    mkdir -p "$dest"
    # Check if user has it somewhere common
    for candidate in /data/ShapeNet /data/shapenet /datasets/ShapeNet ~/datasets/ShapeNet; do
        if [ -d "$candidate" ]; then
            echo "  [shapenet] Found existing dataset at $candidate — creating symlink."
            ln -sfn "$candidate" "$dest"
            return
        fi
    done
    echo "  [shapenet] Skipping for now. Re-run after downloading."
}

download_objaverse() {
    local dest="$DATA_RAW/objaverse"
    if [ -d "$dest" ] && [ "$(find "$dest" -name '*.glb' -o -name '*.obj' 2>/dev/null | head -1)" ]; then
        echo "[objaverse] Already downloaded, skipping."
        return
    fi
    echo "[objaverse] Downloading via objaverse Python package..."
    mkdir -p "$dest"
    pip install -q objaverse 2>/dev/null || true
    python3 -c "
import objaverse
import shutil, os

dest = '$dest'
print('  [objaverse] Loading UIDs...')
uids = objaverse.load_uids()
print(f'  [objaverse] {len(uids)} objects available.')

# Download objects (objaverse caches them automatically)
max_download = int(os.environ.get('OBJAVERSE_MAX', '100000'))
subset = uids[:max_download]
print(f'  [objaverse] Downloading {len(subset)} objects...')
objects = objaverse.load_objects(subset, download_processes=32)

# Symlink/copy to dest
for uid, path in objects.items():
    dst = os.path.join(dest, os.path.basename(path))
    if not os.path.exists(dst):
        os.symlink(path, dst)

print(f'  [objaverse] {len(objects)} objects ready in {dest}')
"
}

download_objaverse_xl() {
    local dest="$DATA_RAW/objaverse_xl"
    if [ -d "$dest" ] && [ "$(find "$dest" -type f 2>/dev/null | head -1)" ]; then
        echo "[objaverse_xl] Already downloaded, skipping."
        return
    fi
    echo "[objaverse_xl] Downloading via objaverse Python package (XL subset)..."
    mkdir -p "$dest"
    pip install -q objaverse 2>/dev/null || true
    python3 -c "
import objaverse.xl as oxl
import os

dest = '$dest'
print('  [objaverse_xl] Loading annotations...')
annotations = oxl.get_annotations()

# Download from GitHub source (largest free source)
max_download = int(os.environ.get('OBJAVERSE_XL_MAX', '500000'))
github_annots = annotations[annotations['source'] == 'github'].head(max_download)
print(f'  [objaverse_xl] Downloading {len(github_annots)} objects from GitHub source...')

oxl.download_objects(
    github_annots,
    download_dir=dest,
    processes=8,
)
print(f'  [objaverse_xl] Done.')
"
}

download_thingi10k() {
    local dest="$DATA_RAW/thingi10k"
    if [ -d "$dest" ] && [ "$(find "$dest" -name '*.stl' -o -name '*.obj' 2>/dev/null | head -1)" ]; then
        echo "[thingi10k] Already downloaded, skipping."
        return
    fi
    echo "[thingi10k] Downloading Thingi10K..."
    mkdir -p "$dest"

    # Try HuggingFace mirror first (tar.gz)
    local archive="$dest/Thingi10K.tar.gz"
    local hf_url="https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/Thingi10K.tar.gz"
    echo "  [thingi10k] Fetching from HuggingFace mirror..."
    if download_file "$hf_url" "$archive" 2>/dev/null; then
        echo "  [thingi10k] Extracting..."
        tar -xzf "$archive" -C "$dest"
        echo "  [thingi10k] Done."
        return
    fi

    # Fallback: use thingi10k Python package with raw variant
    echo "  [thingi10k] HuggingFace download failed, trying Python package..."
    pip install -q thingi10k 2>/dev/null || true
    python3 -c "
import thingi10k, shutil, os
dest = '$dest'
print('  [thingi10k] Downloading via thingi10k package (raw variant)...')
thingi10k.init(variant='raw')
db = thingi10k.dataset()
count = 0
for entry in db:
    src = entry.path
    if src and os.path.exists(src):
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
        count += 1
print(f'  [thingi10k] {count} meshes copied to {dest}')
"
}

download_partnet() {
    local dest="$DATA_RAW/partnet"
    if [ -d "$dest" ] && [ "$(find "$dest" -name '*.obj' 2>/dev/null | head -1)" ]; then
        echo "[partnet] Already downloaded, skipping."
        return
    fi
    echo "[partnet] PartNet requires manual registration at https://partnet.cs.stanford.edu/"
    echo "  1. Go to https://partnet.cs.stanford.edu/ and request access"
    echo "  2. Download the dataset"
    echo "  3. Extract to: $dest"
    echo ""
    echo "  If you already have it elsewhere, symlink it:"
    echo "    ln -s /path/to/PartNet $dest"
    echo ""
    mkdir -p "$dest"
    for candidate in /data/PartNet /data/partnet /datasets/PartNet ~/datasets/PartNet; do
        if [ -d "$candidate" ]; then
            echo "  [partnet] Found existing dataset at $candidate — creating symlink."
            ln -sfn "$candidate" "$dest"
            return
        fi
    done
    echo "  [partnet] Skipping for now. Re-run after downloading."
}

download_fusion360() {
    local dest="$DATA_RAW/fusion360"
    if [ -d "$dest" ] && [ "$(find "$dest" -name '*.st*' -o -name '*.obj' 2>/dev/null | head -1)" ]; then
        echo "[fusion360] Already downloaded, skipping."
        return
    fi
    echo "[fusion360] Downloading Fusion 360 Gallery Dataset..."
    mkdir -p "$dest"
    # Clone the repo for metadata, then download the actual data
    if [ ! -d "$dest/repo" ]; then
        git clone --depth 1 https://github.com/AutodeskAILab/Fusion360GalleryDataset.git "$dest/repo" 2>/dev/null || true
    fi
    # The actual mesh data is hosted on S3/links in the repo
    if [ -f "$dest/repo/tools/download.py" ]; then
        echo "  [fusion360] Running official download script..."
        cd "$dest/repo" && python3 tools/download.py --output "$dest/data" 2>/dev/null || {
            echo "  [fusion360] Official download script failed. Check $dest/repo/README.md for manual instructions."
        }
        cd - > /dev/null
    else
        echo "  [fusion360] Download meshes manually — see $dest/repo/README.md"
    fi
}

download_mfcad() {
    local dest="$DATA_RAW/mfcad"
    if [ -d "$dest" ] && [ "$(find "$dest" -name '*.st*' -o -name '*.obj' 2>/dev/null | head -1)" ]; then
        echo "[mfcad] Already downloaded, skipping."
        return
    fi
    echo "[mfcad] Downloading MFCAD++ dataset..."
    mkdir -p "$dest"
    # MFCAD++ is typically hosted on Kaggle or direct links
    echo "  [mfcad] Attempting GitHub download..."
    git clone --depth 1 https://github.com/hducg/MFCAD.git "$dest/repo" 2>/dev/null || true
    if [ -d "$dest/repo" ]; then
        # Move STEP files to dest
        find "$dest/repo" -name "*.step" -o -name "*.stp" -o -name "*.stl" | while read f; do
            cp "$f" "$dest/"
        done
        echo "  [mfcad] Done."
    else
        echo "  [mfcad] Could not clone. Download manually from https://github.com/hducg/MFCAD"
    fi
}

# ---------------------------------------------------------------------------
# Map config size to datasets
# ---------------------------------------------------------------------------

case "$SIZE" in
    small)
        DATASETS="thingi10k mfcad fusion360"
        echo "=== Downloading datasets for SMALL config ==="
        echo "=== Datasets: $DATASETS (~10-15 GB preprocessed) ==="
        ;;
    medium)
        DATASETS="thingi10k objaverse mfcad fusion360 partnet"
        echo "=== Downloading datasets for MEDIUM config ==="
        echo "=== Datasets: $DATASETS (~50-60 GB preprocessed) ==="
        ;;
    large)
        DATASETS="abc objaverse objaverse_xl thingi10k partnet fusion360 mfcad"
        echo "=== Downloading datasets for LARGE config ==="
        echo "=== Datasets: $DATASETS (~500 GB+ preprocessed) ==="
        ;;
    *)
        echo "Usage: $0 [small|medium|large]"
        echo "  small  : thingi10k, mfcad, fusion360"
        echo "  medium : thingi10k, objaverse, mfcad, fusion360, partnet"
        echo "  large  : abc, objaverse, objaverse_xl, thingi10k, partnet, fusion360, mfcad"
        exit 1
        ;;
esac

echo ""

for ds in $DATASETS; do
    echo "------------------------------------------------------------"
    download_"$ds"
    echo ""
done

echo "------------------------------------------------------------"
echo "=== Downloads complete ==="
echo ""
echo "Next step — preprocess into training-ready .pt files:"
echo "  python -m shape_foundation.scripts.prepare_dataset --source <name> --root data_raw/<name> --output data_cache/<name>"
echo ""
echo "Or preprocess all at once (uses config defaults):"
echo "  python -m shape_foundation.scripts.prepare_dataset --config configs/${SIZE}.yaml"
echo ""
echo "NOTE: For the config-based preprocessing, update the source roots in the config"
echo "or symlink data_raw/<name> to data_cache/<name> so prepare_dataset can find them."
