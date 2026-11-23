#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=s2-download
#SBATCH --partition=bigdata
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=24:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/s2_download_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/s2_download_%j.err
#SBATCH --mail-type=END,FAIL

# Downloads the Semantic Scholar bulk datasets and joins them into papers.parquet and
# references.parquet. Runs on the "bigdata" partition, whose nodes have internet access.

set -euo pipefail

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SCRIPT_DIR=$BASE_DIR/src/data/s2
RAW_DIR=$BASE_DIR/data/raw/s2
PROC_DIR=$BASE_DIR/data/processed/s2
TMP_DIR=$BASE_DIR/duckdb_tmp
VENV_DIR=$BASE_DIR/.venv_s2_download

mkdir -p "$RAW_DIR" "$PROC_DIR" "$BASE_DIR/logs"

# The default python3 is too old for duckdb>=0.10.
module load cray-python/3.11.7

# A small venv for the download scripts; the marker file is written only after a successful install.
SETUP_MARKER="$VENV_DIR/.setup_complete"
if [ ! -f "$SETUP_MARKER" ]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
    touch "$SETUP_MARKER"
fi

cd "$SCRIPT_DIR"

echo "[1/2] Downloading raw shards (papers, abstracts, citations)..."
"$VENV_DIR/bin/python" download_shards.py --out-dir "$RAW_DIR"

echo "[2/2] Building consolidated mag_id-keyed papers.parquet / references.parquet..."
"$VENV_DIR/bin/python" build_dataset.py \
    --raw-dir "$RAW_DIR" \
    --out-dir "$PROC_DIR" \
    --tmp-dir "$TMP_DIR" \
    --threads 16 \
    --memory-limit 48GB

echo "Done. Outputs in $PROC_DIR/"

