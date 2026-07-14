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

# Downloads the full Semantic Scholar bulk dataset (papers/abstracts/citations)
# and joins it into mag_id-keyed papers.parquet / references.parquet.
#
# Runs on the "bigdata" partition rather than the "ai" GPU partition used by
# run_generate.sh/run_train_graphsage_full.sh: this job is CPU/I/O-bound (no
# GPU needed), and unlike the "ai" compute nodes, bigdata nodes have internet
# access, so the download can run as a normal sbatch job instead of on the
# login node.

set -euo pipefail

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SCRIPT_DIR=$BASE_DIR/Download_script
DATA_DIR=$BASE_DIR/s2_data
VENV_DIR=$BASE_DIR/.venv_s2_download

mkdir -p "$DATA_DIR" "$BASE_DIR/logs"

# Set up a small venv for the download scripts (requests/duckdb/dotenv only)
# - lightweight enough not to need the GPU singularity container used by
# run_generate.sh/run_train_graphsage_full.sh.
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

cd "$SCRIPT_DIR"

echo "[1/2] Downloading raw shards (papers, abstracts, citations)..."
"$VENV_DIR/bin/python" download_shards.py --out-dir "$DATA_DIR"

echo "[2/2] Building consolidated mag_id-keyed papers.parquet / references.parquet..."
"$VENV_DIR/bin/python" build_dataset.py \
    --raw-dir "$DATA_DIR/raw" \
    --out-dir "$DATA_DIR/processed" \
    --tmp-dir "$DATA_DIR/duckdb_tmp" \
    --threads 16 \
    --memory-limit 48GB

echo "Done. Outputs in $DATA_DIR/processed/"

