#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=s2-add-titles
#SBATCH --partition=bigdata
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=08:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/s2_add_titles_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/s2_add_titles_%j.err
#SBATCH --mail-type=END,FAIL

# Adds `title` and `year` columns to data/processed/s2/papers_filtered.parquet
# by re-reading the raw 'papers' shards (already downloaded by
# run_download_dataset.sh) - build_dataset.py only ever kept corpusid /
# externalids.MAG from those shards, so title/year never made it into
# papers.parquet / papers_filtered.parquet.
#
# CPU/DuckDB-only (no GPU, no internet needed) - runs on "bigdata" purely
# for the CPU/memory sizing, reusing the venv set up by
# run_download_dataset.sh. Requires data/raw/s2/papers/*.jsonl.gz (from
# run_download_dataset.sh) and data/processed/s2/papers_filtered.parquet
# (from run_filter_ogbn_references.sh) to already exist.

set -euo pipefail

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SCRIPT_DIR=$BASE_DIR/src/data/s2
RAW_DIR=$BASE_DIR/data/raw/s2
PROC_DIR=$BASE_DIR/data/processed/s2
TMP_DIR=$BASE_DIR/duckdb_tmp
VENV_DIR=$BASE_DIR/.venv_s2_download

mkdir -p "$BASE_DIR/logs"

module load cray-python/3.11.7

SETUP_MARKER="$VENV_DIR/.setup_complete"
if [ ! -f "$SETUP_MARKER" ]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
    touch "$SETUP_MARKER"
fi

cd "$SCRIPT_DIR"

"$VENV_DIR/bin/python" add_titles_years.py \
    --raw-dir "$RAW_DIR" \
    --papers-filtered "$PROC_DIR/papers_filtered.parquet" \
    --tmp-dir "$TMP_DIR" \
    --threads 16 \
    --memory-limit 48GB

echo "Done. Updated $PROC_DIR/papers_filtered.parquet with title/year columns."
