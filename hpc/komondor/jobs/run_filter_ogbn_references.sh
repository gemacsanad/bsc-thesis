#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=s2-filter-ogbn
#SBATCH --partition=bigdata
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=06:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/s2_filter_ogbn_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/s2_filter_ogbn_%j.err
#SBATCH --mail-type=END,FAIL

# Expands the ogbn-arxiv node set with the references those nodes make
# (via the full data/processed/s2/references.parquet citation graph built
# by run_download_dataset.sh) and writes papers_filtered.parquet /
# references_subset.parquet.
#
# CPU/DuckDB-only (no GPU, no internet needed) - runs on "bigdata" purely
# for the CPU/memory sizing, reusing the venv set up by
# run_download_dataset.sh (must have been run at least once already so
# data/processed/s2/{papers,references}.parquet exist).

set -euo pipefail

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SCRIPT_DIR=$BASE_DIR/src/data/s2
PROC_DIR=$BASE_DIR/data/processed/s2
TMP_DIR=$BASE_DIR/duckdb_tmp
VENV_DIR=$BASE_DIR/.venv_s2_download
NODES_CSV=$BASE_DIR/data/raw/ogbn_arxiv/nodes.csv

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

"$VENV_DIR/bin/python" filter_ogbn_references.py \
    --nodes-csv "$NODES_CSV" \
    --papers "$PROC_DIR/papers.parquet" \
    --references "$PROC_DIR/references.parquet" \
    --out-dir "$PROC_DIR" \
    --tmp-dir "$TMP_DIR" \
    --threads 16 \
    --memory-limit 48GB

echo "Done. Outputs in $PROC_DIR/"
