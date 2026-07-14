#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=graphsage-full
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=02:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/graphsage_full_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/graphsage_full_%j.err
#SBATCH --mail-type=END,FAIL

# Full-graph GraphSAGE node classification (no k-hop subsampling).
# Reads GraphSAGE_classification/full_graph_data.pt (built by
# full_graph_extraction.ipynb — run that notebook first if the file
# doesn't exist yet) and logs metrics live to Weights & Biases.
# WANDB_KEY is read from GraphSAGE_classification/.env by the training
# script itself (not passed on the command line).

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/Komondor_files/graphsage-gpu.sif

mkdir -p $BASE_DIR/logs

module load singularity

singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    cd /app/GraphSAGE_classification
    nvidia-smi
    python train_graphsage_full.py
'
