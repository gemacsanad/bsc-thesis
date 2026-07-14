#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=baseline-generate-random
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=06:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/generate_random_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/generate_random_%j.err
#SBATCH --mail-type=END,FAIL

# Runs lmsys/vicuna-7b-v1.5 as a BASELINE (no LoRA adapter).
# Uses randomly-wired edges (random_sample/) instead of real citation edges.
# Model must be pre-downloaded into $BASE_DIR/.hf_cache on the login node
# before submitting (see instructions). Compute nodes have no internet.
# Results are written to random_sample/test_results_random_sampled.csv.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/baseline-gpu.sif
MODEL_ID=lmsys/vicuna-7b-v1.5

mkdir -p $BASE_DIR/logs

module load singularity

singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app
    nvidia-smi
    python generate_test_csv.py \
        --model_path=lmsys/vicuna-7b-v1.5 \
        --nodes=random_sample/nodes_subset_random_sampled.csv \
        --edges=random_sample/edges_subset_random_sampled.csv \
        --output=random_sample/test_results_random_sampled.csv \
        --max_length=4096 \
        --max_new_tokens=256
'
