#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=baseline-generate-filtered
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/generate_filtered_%A_%a.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/generate_filtered_%A_%a.err
#SBATCH --mail-type=END,FAIL

# Runs lmsys/vicuna-7b-v1.5 as a BASELINE (no LoRA adapter), once per cosine
# filter configuration, using cosine_filter/generation/generate_filtered_csv.py.
# Thresholds come from cosine_filter/analysis.ipynb (the summary table cell)
# for target non-relevant retention rates of 30%, 60% and 90% on
# cosine_score_mean and centroid_cosine_distance.
#
# Model must be pre-downloaded into $BASE_DIR/.hf_cache on the login node
# before submitting (see instructions). Compute nodes have no internet.
#
# Each array task (SLURM_ARRAY_TASK_ID 0-5) runs one configuration and writes
# to its own folder under cosine_filter/generation/<name>/test_results.csv.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/baseline-gpu.sif
MODEL_ID=lmsys/vicuna-7b-v1.5

mkdir -p $BASE_DIR/logs

# name                          metric                     threshold
CONFIGS=(
    "cosine_score_mean_30        cosine_score_mean        0.628869"
    "cosine_score_mean_60        cosine_score_mean        0.560702"
    "cosine_score_mean_90        cosine_score_mean        0.473255"
    "centroid_cosine_distance_30 centroid_cosine_distance 0.098132"
    "centroid_cosine_distance_60 centroid_cosine_distance 0.128711"
    "centroid_cosine_distance_90 centroid_cosine_distance 0.179584"
)

read -r NAME METRIC THRESHOLD <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUTDIR=cosine_filter/generation/$NAME
mkdir -p $BASE_DIR/$OUTDIR

module load singularity

singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc "
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app
    nvidia-smi
    python cosine_filter/generation/generate_filtered_csv.py \
        --model_path=$MODEL_ID \
        --nodes=ogbn_arxiv_dataset/nodes_subset.csv \
        --edges=ogbn_arxiv_dataset/edges_subset.csv \
        --output=$OUTDIR/test_results.csv \
        --filter_metric=$METRIC \
        --filter_threshold=$THRESHOLD \
        --max_length=4096 \
        --max_new_tokens=256
"
