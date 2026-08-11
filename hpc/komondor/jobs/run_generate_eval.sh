#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=adapter-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=12:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/generate_eval_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/generate_eval_%j.err
#SBATCH --mail-type=END,FAIL

# Stage 5 — generates both evaluation CSVs over the frozen 945-node capped eval set:
#
#   experiments/<date>_vicuna-baseline/generations.csv   frozen Vicuna-7B, no prefix
#   experiments/<date>_adapter/generations.csv           GraphSAGE + prefix adapter, same frozen Vicuna
#
# Both conditions run back to back in ONE job, on the same node and the same GPU, so nothing but
# the prefix differs between the two files. Greedy decoding, bf16, batch size 1 — see
# docs/adapter_evaluation.md for why each of those is fixed.
#
# Nothing is trained and nothing is written to W&B; no .env is needed.
#
# Prerequisites:
#   - adapter-gpu.sif built (hpc/komondor/containers/adapter-gpu.def) — the same image train.py used,
#     with no extra packages: generate.py imports nothing train.py did not
#   - a checkpoint in checkpoints/ (produced by run_train_adapter.sh)
#   - lmsys/vicuna-7b-v1.5 already in $BASE_DIR/.hf_cache — compute nodes do not re-download it
#
# Walltime is 12 h against an estimated ~2.9 h per condition (945 nodes x 256 greedy tokens at
# batch size 1, twice — 1.9x the old 495-node set, and none of them lost to overflow). That
# estimate has never been checked on a real A100. Lower it once one condition has actually been
# timed — the cost is queue time, and the resume below covers a timeout anyway.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-gpu.sif

mkdir -p $BASE_DIR/logs

# best.pt is the epoch with the lowest validation loss, written by train.py. Prefer it: with
# early stopping the *last* epoch is by definition worse than the best one, so picking the
# highest-numbered file would score a model the run had already decided to abandon.
# Fall back to the newest epoch file (same `sort -V` idiom, so epoch10 beats epoch2) for runs
# that predate best.pt.
LATEST="$BASE_DIR/checkpoints/best.pt"
if [ ! -f "$LATEST" ]; then
    LATEST=$(ls -1 "$BASE_DIR"/checkpoints/epoch*.pt 2>/dev/null | sort -V | tail -1)
    echo "no best.pt — falling back to the newest epoch checkpoint"
fi
if [ -z "$LATEST" ]; then
    echo "no checkpoint in $BASE_DIR/checkpoints — run run_train_adapter.sh first" >&2
    exit 1
fi
echo "evaluating checkpoint $LATEST"
export SINGULARITYENV_CKPT="/app/checkpoints/$(basename "$LATEST")"

# Each condition gets its own experiment directory, named for the day and the job.
export SINGULARITYENV_RUN_TAG="$(date +%F)-${SLURM_JOB_ID:-manual}"

module load singularity

# Both modes are resumable: generate.py appends only the node_ids missing from its output CSV,
# so a resubmit after a timeout costs at most one node, and an already-finished condition costs
# only the model load. Rerunning this script is therefore safe.
singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app/src/adapter
    nvidia-smi

    # Cheap and CPU-only (~10 s): re-proves that the prefix survives generate(). The failure it
    # guards against is silent — a cache_position off by 10 yields fluent text, not an error —
    # so it runs before the GPU hours, not after.
    python verify_generation.py || exit 1

    echo "=== baseline: frozen Vicuna-7B, no prefix ==="
    python generate.py --mode baseline \
        --output "/app/experiments/${RUN_TAG}_vicuna-baseline/generations.csv"

    echo "=== adapter: GraphSAGE + prefix adapter + frozen Vicuna-7B ==="
    python generate.py --mode adapter --checkpoint "$CKPT" \
        --output "/app/experiments/${RUN_TAG}_adapter/generations.csv"
'
