#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=adapter-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=12:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/train_adapter_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/train_adapter_%j.err
#SBATCH --mail-type=END,FAIL

# Trains the GraphSAGE encoder + prefix adapter against a frozen Vicuna-7B.
# The LLM is never updated; only checkpoints/epoch*.pt are written (~196 MB each:
# 65 MB of weights plus AdamW's two moment buffers, which --resume needs).
#
# Metrics go live to Weights & Biases. WANDB_KEY is read from the repo-root .env by train.py itself
# (not passed on the command line), the same way train_graphsage_full.py does it.
#
# Prerequisites:
#   - adapter-gpu.sif built (hpc/komondor/containers/adapter-gpu.def)
#   - data/frozen/training_sets.json present (produced by src/adapter/text.py)
#   - .env at the repo root containing WANDB_KEY=...
#   - lmsys/vicuna-7b-v1.5 already in $BASE_DIR/.hf_cache — compute nodes cannot download it
#
# Walltime is 12 h against an estimated ~1.5 h/epoch, because that estimate has never been
# checked on a real A100. Lower it if the queue wait bites; the resume below covers a timeout.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-gpu.sif

mkdir -p $BASE_DIR/logs

# Pick up where a timed-out or crashed run left off. train.py restores the encoder, adapter,
# optimizer and scheduler, and restarts at the next *epoch* — progress within an epoch is lost.
LATEST=$(ls -1 "$BASE_DIR"/checkpoints/epoch*.pt 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    export SINGULARITYENV_RESUME="--resume /app/checkpoints/$(basename "$LATEST")"
    echo "resuming from $LATEST"
else
    echo "no checkpoint found — starting from scratch"
fi

module load singularity

singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app/src/adapter
    nvidia-smi
    python train.py \
        --epochs 2 \
        --batch-size 4 \
        --grad-accum 4 \
        --log-every 10 \
        --eval-every 100 \
        --wandb \
        --wandb-project dytag-adapter \
        --wandb-run "adapter-$SLURM_JOB_ID" \
        ${RESUME:-}
'
