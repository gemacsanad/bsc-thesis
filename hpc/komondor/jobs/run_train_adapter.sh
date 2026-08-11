#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=adapter-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=48:00:00
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
#   - data/frozen/training_sets_capped.json present (produced by src/adapter/prepare_data_capped.py)
#   - .env at the repo root containing WANDB_KEY=...
#   - lmsys/vicuna-7b-v1.5 already in $BASE_DIR/.hf_cache — compute nodes cannot download it
#   - checkpoints/ holding no run on the *legacy* node sets: train.py refuses to resume one,
#     so a stale epoch*.pt from the uncapped experiment aborts this job at startup. Move it
#     to checkpoints/legacy-uncapped/ before the first capped run.
#
# Walltime is 48 h. Training is capped at 10 epochs but stops early after 2 epochs without a
# better validation loss, so the run usually ends well before that. The estimate is ~4.3 h per
# epoch — 7,750 nodes at a ~3,428-token mean, 2.9x the token count of the old 4,000-node set —
# and it has never been checked on a real A100. The resume below covers a timeout, and the
# patience counter survives it because train.py stores it in the checkpoint.
#
# batch-size 2 x grad-accum 8 keeps the effective batch at 16 while halving the micro-batch:
# sequences now average ~3,428 tokens against the ~2,300 the batch-4 memory figure was measured
# at. If a smoke run shows headroom, 4 x 4 is the faster setting.

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

    # Cheap and CPU-only (~20 s): re-proves that the prompts train.py will build are the ones
    # the frozen node sets were selected under. The failure it guards against is silent — the
    # wrong references yield a fluent abstract, not an error — so it runs before the GPU hours.
    python verify_capped_prompts.py || exit 1

    python train.py \
        --epochs 10 \
        --patience 2 \
        --batch-size 2 \
        --grad-accum 8 \
        --log-every 10 \
        --eval-every 250 \
        --wandb \
        --wandb-project dytag-adapter \
        --wandb-run "adapter-$SLURM_JOB_ID" \
        ${RESUME:-}
'
