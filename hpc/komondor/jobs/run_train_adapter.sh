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
# The hyperparameters are NOT here. Epochs, patience, batch size, gradient accumulation,
# logging cadence, seed, base model and the W&B project are constants in src/adapter/config.py,
# so they are changed in one place and this script does not restate them. What is passed below
# is only what differs between two submissions.
#
# Walltime is 48 h against config.EPOCHS = 10 and PATIENCE = 2, so the run usually ends well
# before that. The estimate is ~4.3 h per epoch — 7,750 nodes at a ~3,428-token mean, 2.9x the
# token count of the old 4,000-node set — and it has never been checked on a real A100. The
# resume below covers a timeout, and the patience counter survives it because train.py stores
# it in the checkpoint.
#
# Encoder depth: edit --gnn-layers and CKPT_DIR below, submit, then edit them again for the
# next depth. That is safe, and only because it is *this* file: sbatch copies the batch script
# into the job spool at submit time, so a queued job keeps the version it was submitted with.
# The Python under /app has no such protection — it is read when the job starts, after however
# long the queue takes. That is precisely why the depth is a --gnn-layers argument and not the
# GNN_LAYERS constant in config.py: editing the constant between two submissions would let a
# still-queued job pick up the value meant for the next one.
#
# The two must be edited together. Each depth needs its own checkpoint directory: the resume
# below takes the newest epoch file it finds and best.pt is a single fixed name, so two depths
# sharing one directory would resume from and overwrite each other. train.py refuses a resume
# whose recorded depth differs from --gnn-layers, so the mistake aborts rather than trains.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-gpu.sif

CKPT_DIR=$BASE_DIR/checkpoints          # 3 layers -> $BASE_DIR/checkpoints/gnn3, etc.
LORA=1                                  # 1 also trains LoRA on the LLM's q/v projections

mkdir -p "$CKPT_DIR" $BASE_DIR/logs

# LoRA is recorded in every checkpoint and train.py refuses to resume across a change of it,
# so a LoRA run needs its own CKPT_DIR — the resume below picks up whatever epoch file is
# newest, which would otherwise be another run's.
[ "$LORA" = 1 ] && export SINGULARITYENV_LORA="--lora"

# Pick up where a timed-out or crashed run left off. train.py restores the encoder, adapter,
# optimizer and scheduler, and restarts at the next *epoch* — progress within an epoch is lost.
LATEST=$(ls -1 "$CKPT_DIR"/epoch*.pt 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    export SINGULARITYENV_RESUME="--resume /app${CKPT_DIR#$BASE_DIR}/$(basename "$LATEST")"
    echo "resuming from $LATEST"
else
    echo "no checkpoint found — starting from scratch"
fi
export SINGULARITYENV_CKPT_DIR="/app${CKPT_DIR#$BASE_DIR}"

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

    # Same idea for LoRA: it edits the module tree the prefix is injected into, and every way
    # that can go wrong yields fluent text rather than an error.
    [ -n "${LORA:-}" ] && { python verify_lora_prefix.py || exit 1; }

    python train.py \
        --gnn-layers 2 \
        --out-dir "$CKPT_DIR" \
        --wandb \
        --wandb-run "adapter-gnn2-$SLURM_JOB_ID" \
        ${LORA:-} ${RESUME:-}
'
