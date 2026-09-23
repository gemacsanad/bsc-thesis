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

# GRAPH + LoRA. This IS a result arm — the headline number of the whole soft-prompt-only
# stage, and the first graph run measured on an apparatus that works.
#
# Trains the GraphSAGE encoder and prefix adapter against Qwen3-14B, soft-prompt-only: the
# prompt is one fixed system message and user turn with no references, so the prefix is the
# only channel to the paper (src/adapter/text.py). See docs/soft_prompt_only.md, step 5.
#
# Why rerun a configuration that already produced a null. Steps 1 and 2 held the LLM frozen and
# read ~chance identifiability as "the graph contributes nothing". Step 4 then ran the oracle
# — a perfect target vector through the same 32-token, 40-layer prefix — twice, frozen and with
# LoRA, and top-1 identifiability moved from 0.4% to 12.0% (chance 0.11%) with no change to the
# channel at all. The prefix was always carrying paper identity; a frozen Qwen could not read
# it. That makes every graph null so far a measurement of a crippled apparatus, and this run is
# what replaces them.
#
# Exactly one thing changes against step 2's graph run (checkpoints/qwen14b-graphonly-kv192):
#
#   --lora    trains rank-16 adapters on the LLM's q/v projections alongside the prefix.
#
# Everything else is held: same 2-layer encoder, same kv192 adapter, same graph-only prompt,
# same 10 epochs, same node sets. That discipline is deliberate. Step 4 changed two things at
# once (LoRA and the oracle) and its 1.25 -> 2.71 therefore cannot be split between them; this
# run is the control that makes the next architecture decision attributable.
#
# Qwen3-14B's own weights are never written; only checkpoints/epoch*.pt are written (~560 MB
# each: 145 MB of adapter weights plus AdamW's two moment buffers, which --resume needs, and
# ~126 MB more than the frozen runs because the LoRA tensors and their moments now ride along.
# 10 epochs at ~560 MB is ~5.6 GB if every epoch improves — watch the quota).
#
# Metrics go live to Weights & Biases. WANDB_KEY is read from the repo-root .env by train.py itself
# (not passed on the command line), the same way train_graphsage_full.py does it.
#
# Prerequisites:
#   - adapter-qwen-gpu.sif built (hpc/komondor/containers/adapter-qwen-gpu.def)
#   - data/frozen/training_sets_capped.json present (produced by src/adapter/prepare_data_capped.py)
#   - .env at the repo root containing WANDB_KEY=...
#   - Qwen/Qwen3-14B already in $BASE_DIR/.hf_cache — compute nodes cannot download it
#   - checkpoints/ holding no run on the *legacy* node sets: train.py refuses to resume one,
#     so a stale epoch*.pt from the uncapped experiment aborts this job at startup. Move it
#     to checkpoints/legacy-uncapped/ before the first capped run.
#
# The hyperparameters are NOT here. Epochs, patience, batch size, gradient accumulation,
# logging cadence, seed, base model and the W&B project are constants in src/adapter/config.py,
# so they are changed in one place and this script does not restate them. What is passed below
# is only what differs between two submissions.
#
# --epochs is NOT passed, so config.EPOCHS = 10 applies — the same budget steps 1, 2 and 4 had.
# Holding it there is part of the single-change discipline above: a graph arm trained for a
# different number of epochs than the run it is compared against would confound the comparison
# the same way step 4's two simultaneous changes did. Early stopping (PATIENCE = 2) ends it
# sooner if it is going to.
#
# Walltime: step 4 ran this budget with LoRA on graph-only sequences and finished, so 48 h is
# no longer a guess for the epoch count — but the encoder now trains too (it was bypassed
# under --oracle), which adds a sampled 2-hop neighbourhood load and a backward pass per step.
# Expect it to be slower than step 4, not faster. The resume below covers a timeout.
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
SIF=$BASE_DIR/adapter-qwen-gpu.sif

# Its own directory, and not optional. The resume below takes the newest epoch*.pt it finds and
# best.pt is a fixed name, so sharing a directory with the Vicuna checkpoints would resume from
# and overwrite the wrong run. train.py refuses a resume across a change of --sets or
# --gnn-layers, but it has no way to know the base model changed underneath it.
# graphonly-kv192-lora: the graph run on the kv192 adapter, with LoRA. Its own directory, and
# NOT step 2's checkpoints/qwen14b-graphonly-kv192: train.py refuses to resume across a change
# of --lora (the prefix would continue against LLM weights it never saw), but best.pt is a fixed
# name in both, so pointing this at step 2's directory would abort the job at startup at best
# and overwrite step 2's result at worst. Step 2's directory is the comparison arm for this
# run — leave it intact. Same argument against checkpoints/qwen14b-oracle-kv192-lora, which
# holds step 4's ceiling run and differs only by --oracle.
# graphonly: separate from checkpoints/qwen14b/, which holds the text-path run. train.py refuses
# to resume across a prompt change, but the fixed best.pt name would still be overwritten.
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-graphonly-kv192-lora
LORA=1                                  # also trains LoRA on the LLM's q/v projections. ON here,
                                        # and it is the point of the run: every *graph* arm so
                                        # far has held the LLM frozen, so every graph null is a
                                        # statement about a frozen model only

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

    # Cheap and CPU-only (seconds): pins the exact graph-only prompt, checks it carries no
    # reference text, and that every frozen node has a trainable target. A moved chat template
    # yields fluent text, not an error — so it runs before the GPU hours.
    python verify_graph_only_prompt.py || exit 1

    # And that the loss train.py computes is the one transformers would have computed. The
    # head is applied to the supervised positions only, to keep a 151,936-token vocabulary
    # off the card; a wrong shift or denominator there also just trains.
    python verify_sliced_loss.py || exit 1

    # Same idea for LoRA: it edits the module tree the prefix is injected into, and every way
    # that can go wrong yields fluent text rather than an error.
    [ -n "${LORA:-}" ] && { python verify_lora_prefix.py || exit 1; }

    python train.py \
        --gnn-layers 2 \
        --out-dir "$CKPT_DIR" \
        --wandb \
        --wandb-run "graphonly-qwen14b-kv192-lora-$SLURM_JOB_ID" \
        ${LORA:-} ${RESUME:-}
'
