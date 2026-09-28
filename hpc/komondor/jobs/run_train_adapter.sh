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

# STAGE 9 — per-neighbour tokens, GATv2 and r=64, all three at once.
#
# Trains the graph encoder, prefix adapter and neighbour-token adapter against Qwen3-14B,
# soft-prompt-only: the prompt is one fixed system message and user turn with no references,
# so the graph is the only channel to the paper (src/adapter/text.py).
# See docs/soft_prompt_only.md, step 6.
#
# What step 5 found, and why this is the answer to it. Step 5 added LoRA to the graph arm and
# it got WORSE: judge accuracy 1.737 -> 1.490, distinct abstracts 809 -> 453 of 945, top-1
# identifiability 1.69% -> 0.95%. That is not noise, it is conditioning collapse — given a
# prefix carrying little about the paper, a decoder with the capacity to adapt learns to
# ignore it and emit a good generic abstract instead. The oracle arm rules out the channel as
# the cause: a *perfect* target vector through the identical 32-token, 40-layer prefix reaches
# 11.96% top-1. The channel works; the single pooled 1024-d vector is what does not carry the
# paper. So the fix has to add information, not capacity.
#
# Three changes against step 5 (checkpoints/qwen14b-graphonly-kv192-lora):
#
#   neighbour tokens   each 1-hop reference becomes its own input embedding, prepended to the
#                      prompt (config.NUM_NEIGHBOUR_TOKENS = 64, chosen by centroid distance).
#                      This is the one that removes the compression; the other two are cheap
#                      riders on the same run.
#   --gnn-conv gatv2   learned attention over the references instead of an unweighted mean.
#   LoRA r=64          four times the rank, plus k_proj, which the neighbour tokens need —
#                      their keys are computed rather than supplied (src/adapter/lora.py).
#
# Three at once is a deliberate trade against the single-change discipline the earlier steps
# kept, and it costs attribution: a gain cannot be split between them from this run alone.
# It was taken because GPU time is short. The ablations that recover the attribution are all
# cheap and generation-only, and run_generate_eval.sh already carries two of them (gate0 and
# ngate0) against these same weights.
#
# Qwen3-14B's own weights are never written; only checkpoints/epoch*.pt are written. They are
# larger than step 5's ~560 MB: LoRA goes from 10.5M parameters to ~42M at r=64 with k_proj
# added, and the neighbour-token adapter adds 5.3M, each carrying AdamW's two moment buffers
# on top. Budget ~1 GB per epoch and watch the quota — 10 epochs is ~10 GB if every epoch
# improves.
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
# --epochs is NOT passed, so config.EPOCHS = 10 applies — the same budget every step since 1
# has had. That one IS held, even though three other things move: an arm trained for a
# different number of epochs than the run it is compared against would confound the comparison
# on top of the three changes already being made. Early stopping (PATIENCE = 2) ends it sooner
# if it is going to.
#
# Walltime: step 5 ran this budget end to end with the encoder training, so 48 h is measured
# rather than guessed for the *old* sequence length. This run adds up to 64 input tokens per
# node on top of a ~250-token prompt, so the forward is roughly 25% longer, and r=64 on three
# projections costs more in the backward. Expect it to be slower than step 5. The resume
# below covers a timeout.
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
# ntok64-gatv2-lora64: its own directory, and NOT step 5's
# checkpoints/qwen14b-graphonly-kv192-lora. train.py refuses to resume across a change of
# --gnn-conv or --neighbour-tokens, so the mistake aborts rather than trains — but best.pt is
# a fixed name in every one of these directories, so pointing this at step 5's would overwrite
# the arm this run is measured against. Step 5's directory is the comparison; leave it intact.
# Same argument against checkpoints/qwen14b-oracle-kv192-lora, which holds the ceiling run.
# graphonly: separate from checkpoints/qwen14b/, which holds the text-path run. train.py refuses
# to resume across a prompt change, but the fixed best.pt name would still be overwritten.
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-ntok64-gatv2-lora64
LORA=1                                  # rank 64 on q/k/v, per config.py. ON here and not
                                        # optional: the neighbour tokens are input embeddings,
                                        # so without k_proj the model cannot learn to attend
                                        # to them at all

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

    # Cheap and CPU-only (~15 s): proves the neighbour tokens mean the same thing in the
    # one-pass forward training uses and the two-pass prefill generation uses, and that
    # the 1-hop selection picks the seed own references in neighbour_cap order. Every way
    # this can be wrong yields fluent text rather than an error.
    # NOTE: no apostrophes inside this block — it is one single-quoted argument to
    # bash -lc, and an apostrophe silently ends the quote and splits the script.
    python verify_neighbour_tokens.py || exit 1

    python train.py \
        --gnn-layers 2 \
        --gnn-conv gatv2 \
        --neighbour-tokens 64 \
        --out-dir "$CKPT_DIR" \
        --wandb \
        --wandb-run "ntok64-gatv2-lora64-$SLURM_JOB_ID" \
        ${LORA:-} ${RESUME:-}
'
