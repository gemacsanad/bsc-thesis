#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=adapter-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=24:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/generate_eval_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/generate_eval_%j.err
#SBATCH --mail-type=END,FAIL

# ORACLE CEILING RUN (--oracle). NOT a result arm: the adapter is fed the target abstract's own
# voyage embedding, so this arm sees the answer it is asked to write. Its score bounds what any
# graph could achieve through this prefix, and must never be reported beside the graph arms
# without that label. See docs/soft_prompt_only.md, step 4.
#
# Stage 5 — generates the evaluation CSV over the frozen 945-node capped eval set:
#
#   experiments/<date>_qwen14b-baseline/generations.csv  frozen Qwen3-14B, no prefix
#   experiments/<date>_$ARM/generations.csv              GraphSAGE + prefix adapter, same frozen Qwen3
#
# Both conditions run back to back in ONE job, on the same node and the same GPU, so nothing but
# the prefix differs between the two files. Greedy decoding, bf16, batch size 1 — see
# docs/adapter_evaluation.md for why each of those is fixed.
#
# RUN_BASELINE=0 skips the control and halves the walltime. Only do that when the decoding
# settings are unchanged from a run whose baseline already exists — the GNN gap is a difference
# between the two arms, so a baseline generated under different decoding cannot be reused.
# It is 0 for the soft-prompt-only run. Its baseline would get the graph-only prompt with no
# prefix — no information about the paper at all — and write one identical abstract for all 945
# nodes under greedy decoding. The comparison arm is the existing text-path Qwen baseline,
# experiments/2026-09-10-14364743_qwen14b-baseline (docs/soft_prompt_only.md).
#
# RUN_GATE0=1 adds a third arm: the checkpoint's own weights with the gate forced to zero, so
# the prefix is node-independent. It is 1 here, and this is the first run where it is not
# optional. This run trains LoRA, so the adapter arm differs from the 2026-09-10 frozen-LLM
# baseline by TWO things at once — LoRA and the oracle vector — and a gain could be credited to
# either. gate=0 keeps the LoRA weights and disconnects only the vector, so adapter-minus-gate0
# is the oracle's contribution and nothing else. Steps 1, 2 and 3 all shipped without it and all
# three recorded that as a known limitation; it is the missing control of this whole stage.
#
# It needs --oracle too. generate.py loads the checkpoint for gate0 exactly as for adapter, and
# load_adapter asserts the recorded `oracle` flag matches the command line, so omitting it here
# aborts the arm. The flag is harmless in this mode: the oracle vector is still computed and
# then multiplied by a zero gate.
#
# Nothing is trained and nothing is written to W&B; no .env is needed.
#
# Prerequisites:
#   - adapter-qwen-gpu.sif built (hpc/komondor/containers/adapter-qwen-gpu.def) — the same image
#     train.py used, with no extra packages: generate.py imports nothing train.py did not
#   - a checkpoint in checkpoints/qwen14b-oracle-kv192-lora/ (produced by run_train_adapter.sh)
#   - Qwen/Qwen3-14B already in $BASE_DIR/.hf_cache — compute nodes do not re-download it
#
# Walltime is 24 h against an estimated ~7 h per condition, twice. That scales the old Vicuna
# estimate of ~2.9 h (945 nodes, batch size 1) by 2x for Qwen3-14B's parameter count and 1.25x
# for MAX_NEW_TOKENS going 256 -> 320. Both factors are arithmetic, and the 2.9 h it scales was
# itself never checked on a real A100 — so treat 24 h as absorbing a miss, not as a measurement.
# The resume below covers a timeout anyway; lower it once one condition has been timed.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-qwen-gpu.sif

mkdir -p $BASE_DIR/logs

# Which run to score: point CKPT_DIR at the depth's training output, and name RUN_TAG's
# suffix to match. Nothing here selects an architecture — generate.py reads the encoder depth
# back out of the checkpoint — so a wrong directory scores another run's weights, but can
# never build the wrong encoder for them.
#
# It must match run_train_adapter.sh's CKPT_DIR. The Qwen run writes to checkpoints/qwen14b/;
# $BASE_DIR/checkpoints still holds the Vicuna run, and best.pt is a fixed name in both, so
# the unqualified directory would silently select the wrong model's weights. The load fails on
# a shape mismatch rather than scoring the wrong thing — the adapter output moved from
# 32x2x32x10x128 to 40x2x8x32x128 — but only after the 29.6 GB model is on the card.
# This run writes to checkpoints/qwen14b-oracle-kv192-lora/ — NOT step 3's frozen-LLM oracle
# directory. An oracle checkpoint has exactly the shapes of a graph one, so the directory alone
# would not catch a mix-up — generate.py refuses it on the checkpoint's recorded `oracle` flag,
# and --oracle below must match it. The LoRA tensors ride in the checkpoint and generate.py
# injects them automatically, so there is no LoRA flag to keep in sync here.
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-oracle-kv192-lora
ARM=oracle-qwen14b-kv192-lora           # experiments/<date>-<job>_$ARM/
RUN_BASELINE=0                          # 0 to generate the adapter arm only
RUN_GATE0=1                             # the control that isolates the oracle from LoRA (above)

# best.pt is the epoch with the lowest validation loss, written by train.py. Prefer it: with
# early stopping the *last* epoch is by definition worse than the best one, so picking the
# highest-numbered file would score a model the run had already decided to abandon.
# Fall back to the newest epoch file (same `sort -V` idiom, so epoch10 beats epoch2) for runs
# that predate best.pt.
LATEST="$CKPT_DIR/best.pt"
if [ ! -f "$LATEST" ]; then
    LATEST=$(ls -1 "$CKPT_DIR"/epoch*.pt 2>/dev/null | sort -V | tail -1)
    echo "no best.pt — falling back to the newest epoch checkpoint"
fi
if [ -z "$LATEST" ]; then
    echo "no checkpoint in $CKPT_DIR — run run_train_adapter.sh first" >&2
    exit 1
fi
echo "evaluating checkpoint $LATEST as $ARM"
export SINGULARITYENV_CKPT="/app${CKPT_DIR#$BASE_DIR}/$(basename "$LATEST")"

# Each condition gets its own experiment directory, named for the day, the job and the arm,
# so several depths evaluated on one day stay tellable apart in experiments/.
export SINGULARITYENV_RUN_TAG="$(date +%F)-${SLURM_JOB_ID:-manual}"
export SINGULARITYENV_ARM="$ARM"
export SINGULARITYENV_RUN_BASELINE="$RUN_BASELINE"
export SINGULARITYENV_RUN_GATE0="$RUN_GATE0"

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

    # And that generate.py builds the exact graph-only prompt the adapter was trained behind,
    # with no reference text in it. Seconds on CPU.
    python verify_graph_only_prompt.py || exit 1

    if [ "$RUN_BASELINE" = 1 ]; then
        echo "=== baseline: frozen Qwen3-14B, no prefix ==="
        python generate.py --mode baseline \
            --output "/app/experiments/${RUN_TAG}_qwen14b-baseline/generations.csv"
    else
        echo "=== baseline skipped (RUN_BASELINE=0) ==="
    fi

    if [ "$RUN_GATE0" = 1 ]; then
        echo "=== gate=0: the same weights with the graph disconnected ==="
        python generate.py --mode gate0 --oracle --checkpoint "$CKPT" \
            --output "/app/experiments/${RUN_TAG}_${ARM}-gate0/generations.csv"
    fi

    echo "=== ORACLE CEILING: target-embedding prefix + frozen Qwen3-14B ($ARM) ==="
    echo "    this arm sees the abstract it is asked to write — a ceiling, not a result"
    python generate.py --mode adapter --oracle --checkpoint "$CKPT" \
        --output "/app/experiments/${RUN_TAG}_${ARM}/generations.csv"
'
