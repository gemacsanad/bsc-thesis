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

# STAGE 10 — scores the checkpoint from run_train_adapter.sh's step-7 run: a pretrained
# encoder, the contrastive ranking loss, opened gates and LoRA back to r=16. Pooled prefix
# only — this run has no per-neighbour tokens.
# See docs/soft_prompt_only.md, step 7, and docs/step6_failure_analysis.md.
#
# THE BAR IS STEP 2, not step 5 or step 6. Beating the runs that collapsed proves nothing.
# The frozen-LLM graph run (experiments/2026-09-17-14395431_...) is the best the graph has
# ever done and every LoRA run since has been worse than it:
#   judge accuracy 1.737, distinct abstracts 809/945, rows scoring 3+ 137, top-1 ID 2.86%.
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
# RUN_GATE0=1 adds a second arm: the checkpoint's own weights with the gate forced to zero, so
# the prefix is node-independent. It is 1 here and it is not optional — it is the control the
# whole stage has been missing. This run trains LoRA, so the adapter arm differs from step 2's
# frozen graph run by TWO things at once, LoRA and the graph, and a gain could be credited to
# either. gate=0 keeps the LoRA weights and disconnects only the graph, so adapter-minus-gate0
# is the graph's contribution and nothing else. Steps 1-4 all shipped without it; step 4's job
# asked for it and it did not land, which is why step 4 still cannot separate LoRA from its
# oracle vector.
#
# Read it as a FLOOR, not as a paired control. In the graph-only regime the prompt carries no
# node id either (text.build_graph_only_prompt takes no node argument), so at gate=0 both the
# prompt and the prefix are node-independent and greedy decoding emits ONE identical abstract
# for all 945 nodes. The scores still vary, because each row is scored against a different
# gold, so the 945-row mean is a real number — it is what LoRA plus a generic prefix earns
# knowing nothing about the paper. But "improved on N nodes" against it would be measuring the
# golds, not the model.
#
# That also means the arm spends ~7 h to produce one distinct string. It is left on because
# the walltime fits both arms and this run must not lose the control a second time; if the GPU
# budget is tight, the cheap equivalent is to generate a handful of nodes, verify the output is
# constant, and replicate it across the 945 rows for scoring.
#
# Nothing is trained and nothing is written to W&B; no .env is needed.
#
# Prerequisites:
#   - adapter-qwen-gpu.sif built (hpc/komondor/containers/adapter-qwen-gpu.def) — the same image
#     train.py used, with no extra packages: generate.py imports nothing train.py did not
#   - a checkpoint in checkpoints/qwen14b-ntok64-gatv2-lora64/ (produced by run_train_adapter.sh)
#   - Qwen/Qwen3-14B already in $BASE_DIR/.hf_cache — compute nodes do not re-download it
#
# Walltime is 24 h against an estimated ~7 h per condition, TWO conditions here (adapter and
# gate0) and no extra input tokens, since this run is pooled-prefix only. That fits with
# margin. ngate0 is off because it is meaningless without neighbour tokens — generate.py
# asserts on it rather than quietly scoring the adapter arm under an ablation's name.
#
# If the queue is tight, drop RUN_GATE0: with no neighbour tokens, k no longer varies, so the
# arm really does spend ~7 h producing one distinct string, and the cheap equivalent is to
# generate a handful of nodes, verify the output is constant and replicate it for scoring.
#
# The original estimate follows.
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
# This run reads checkpoints/qwen14b-rank-gatv2-lora16/ — NOT the oracle directory, and not
# step 5's or step 6's. An oracle checkpoint has exactly the shapes of a graph one, so the directory alone
# would not catch a mix-up; generate.py refuses it on the checkpoint's recorded `oracle` flag,
# which is why --oracle is absent from every invocation below and must stay absent.
# Nothing here names an architecture either: the encoder kind, the encoder depth, the
# neighbour-token width and the LoRA tensors are all read back out of the checkpoint, so a
# wrong directory scores another run's weights but can never build the wrong model for them.
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-rank-gatv2-lora16
ARM=rank-gatv2-lora16                   # experiments/<date>-<job>_$ARM/
RUN_BASELINE=0                          # 0 to generate the adapter arm only
RUN_GATE0=1                             # graph off in BOTH channels — the floor
RUN_NGATE0=0                            # OFF: there are no neighbour tokens in this run, so
                                        # the arm has nothing to ablate. generate.py asserts
                                        # rather than scoring the adapter arm under its name

# The LAST epoch, not best.pt. best.pt is chosen by train.py's selection_score, which is meant
# to read validation ranking accuracy — but NegativePool is indexed over the *training* nodes,
# so no validation node is ever in it, rank_acc is never computed, and selection silently falls
# back to LM validation loss: the criterion step 6 proved is anti-correlated with conditioning.
# It picked epoch 4 here, and the ranking loss did not leave chance until step ~2.1k (epoch 5),
# where rank_hit went 0.11 -> 0.35 and the gates opened while the LM validation loss rose. So
# best.pt is the checkpoint from immediately before the effect this run exists to measure.
# `sort -V` so epoch10 would beat epoch2. Restore best.pt once that selection bug is fixed.
LATEST=$(ls -1 "$CKPT_DIR"/epoch*.pt 2>/dev/null | sort -V | tail -1)
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
export SINGULARITYENV_RUN_NGATE0="$RUN_NGATE0"

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

    # And that the two-pass prefill these arms decode through is the one-pass forward that
    # training used. A mismatch here is fluent text at a silent offset, not an error.
    python verify_neighbour_tokens.py || exit 1

    if [ "$RUN_BASELINE" = 1 ]; then
        echo "=== baseline: frozen Qwen3-14B, no prefix ==="
        python generate.py --mode baseline \
            --output "/app/experiments/${RUN_TAG}_qwen14b-baseline/generations.csv"
    else
        echo "=== baseline skipped (RUN_BASELINE=0) ==="
    fi

    if [ "$RUN_GATE0" = 1 ]; then
        echo "=== gate=0: the same weights and LoRA with the graph disconnected ==="
        echo "    node-independent prompt AND prefix, and no neighbour tokens in this run —"
        echo "    expect one identical abstract x945"
        python generate.py --mode gate0 --checkpoint "$CKPT" \
            --output "/app/experiments/${RUN_TAG}_${ARM}-gate0/generations.csv"
    fi

    if [ "$RUN_NGATE0" = 1 ]; then
        echo "=== ngate=0: the k neighbour slots kept, emptied; pooled prefix still on ==="
        echo "    this is the step-5 architecture measured on step-6 weights"
        python generate.py --mode ngate0 --checkpoint "$CKPT" \
            --output "/app/experiments/${RUN_TAG}_${ARM}-ngate0/generations.csv"
    fi

    echo "=== stage 10: pretrained GATv2 prefix + ranking loss + LoRA r=16 ($ARM) ==="
    python generate.py --mode adapter --checkpoint "$CKPT" \
        --output "/app/experiments/${RUN_TAG}_${ARM}/generations.csv"
'
