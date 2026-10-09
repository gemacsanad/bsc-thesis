#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=decoder-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=24:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/generate_decoder_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/generate_decoder_%j.err
#SBATCH --mail-type=END,FAIL

# RUN 9 — evaluates the decoder from run_train_decoder.sh four ways, on the same 945 papers as
# every earlier run, changing only the vector the trained adapter is fed:
#
#   floor      zero vector. Generated twice, checked identical, copied to every row (minutes)
#   refs-mean  the plain mean of each paper's references — THE GRAPH ARM, no graph training.
#              Compare: step 1 (judge 1.954, 4.2% top-1), retrieval baseline (27.5% top-1)
#   stage-a    head(GNN) from run 8's VICReg encoder, head loaded — does a GNN beat averaging?
#   oracle     the paper's own embedding — a CEILING, it sees the answer. Compare: step 4
#              (judge 2.712, 16.5% top-1). Never report beside the other arms unlabelled.
#
# Ordered by importance, so a timeout loses the least important arm. ~7 h per arm is the old
# estimate; three full arms may not fit 24 h — resubmit, every arm appends only missing rows.
#
# Greedy decoding, bf16, batch 1, the graph-only prompt — all unchanged from every earlier run.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-qwen-gpu.sif
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-run9-decoder
STAGE_A=$BASE_DIR/checkpoints/encoder-gatv2-vicreg.pt   # run 8 stage 10a, whitened features
ARM=run9-decoder

CKPT=$CKPT_DIR/best.pt
[ -f "$CKPT" ] || { echo "no best.pt in $CKPT_DIR — run run_train_decoder.sh first" >&2; exit 1; }
[ -f "$STAGE_A" ] || { echo "missing $STAGE_A (run 8 stage 10a)" >&2; exit 1; }
echo "evaluating $CKPT as $ARM"

mkdir -p $BASE_DIR/logs
export SINGULARITYENV_CKPT="/app${CKPT_DIR#$BASE_DIR}/best.pt"
export SINGULARITYENV_STAGE_A="/app${STAGE_A#$BASE_DIR}"
# The tag names the experiment directories. A resubmit gets a new job id, so to RESUME the same
# directories pass the first job's tag:  RUN_TAG=2026-10-08-14500000 sbatch run_generate_decoder.sh
export SINGULARITYENV_RUN_TAG="${RUN_TAG:-$(date +%F)-${SLURM_JOB_ID:-manual}}"
export SINGULARITYENV_ARM="$ARM"

module load singularity

# NOTE: no apostrophes inside this block — it is one single-quoted argument to bash -lc.
singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app/src/adapter
    nvidia-smi

    python verify_decoder_inputs.py || exit 1
    python verify_generation.py || exit 1
    python verify_graph_only_prompt.py || exit 1

    for INPUT in zero refs-mean stage-a oracle; do
        echo "=== run 9 arm: --input $INPUT ==="
        python generate.py --mode adapter --sets run9 --input "$INPUT" \
            --checkpoint "$CKPT" --stage-a-encoder "$STAGE_A" \
            --output "/app/experiments/${RUN_TAG}_${ARM}-${INPUT}/generations.csv" || exit 1
    done
'
