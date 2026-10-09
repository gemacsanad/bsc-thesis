#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=run9-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=72:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/run9_all_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/run9_all_%j.err
#SBATCH --mail-type=END,FAIL

# RUN 9, TRAINING AND EVALUATION IN ONE JOB — one queue wait instead of two.
# Same commands as run_train_decoder.sh followed by run_generate_decoder.sh, which stay as the
# separate versions. Plan and reasoning: docs/run9_embedding_decoder.md.
#
#   1. CPU verify scripts (seconds)
#   2. train the embedding -> abstract decoder (train.py --oracle --oracle-noise 0.3, 6 epochs
#      of 15,000 papers, early stopping on validation identifiability)
#   3. evaluate best.pt four ways on the 945 eval papers: zero (floor, minutes), refs-mean
#      (THE GRAPH ARM), stage-a (GNN + head), oracle (a ceiling)
#
# WALLTIME 72 h. Estimate, not a measurement: training ~45M tokens (6 x 15k papers x ~500)
# at ~600 tok/s is ~20 h, plus ~4 h of validation and identifiability passes; evaluation is
# ~7 h per arm x 3, plus minutes for the floor. ~45 h total, so 48 h had no margin. If the gpu
# partition caps lower, sbatch rejects the request at submit time (check: sinfo -p gpu -o %l).
#
# RESUBMITTING IS ALWAYS SAFE, AND IS HOW A TIMEOUT IS HANDLED. Just run `sbatch` on this file
# again:
#   - training resumes from the newest epoch checkpoint; once it has finished (6 epochs or
#     early stopping), train.py says "nothing to do" in seconds and the job moves on;
#   - evaluation writes into directories named by a tag saved on its first start
#     ($CKPT_DIR/eval_tag), so every later submission appends the missing rows to the SAME
#     experiments/ folders instead of starting over.
#
# THE ONE CASE NOT TO RESUBMIT: a stop rule fired (train.py exit code 3 — the canary collapsed
# or the decoder cannot read its own embedding). This job then writes $CKPT_DIR/STOPPED and
# skips evaluation, and every later submission refuses to start until that file is removed.
#
# Prerequisites:
#   - adapter-qwen-gpu.sif REBUILT from the current .def (it needs scikit-learn)
#   - data/frozen/training_sets_run9.json
#   - data/processed/joint/node_embeddings_whitened.npy and checkpoints/encoder-gatv2-vicreg.pt
#     (both already on Komondor from run 8)
#   - .env with WANDB_KEY; Qwen/Qwen3-14B in $BASE_DIR/.hf_cache

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-qwen-gpu.sif
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-run9-decoder
STAGE_A=$BASE_DIR/checkpoints/encoder-gatv2-vicreg.pt   # run 8 stage 10a, whitened features
ARM=run9-decoder

mkdir -p "$CKPT_DIR" $BASE_DIR/logs

if [ -f "$CKPT_DIR/STOPPED" ]; then
    echo "REFUSING: $CKPT_DIR/STOPPED exists — a run-9 stop rule fired in an earlier job." >&2
    echo "Read that job's log first. Delete the file only to deliberately continue the run." >&2
    exit 1
fi
[ -f "$STAGE_A" ] || { echo "missing $STAGE_A (run 8 stage 10a)" >&2; exit 1; }

LATEST=$(ls -1 "$CKPT_DIR"/epoch*.pt 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    export SINGULARITYENV_RESUME="--resume /app${CKPT_DIR#$BASE_DIR}/$(basename "$LATEST")"
    echo "training resumes from $LATEST"
else
    echo "no checkpoint found — training starts from scratch"
fi

# Fixed on the first submission and reused by every later one, so evaluation always resumes
# into the same experiment directories.
[ -f "$CKPT_DIR/eval_tag" ] || echo "$(date +%F)-${SLURM_JOB_ID:-manual}" > "$CKPT_DIR/eval_tag"
echo "evaluation tag: $(cat "$CKPT_DIR/eval_tag")"

export SINGULARITYENV_CKPT_DIR="/app${CKPT_DIR#$BASE_DIR}"
export SINGULARITYENV_STAGE_A="/app${STAGE_A#$BASE_DIR}"
export SINGULARITYENV_RUN_TAG="$(cat "$CKPT_DIR/eval_tag")"
export SINGULARITYENV_ARM="$ARM"

module load singularity

# NOTE: no apostrophes inside this block — it is one single-quoted argument to bash -lc.
singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app/src/adapter
    nvidia-smi

    # --- 1. CPU checks, seconds each, before any GPU hours --------------------------------
    python verify_decoder_inputs.py || exit 1
    python verify_graph_only_prompt.py || exit 1
    python verify_sliced_loss.py || exit 1
    python verify_lora_prefix.py || exit 1
    python verify_generation.py || exit 1
    [ -f /app/data/frozen/training_sets_run9.json ] || { echo "missing training_sets_run9.json"; exit 1; }
    [ -f /app/data/processed/joint/node_embeddings_whitened.npy ] || { echo "missing whitened features"; exit 1; }

    # --- 2. training ----------------------------------------------------------------------
    echo "=== run 9 training: the embedding -> abstract decoder ==="
    python train.py \
        --oracle \
        --lora \
        --sets run9 \
        --oracle-noise 0.3 \
        --epoch-size 15000 \
        --epochs 6 \
        --neighbour-tokens 0 \
        --lambda-rank 0 \
        --gate-init 0.2 \
        --out-dir "$CKPT_DIR" \
        --wandb \
        --wandb-run "run9-decoder-$SLURM_JOB_ID" \
        ${RESUME:-}
    rc=$?
    if [ $rc = 3 ]; then
        touch "$CKPT_DIR/STOPPED"
        echo "=== a stop rule fired: evaluation skipped, $CKPT_DIR/STOPPED written ==="
        exit 3
    fi
    [ $rc = 0 ] || { echo "=== training failed (exit $rc): resubmit to resume ==="; exit $rc; }
    [ -f "$CKPT_DIR/best.pt" ] || { echo "training finished without a best.pt"; exit 1; }

    # --- 3. evaluation, most important arm first after the minutes-long floor ---------------
    for INPUT in zero refs-mean stage-a oracle; do
        echo "=== run 9 evaluation arm: --input $INPUT ==="
        python generate.py --mode adapter --sets run9 --input "$INPUT" \
            --checkpoint "$CKPT_DIR/best.pt" --stage-a-encoder "$STAGE_A" \
            --output "/app/experiments/${RUN_TAG}_${ARM}-${INPUT}/generations.csv" || exit 1
    done
    echo "=== run 9 complete: experiments/${RUN_TAG}_${ARM}-{zero,refs-mean,stage-a,oracle} ==="
'
