#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=run10a-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=48:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/run10a_all_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/run10a_all_%j.err
#SBATCH --mail-type=END,FAIL

# RUN 10a, TRAINING AND EVALUATION IN ONE JOB. Same structure as run_decoder_all.sh (run 9),
# which stays as the record of that run. Plan and reasoning: docs/run9_embedding_decoder.md,
# section "Run 10a".
#
#   1. CPU verify scripts (seconds)
#   2. train the embedding -> abstract decoder on step 4's recipe: raw features, gate from 0,
#      no synthetic noise; each training paper is fed its own embedding or, half the time, the
#      mean of its references (--refs-mix 0.5). 6 epochs of 15,000 papers, stop rules from
#      epoch 2, selection on validation reference-mean identifiability
#   3. evaluate best.pt three ways on the 945 eval papers: zero (floor, minutes), refs-mean
#      (THE GRAPH ARM), oracle (a ceiling). No stage-a arm: run 8's encoder reads whitened
#      features
#
# WALLTIME 48 h. Measured on run 9: one 15k-paper epoch took 1.91 h including its validation
# passes, so training is ~12 h (a little more for loading each paper's references). Evaluation
# is ~7 h per arm x 2, plus minutes for the floor. ~26 h total.
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
#   - data/processed/joint/node_embeddings.npy (the raw features, already on Komondor)
#   - .env with WANDB_KEY; Qwen/Qwen3-14B in $BASE_DIR/.hf_cache

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-qwen-gpu.sif
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-run10a-decoder
ARM=run10a-decoder

mkdir -p "$CKPT_DIR" $BASE_DIR/logs

if [ -f "$CKPT_DIR/STOPPED" ]; then
    echo "REFUSING: $CKPT_DIR/STOPPED exists — a run-10a stop rule fired in an earlier job." >&2
    echo "Read that job's log first. Delete the file only to deliberately continue the run." >&2
    exit 1
fi

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
    [ -f /app/data/processed/joint/node_embeddings.npy ] || { echo "missing raw features"; exit 1; }

    # --- 2. training ----------------------------------------------------------------------
    echo "=== run 10a training: the embedding -> abstract decoder, step 4 recipe ==="
    python train.py \
        --oracle \
        --lora \
        --sets run9 \
        --features raw \
        --refs-mix 0.5 \
        --epoch-size 15000 \
        --epochs 6 \
        --neighbour-tokens 0 \
        --lambda-rank 0 \
        --gate-init 0 \
        --out-dir "$CKPT_DIR" \
        --wandb \
        --wandb-run "run10a-decoder-$SLURM_JOB_ID" \
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
    for INPUT in zero refs-mean oracle; do
        echo "=== run 10a evaluation arm: --input $INPUT ==="
        python generate.py --mode adapter --sets run9 --input "$INPUT" --features raw \
            --checkpoint "$CKPT_DIR/best.pt" \
            --output "/app/experiments/${RUN_TAG}_${ARM}-${INPUT}/generations.csv" || exit 1
    done
    echo "=== run 10a complete: experiments/${RUN_TAG}_${ARM}-{zero,refs-mean,oracle} ==="
'
