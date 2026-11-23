#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=run11-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=72:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/run11_all_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/run11_all_%j.err
#SBATCH --mail-type=END,FAIL

# Run 11: the whole final pipeline in one job.
#   1. CPU checks
#   2. GNN grid: {GATv2, GraphSAGE} x {1, 2 layers} x {cosine, cosine + InfoNCE};
#      select_encoder.py keeps the best validation cosine as $GNN_DIR/winner.pt
#   3. the reader (own embedding, or the reference mean half of the time)
#   4. decoding test with and without --no-repeat-ngram 6 (loop_check.py); the choice is used below
#   5. the reader on the 945 evaluation papers: zero, GNN estimate, reference mean
#   6. joint fine-tuning of the reader and the selected GNN
#   7. the joint model on the 945 evaluation papers
#   8. the reader from the paper's own embedding (oracle)
#
# Resubmitting resumes: finished steps are skipped and evaluation appends only missing rows.
# If a reader stop rule fired (exit 3), $READER_DIR/STOPPED blocks further submissions.
#
# Needs: adapter-qwen-gpu.sif, data/frozen/training_sets_run9.json,
# data/processed/joint/node_embeddings.npy, .env with WANDB_KEY, Qwen/Qwen3-14B in .hf_cache.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-qwen-gpu.sif
RUN_DIR=$BASE_DIR/checkpoints/run11
READER_DIR=$RUN_DIR/reader
GNN_DIR=$RUN_DIR/gnn
JOINT_DIR=$RUN_DIR/joint

mkdir -p "$READER_DIR" "$GNN_DIR" "$JOINT_DIR" $BASE_DIR/logs

if [ -f "$READER_DIR/STOPPED" ]; then
    echo "REFUSING: $READER_DIR/STOPPED exists — a reader stop rule fired in an earlier job." >&2
    echo "Read that job's log first. Delete the file only to deliberately continue the run." >&2
    exit 1
fi

LATEST=$(ls -1 "$READER_DIR"/epoch*.pt 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    export SINGULARITYENV_RESUME_READER="--resume /app${READER_DIR#$BASE_DIR}/$(basename "$LATEST")"
    echo "reader training resumes from $LATEST"
fi
LATEST=$(ls -1 "$JOINT_DIR"/epoch*.pt 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    export SINGULARITYENV_RESUME_JOINT="--resume /app${JOINT_DIR#$BASE_DIR}/$(basename "$LATEST")"
    echo "joint fine-tune resumes from $LATEST"
fi

# Fixed on the first submission, so evaluation resumes into the same experiment directories.
[ -f "$RUN_DIR/eval_tag" ] || echo "$(date +%F)-${SLURM_JOB_ID:-manual}" > "$RUN_DIR/eval_tag"
echo "evaluation tag: $(cat "$RUN_DIR/eval_tag")"

export SINGULARITYENV_RUN_DIR="/app${RUN_DIR#$BASE_DIR}"
export SINGULARITYENV_READER_DIR="/app${READER_DIR#$BASE_DIR}"
export SINGULARITYENV_GNN_DIR="/app${GNN_DIR#$BASE_DIR}"
export SINGULARITYENV_JOINT_DIR="/app${JOINT_DIR#$BASE_DIR}"
export SINGULARITYENV_RUN_TAG="$(cat "$RUN_DIR/eval_tag")"

module load singularity

# NOTE: no apostrophes inside this block — it is one single-quoted argument to bash -lc.
singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app/src/adapter
    nvidia-smi
    EXP=/app/experiments/${RUN_TAG}_run11
    GNN=$GNN_DIR/winner.pt

    # --- 1. CPU checks ----------------------------------------------------------------------
    python verify_decoder_inputs.py || exit 1
    python verify_graph_only_prompt.py || exit 1
    python verify_sliced_loss.py || exit 1
    python verify_lora_prefix.py || exit 1
    python verify_generation.py || exit 1
    [ -f /app/data/frozen/training_sets_run9.json ] || { echo "missing training_sets_run9.json"; exit 1; }
    [ -f /app/data/processed/joint/node_embeddings.npy ] || { echo "missing raw features"; exit 1; }

    # --- 2. phase 2: the GNN grid -------------------------------------------------------------
    # InfoNCE is weighted by tau (0.05) to put it on the scale of the cosine loss.
    for CONV in gatv2 sage; do for LAYERS in 1 2; do for LOSS in cos cosnce; do
        NAME=$CONV-l$LAYERS-$LOSS
        [ -f "$GNN_DIR/$NAME.pt" ] && { echo "=== $NAME already trained, skipped ==="; continue; }
        NCE=0; [ $LOSS = cosnce ] && NCE=0.05
        echo "=== run 11 phase 2: $NAME ==="
        python pretrain_encoder.py --sets run9 --features raw --gnn-conv $CONV --gnn-layers $LAYERS \
            --lambda-cos 1 --lambda-nce $NCE --vicreg-var 0 --vicreg-cov 0 --cosine-gate 1 \
            --out "$GNN_DIR/$NAME.pt" --wandb --wandb-run "run11-gnn-$NAME-$SLURM_JOB_ID" || exit 1
    done; done; done
    python select_encoder.py "$GNN_DIR" || exit 1
    # The winner fixes the joint encoder shape; train.py checks it against these.
    CONV=$(python -c "import torch; print(torch.load(\"$GNN\", map_location=\"cpu\")[\"gnn_conv\"])")
    LAYERS=$(python -c "import torch; print(torch.load(\"$GNN\", map_location=\"cpu\")[\"gnn_layers\"])")
    echo "winning GNN: $LAYERS-layer $CONV"

    # --- 3. phase 1: the reader ---------------------------------------------------------------
    echo "=== run 11 phase 1: the reader ==="
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
        --out-dir "$READER_DIR" \
        --wandb \
        --wandb-run "run11-reader-$SLURM_JOB_ID" \
        ${RESUME_READER:-}
    rc=$?
    if [ $rc = 3 ]; then
        touch "$READER_DIR/STOPPED"
        echo "=== a stop rule fired: everything after the reader skipped, $READER_DIR/STOPPED written ==="
        exit 3
    fi
    [ $rc = 0 ] || { echo "=== reader training failed (exit $rc): resubmit to resume ==="; exit $rc; }
    [ -f "$READER_DIR/best.pt" ] || { echo "reader training finished without a best.pt"; exit 1; }

    # --- 4. the loop-fix test -----------------------------------------------------------------
    if [ ! -f "$RUN_DIR/ngram" ]; then
        for N in 0 6; do
            echo "=== run 11 loop test: --no-repeat-ngram $N ==="
            python generate.py --mode adapter --sets run9 --input oracle --features raw \
                --checkpoint "$READER_DIR/best.pt" --subset 200 --no-repeat-ngram $N \
                --output "${EXP}-looptest-ngram${N}/generations.csv" || exit 1
        done
        python loop_check.py --choose 6 "${EXP}-looptest-ngram0/generations.csv" \
            "${EXP}-looptest-ngram6/generations.csv" > /tmp/loop_choice || exit 1
        cat /tmp/loop_choice
        tail -1 /tmp/loop_choice > "$RUN_DIR/ngram"
    fi
    NGRAM=$(cat "$RUN_DIR/ngram")
    echo "every evaluation arm decodes with --no-repeat-ngram $NGRAM"

    # --- 5. the phase-1 reader, separately trained inputs --------------------------------------
    for INPUT in zero stage-a refs-mean; do
        NAME=$INPUT; [ $INPUT = stage-a ] && NAME=gnn
        echo "=== run 11 evaluation: reader <- $NAME ==="
        python generate.py --mode adapter --sets run9 --input "$INPUT" --features raw \
            --checkpoint "$READER_DIR/best.pt" --stage-a-encoder "$GNN" --no-repeat-ngram $NGRAM \
            --output "${EXP}-reader-${NAME}/generations.csv" || exit 1
    done

    # --- 6. phase 3: joint fine-tune ------------------------------------------------------------
    echo "=== run 11 phase 3: joint fine-tune ==="
    python train.py \
        --lora \
        --sets run9 \
        --features raw \
        --reader "$READER_DIR/best.pt" \
        --pretrained-encoder "$GNN" \
        --gnn-conv "$CONV" \
        --gnn-layers "$LAYERS" \
        --epoch-size 15000 \
        --epochs 3 \
        --neighbour-tokens 0 \
        --lambda-rank 0 \
        --gate-init 0 \
        --out-dir "$JOINT_DIR" \
        --wandb \
        --wandb-run "run11-joint-$SLURM_JOB_ID" \
        ${RESUME_JOINT:-}
    rc=$?
    [ $rc = 0 ] || { echo "=== joint fine-tune failed (exit $rc): resubmit to resume ==="; exit $rc; }
    [ -f "$JOINT_DIR/best.pt" ] || { echo "joint fine-tune finished without a best.pt"; exit 1; }

    # --- 7. the joint model ---------------------------------------------------------------------
    echo "=== run 11 evaluation: joint ==="
    python generate.py --mode adapter --sets run9 --features raw \
        --checkpoint "$JOINT_DIR/best.pt" --no-repeat-ngram $NGRAM \
        --output "${EXP}-joint/generations.csv" || exit 1

    # --- 8. the ceiling -------------------------------------------------------------------------
    echo "=== run 11 evaluation: reader <- oracle (ceiling) ==="
    python generate.py --mode adapter --sets run9 --input oracle --features raw \
        --checkpoint "$READER_DIR/best.pt" --no-repeat-ngram $NGRAM \
        --output "${EXP}-reader-oracle/generations.csv" || exit 1
    echo "=== run 11 complete: ${EXP}-{reader-zero,reader-gnn,reader-refs-mean,joint,reader-oracle} ==="
'
