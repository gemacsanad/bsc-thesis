#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=run11-children
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=72:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/run11_children_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/run11_children_%j.err
#SBATCH --mail-type=END,FAIL

# RUN 11 ON THE CHILDREN'S BOOKS, EVERYTHING IN ONE JOB — run11_all.sh with the dataset swapped
# (DYTAG_DATASET=children, --sets children) and the same recipe. Plan and data:
# src/data/children_books/documentation.md.
#
#   1. CPU verify scripts (seconds)
#   2. phase 2, the GNN grid: the same 8 variants as run 11, predicting a book's own embedding
#      from its masked co-purchase neighbourhood; select_encoder.py copies the best validation
#      cosine to $GNN_DIR/winner.pt
#   3. phase 1, the reader: run 11's recipe (own embedding or, half the time, the neighbour
#      mean; raw features; gate from 0; 6 epochs of 15,000 of the 50,999 training books)
#   4. the evaluation arms on the 945 eval books, MOST IMPORTANT FIRST, so a timeout costs the
#      least needed ones: refs-mean and gnn (the GNN-against-mean comparison, same reader),
#      text (frozen Qwen reading the neighbours as text), zero (floor, minutes), oracle
#      (ceiling)
#   5. phase 3: reader and winning GNN fine-tuned together, 3 epochs of 15,000 books; then the
#      joint arm. Last, because on papers it added nothing measurable over reader <- gnn.
#
# NO LOOP TEST. Run 11 chose --no-repeat-ngram 6 on 200 papers (loop rate 14.0% -> 0%, top-1
# -0.5 points); it is reused, and saves ~6 h at book lengths. $RUN_DIR/ngram records it.
#
# WALLTIME 72 h, and it may not be enough. Estimates: grid ~1 h, reader ~12 h, joint ~6 h. An
# arm took ~7 h on papers at ~150 generated tokens; book descriptions are about twice as long
# (MAX_NEW_TOKENS 1,024), so ~14 h per arm, ~70 h for the five long arms. Expect the joint
# phase to need a resubmission.
#
# RESUBMITTING IS ALWAYS SAFE, AND IS HOW A TIMEOUT IS HANDLED: finished GNN variants are skipped,
# both trainings resume from their newest epoch (or say "nothing to do"), and every evaluation
# appends only its missing rows.
#
# NOTHING IS STOPPED ON PURPOSE. The reader runs with --stop-rules warn: the identification and
# canary checks are logged to $READER_DIR/WARNINGS.txt instead of ending the job (job 14626133
# stopped at epoch 2 on identification 0.062 < 0.078 while the vector lowered validation loss by
# 0.027 nats). A failed evaluation arm is logged and the next arm runs; the job exits 1 at the
# end if any arm failed, and resubmitting re-runs only the missing rows.
#
# Prerequisites: adapter-qwen-gpu.sif; data/children_books/{nodes,edges}.csv and
# node_embeddings.npy; data/frozen/children_books/{training_sets,eval_nodes,prompt_refs_text}.json;
# .env with WANDB_KEY; Qwen/Qwen3-14B in .hf_cache.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-qwen-gpu.sif
RUN_DIR=$BASE_DIR/checkpoints/run11_children
READER_DIR=$RUN_DIR/reader
GNN_DIR=$RUN_DIR/gnn
JOINT_DIR=$RUN_DIR/joint

mkdir -p "$READER_DIR" "$GNN_DIR" "$JOINT_DIR" $BASE_DIR/logs

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

# Fixed on the first submission and reused by every later one, so evaluation always resumes
# into the same experiment directories.
[ -f "$RUN_DIR/eval_tag" ] || echo "$(date +%F)-${SLURM_JOB_ID:-manual}" > "$RUN_DIR/eval_tag"
echo "evaluation tag: $(cat "$RUN_DIR/eval_tag")"
[ -f "$RUN_DIR/ngram" ] || echo 6 > "$RUN_DIR/ngram"

export SINGULARITYENV_DYTAG_DATASET=children
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
    [ "$DYTAG_DATASET" = children ] || { echo "DYTAG_DATASET did not reach the container"; exit 1; }
    EXP=/app/experiments/${RUN_TAG}_run11-children
    GNN=$GNN_DIR/winner.pt
    NGRAM=$(cat $RUN_DIR/ngram)
    echo "every evaluation arm decodes with --no-repeat-ngram $NGRAM"

    # --- 1. CPU checks ----------------------------------------------------------------------
    python verify_decoder_inputs.py || exit 1
    python verify_graph_only_prompt.py || exit 1
    python verify_sliced_loss.py || exit 1
    python verify_lora_prefix.py || exit 1
    python verify_generation.py || exit 1
    for F in /app/data/children_books/nodes.csv /app/data/children_books/edges.csv \
             /app/data/children_books/node_embeddings.npy \
             /app/data/frozen/children_books/training_sets.json \
             /app/data/frozen/children_books/eval_nodes.json \
             /app/data/frozen/children_books/prompt_refs_text.json; do
        [ -f $F ] || { echo "missing $F"; exit 1; }
    done

    # --- 2. phase 2: the GNN grid -------------------------------------------------------------
    # Batch 32, not 256: the books graph has mean degree 30, and the full 2-hop neighbourhood of
    # 256 seeds is ~680k edges (up to ~810k), which ran GATv2 out of memory on the 40 GB card.
    # 32 seeds is ~130k edges (up to ~250k). The whole grid uses it, so the variants stay comparable.
    for CONV in gatv2 sage; do for LAYERS in 1 2; do for LOSS in cos cosnce; do
        NAME=$CONV-l$LAYERS-$LOSS
        [ -f "$GNN_DIR/$NAME.pt" ] && { echo "=== $NAME already trained, skipped ==="; continue; }
        NCE=0; [ $LOSS = cosnce ] && NCE=0.05
        echo "=== run 11 children phase 2: $NAME ==="
        python pretrain_encoder.py --sets children --features raw --gnn-conv $CONV --gnn-layers $LAYERS \
            --batch-size 32 --lambda-cos 1 --lambda-nce $NCE --vicreg-var 0 --vicreg-cov 0 --cosine-gate 1 \
            --out "$GNN_DIR/$NAME.pt" --wandb --wandb-run "run11c-gnn-$NAME-$SLURM_JOB_ID" || exit 1
    done; done; done
    python select_encoder.py "$GNN_DIR" || exit 1
    # The winner fixes the joint encoder shape; train.py checks it against these.
    CONV=$(python -c "import torch; print(torch.load(\"$GNN\", map_location=\"cpu\")[\"gnn_conv\"])")
    LAYERS=$(python -c "import torch; print(torch.load(\"$GNN\", map_location=\"cpu\")[\"gnn_layers\"])")
    echo "winning GNN: $LAYERS-layer $CONV"

    # --- 3. phase 1: the reader ---------------------------------------------------------------
    echo "=== run 11 children phase 1: the reader ==="
    python train.py \
        --oracle \
        --lora \
        --sets children \
        --features raw \
        --refs-mix 0.5 \
        --epoch-size 15000 \
        --epochs 6 \
        --neighbour-tokens 0 \
        --lambda-rank 0 \
        --gate-init 0 \
        --stop-rules warn \
        --out-dir "$READER_DIR" \
        --wandb \
        --wandb-run "run11c-reader-$SLURM_JOB_ID" \
        ${RESUME_READER:-}
    rc=$?
    [ -f "$READER_DIR/WARNINGS.txt" ] && { echo "=== reader stop-rule warnings ==="; cat "$READER_DIR/WARNINGS.txt"; }
    [ $rc = 0 ] || { echo "=== reader training failed (exit $rc): resubmit to resume ==="; exit $rc; }
    [ -f "$READER_DIR/best.pt" ] || { echo "reader training finished without a best.pt"; exit 1; }

    # --- 4. the evaluation arms, most important first -------------------------------------------
    # A failed arm is recorded in FAILED and the job moves on to the next one.
    FAILED=""
    for INPUT in refs-mean stage-a; do
        NAME=$INPUT; [ $INPUT = stage-a ] && NAME=gnn
        echo "=== run 11 children evaluation: reader <- $NAME ==="
        python generate.py --mode adapter --sets children --input "$INPUT" --features raw \
            --checkpoint "$READER_DIR/best.pt" --stage-a-encoder "$GNN" --no-repeat-ngram $NGRAM \
            --output "${EXP}-reader-${NAME}/generations.csv" || FAILED="$FAILED reader-$NAME"
    done

    echo "=== run 11 children evaluation: text arm (frozen Qwen, neighbours as text) ==="
    python generate.py --mode text --sets children --no-repeat-ngram $NGRAM \
        --output "${EXP}-text/generations.csv" || FAILED="$FAILED text"

    echo "=== run 11 children evaluation: reader <- zero (floor) ==="
    python generate.py --mode adapter --sets children --input zero --features raw \
        --checkpoint "$READER_DIR/best.pt" --no-repeat-ngram $NGRAM \
        --output "${EXP}-reader-zero/generations.csv" || FAILED="$FAILED reader-zero"

    echo "=== run 11 children evaluation: reader <- oracle (ceiling) ==="
    python generate.py --mode adapter --sets children --input oracle --features raw \
        --checkpoint "$READER_DIR/best.pt" --no-repeat-ngram $NGRAM \
        --output "${EXP}-reader-oracle/generations.csv" || FAILED="$FAILED reader-oracle"

    # --- 5. phase 3: joint fine-tune, then the joint arm ------------------------------------------
    echo "=== run 11 children phase 3: joint fine-tune ==="
    python train.py \
        --lora \
        --sets children \
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
        --wandb-run "run11c-joint-$SLURM_JOB_ID" \
        ${RESUME_JOINT:-}
    rc=$?
    [ $rc = 0 ] || { echo "=== joint fine-tune failed (exit $rc): resubmit to resume ==="; exit $rc; }
    [ -f "$JOINT_DIR/best.pt" ] || { echo "joint fine-tune finished without a best.pt"; exit 1; }

    echo "=== run 11 children evaluation: joint ==="
    python generate.py --mode adapter --sets children --features raw \
        --checkpoint "$JOINT_DIR/best.pt" --no-repeat-ngram $NGRAM \
        --output "${EXP}-joint/generations.csv" || FAILED="$FAILED joint"
    [ -z "$FAILED" ] || { echo "=== run 11 children: arms FAILED:$FAILED. Resubmit to re-run them ==="; exit 1; }
    echo "=== run 11 children complete: ${EXP}-{reader-refs-mean,reader-gnn,text,reader-zero,reader-oracle,joint} ==="
'
