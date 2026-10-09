#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=decoder-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=48:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/train_decoder_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/train_decoder_%j.err
#SBATCH --mail-type=END,FAIL

# RUN 9 — the embedding -> abstract decoder. Plan and reasoning: docs/run9_embedding_decoder.md.
#
# Trains the step-4 configuration (kv192 prefix adapter, LoRA r16 on q/v, the graph-only
# prompt) to write a paper's abstract from that paper's OWN whitened embedding, rotated to a
# random cosine in [0.3, 1] with the truth so it also learns to write from an estimate. Nothing
# about the graph is trained here: the graph arm is joined at evaluation
# (run_generate_decoder.sh), by feeding the same decoder the mean of a paper's references.
#
# Against step 4, three things move and they are one hypothesis — that the decoder failed for
# lack of variety, not of training: 90,461 train-split papers instead of 7,750 (each seen about
# once, instead of 7,750 seen ten times), the noise, and selection on generated text. The
# whitened features and cleaned targets came with run 8 and are kept.
#
# Six epochs of 15,000 papers is one pass over the training set, ~90k sample-passes against
# step 4's 77,500. If val_id/refs_mrr is still rising at epoch 6, resubmit with --epochs raised:
# the resume continues the same permutation into its second pass.
#
# The run stops itself (exit non-zero, mail FAIL) when the canary writes fewer than 9 distinct
# texts of 16 at any epoch end, or when from epoch 2 the decoder identifies fewer than 10x
# chance of the validation papers from their own embedding. Do NOT blindly resubmit after
# either: the resume would continue the run that just failed.
#
# Walltime is arithmetic, not a measurement: ~15k x ~500 tokens per epoch at an unmeasured
# tokens/s. W&B train/tokens_per_sec in the first hour sizes it; resubmit on timeout.
#
# Prerequisites:
#   - adapter-qwen-gpu.sif REBUILT from the current .def — run 9 needs scikit-learn (identify.py)
#   - data/frozen/training_sets_run9.json (src/adapter/prepare_run9_sets.py, run locally)
#   - data/processed/joint/node_embeddings_whitened.npy (already on Komondor from run 8)
#   - .env with WANDB_KEY; Qwen/Qwen3-14B in $BASE_DIR/.hf_cache

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-qwen-gpu.sif
# Its own directory: best.pt is a fixed name and the resume below takes the newest epoch file.
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-run9-decoder

mkdir -p "$CKPT_DIR" $BASE_DIR/logs

LATEST=$(ls -1 "$CKPT_DIR"/epoch*.pt 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    export SINGULARITYENV_RESUME="--resume /app${CKPT_DIR#$BASE_DIR}/$(basename "$LATEST")"
    echo "resuming from $LATEST"
else
    echo "no checkpoint found — starting from scratch"
fi
export SINGULARITYENV_CKPT_DIR="/app${CKPT_DIR#$BASE_DIR}"

module load singularity

# NOTE: no apostrophes inside this block — it is one single-quoted argument to bash -lc.
singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app/src/adapter
    nvidia-smi

    # CPU, seconds each, before the GPU hours. The first is new for run 9: the noise model,
    # the reference mean (and that it never reads the paper itself), the epoch slicer, the
    # scorer and the stage-a head. The rest re-prove the prompt, the loss, LoRA and generation.
    python verify_decoder_inputs.py || exit 1
    python verify_graph_only_prompt.py || exit 1
    python verify_sliced_loss.py || exit 1
    python verify_lora_prefix.py || exit 1
    python verify_generation.py || exit 1

    [ -f /app/data/frozen/training_sets_run9.json ] || { echo "missing training_sets_run9.json"; exit 1; }
    [ -f /app/data/processed/joint/node_embeddings_whitened.npy ] || { echo "missing whitened features"; exit 1; }

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
'
