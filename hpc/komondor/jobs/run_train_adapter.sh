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

# RUN 8 — whitened features, VICReg in stage A, and the instrumentation that would have
# caught the last three failures at epoch 1. Full change spec in docs/run8_plan.md.
#
# What step 7 found, and why the architecture is not what changed here. Step 7 ran this same
# job with the ranking loss and a pretrained encoder and produced the most collapsed
# generations in the project: 34 distinct abstracts out of 945, 890 of them opening with the
# same eight words, top-1 identifiability 0.11% against a chance of 0.106%. The gate=0 control
# emitted ONE string for all 945 nodes and scored the same 0.11%, and the adapter arm was
# word-identical to it for a median of 93 words. Meanwhile prefix/graph_fraction ran at 1.8
# rising to 7.0 — the graph supplying several times the base prefix norm. The channel was
# loud and carried nothing.
#
# The cause is the input representation, not the encoder, the loss or the adapter. The pooled
# reference vectors sit at pairwise cosine 0.6602 with 0.8127 cosine to the global mean, so two
# thirds of their energy is one direction shared by every node; PrefixAdapter.trunk ends in a
# LayerNorm, which normalises per sample across features and cannot remove a component shared
# BETWEEN nodes. A near-constant input gives a near-constant prefix, and greedy decoding emits
# one string. This is anisotropy plus dimensional collapse, both named and both with published
# remedies — see docs/step8_representation_plan.md for the four converging measurements.
#
# Six changes, in three groups. The first two are the hypothesis:
#
#   whitened features  whiten_features.py ZCA-whitens node_embeddings.npy before anything reads
#                      it, and refuses to write if pooled-reference retrieval on the eval set
#                      does not clear 75%. Measured 64.34% -> 78.52%. It runs in THIS job,
#                      before the GPU work, so a failed gate costs minutes instead of a day.
#   VICReg in stage A  InfoNCE is invariant to anisotropy — normalise then divide by tau=0.05
#                      and it can rank perfectly on a collapsed representation, which is how
#                      stage 10a scored 67% on a vector the adapter reads as a constant. The
#                      variance hinge and the covariance term are what it was missing, and
#                      stage A now also refuses an epoch whose output cosine exceeds 0.40.
#
# Two are objective shaping, and they are the two to ablate if the result is ambiguous:
#
#   lambda warmup      --rank-warmup 484, one epoch. L_LM and L_rank genuinely conflict and
#                      step 7 ran at 1.0 from step 0.
#   random window      --rank-offset random. NOT a bug fix: the golds are 98.6% distinct by
#                      word 8, so the opening was never formulaic. The opening states the
#                      problem and the middle states the method, and methodology is the goal.
#
# Two are corrections that cannot produce a false positive:
#
#   rank_acc fixed     NegativePool now covers train + valid. It was training-only, so no
#                      validation node was ever scored, rank_acc was never computed, and
#                      selection silently fell back to LM validation loss — the criterion
#                      step 6 proved is anti-correlated with conditioning. best.pt was being
#                      chosen by exactly the wrong number.
#   cleaned targets    text.clean_abstract strips LaTeX and maths delimiters from the training
#                      targets. The system prompt instructs "no markup, mathematical notation
#                      or headings" while 13.5% of the targets contained exactly that.
#
# And the instrumentation, which is not optional after three silent collapses: a generation
# canary (16 validation nodes, greedy, count distinct) at every validation, and
# prefix/graph_contrast beside graph_fraction — amplitude is not contrast, and no logged number
# was measuring contrast.
#
# STAGE 10b — the contrastive ranking loss, a pretrained encoder, and the gates opened.
#
# Trains the graph encoder and prefix adapter against Qwen3-14B, soft-prompt-only: the prompt
# is one fixed system message and user turn with no references, so the graph is the only
# channel to the paper (src/adapter/text.py).
# See docs/soft_prompt_only.md, step 7, and docs/step6_failure_analysis.md.
#
# What step 6 found. It changed three things at once (per-neighbour tokens, GATv2, LoRA r=64)
# and produced the worst graph run of the project alongside the LOWEST TRAINING LOSS EVER
# RECORDED here — 1.83, below the oracle and below the text runs. Judge accuracy 1.490 ->
# 1.314, distinct abstracts 453 -> 342 of 945, largest identical cluster 61 -> 165 rows,
# top-1 identifiability 1.16% -> 0.32% against a chance of 0.106%.
#
# Those two facts are not in tension. Cross-entropy over ~230 abstract tokens is dominated by
# generic scientific English, so a model that writes fluent boilerplate scores better than one
# that gets the paper right on a handful of content words. Nothing in that objective ever asked
# the prefix to be about THIS paper. Meanwhile a plain untrained mean of a node references
# already retrieves that node at 64.34% top-1, so information was never the bottleneck.
#
# Four changes against step 6, and every one of them targets that:
#
#   ranking loss       L = L_LM + lambda * L_rank, where L_rank asks whether the right abstract
#                      outscores M wrong ones under this node own prefix (src/adapter/rank.py).
#                      It is the identifiability metric turned into a loss. No voyage embedding
#                      appears in it, so the only ceiling is the LLM own. Half the negatives
#                      come from the node OWN references: a prefix carrying only the topic
#                      would rank those as highly as the target, so they are what force the
#                      topic-to-identity distinction every run so far has failed.
#   pretrained encoder stage 10a runs INSIDE THIS JOB, before training. The encoder arrives
#                      already able to identify the target instead of having to discover that
#                      through forty layers and a loss that never asked for it.
#   gates opened       --gate-init 0.2. gate=0 made ignoring the graph both the default and the
#                      starting point; the ablation is preserved by zeroing at EVAL time.
#   LoRA r=16          decoder capacity is anti-correlated with graph use whenever the signal
#                      is weak: frozen 2.86% top-1, r=16 1.16%, r=64 0.32%, monotonic over
#                      three runs. Rank returns only once the channel carries something.
#
# And model selection changed, which is not a hyperparameter but a correction. best.pt used to
# be the epoch with the lowest LM validation loss — the metric step 6 proved is anti-correlated
# with quality. It is now the epoch with the highest validation RANKING ACCURACY.
#
# TWO STAGES, ONE JOB. Stage 10a (pretrain_encoder.py) is InfoNCE on the graph encoder with no
# LLM loaded: it takes minutes, not hours. It runs here rather than as its own submission so
# that one queue wait covers both — a separate job would spend longer waiting for a GPU than
# using one. It is skipped when checkpoints/encoder-gatv2-vicreg.pt already exists, so a resubmit
# after a timeout goes straight to training and never overwrites the encoder that the
# partially-trained checkpoint was initialised from.
#
# Both convolutions are pretrained, which finally settles a question step 6 could not: that run
# changed GATv2 alongside two other things and could attribute nothing to it. Here it costs
# minutes and each has its own metric. Only the gatv2 encoder is used for training below.
#
# THE NUMBER TO BEAT IN STAGE 10a IS 64.34%, the top-1 retrieval of a plain untrained mean of a
# node references (src/analysis/refsignal.py). An encoder below 40% is destroying signal rather
# than extracting it; pretrain_encoder.py prints a warning and that is the finding.
#
# NEIGHBOUR TOKENS ARE OFF in this run (--neighbour-tokens 0). They are not abandoned: the
# ceiling is raised to 128 in config.py so that when they return, every reference gets a token
# rather than the 64 nearest the centroid, which were the ones most redundant with the pooled
# prefix. They are held back because this run already changes four things and step 6 is the
# standing lesson about that, and because in step 6 the two channels co-adapted into mutual
# noise cancellation: emptying the neighbour slots while keeping the pooled prefix scored
# WORSE than disconnecting the graph entirely. The pooled mean carries 64.34% on its own, so
# this is the cleanest test of whether fixing the objective fixes the collapse. Run 2 adds them.
#
# Qwen3-14B's own weights are never written; only checkpoints/epoch*.pt are written. They are
# smaller than step 6's: LoRA drops back from ~42M parameters at r=64 to ~10.5M at r=16, and
# the neighbour-token adapter is absent. Budget ~600 MB per epoch.
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
# Walltime: the ranking loss costs (M+1) x T = 9 x 32 = 288 tokens of forward and backward on
# top of a ~471-token LM step, so roughly +60%. Against that, the neighbour tokens are gone
# (-52 tokens per node on average) and LoRA is a quarter of the size in the backward. Net
# expectation is modestly slower than step 5 and faster than step 6, but this is arithmetic
# and has never been timed on an A100 — 48 h absorbs a miss and the resume below covers a
# timeout. Lower it once one epoch has been timed.
#
# Candidates are scored directly behind the KV prefix rather than behind the 241-token prompt.
# Scoring behind the prompt would mean replaying those 241 tokens once per candidate, which is
# +460% rather than +60%, and the obvious fix — cache the prompt keys and values once — is not
# available: gradient checkpointing forces use_cache=False, which is the same constraint
# PrefixCache exists to work around. --rank-context prompt runs the expensive variant for
# anyone who wants to check the cheap one ranks the same way. See src/adapter/rank.py.
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
# rank-gatv2-lora16: its own directory, and NOT step 5's or step 6's
# checkpoints/qwen14b-graphonly-kv192-lora. train.py refuses to resume across a change of
# --gnn-conv or --neighbour-tokens, so the mistake aborts rather than trains — but best.pt is
# a fixed name in every one of these directories, so pointing this at step 5's would overwrite
# the arm this run is measured against. STEP 2's frozen-LLM run is the bar this one has to
# clear, and step 5's and step 6's are what it has to explain; leave all three intact.
# Same argument against checkpoints/qwen14b-oracle-kv192-lora, which holds the ceiling run.
# graphonly: separate from checkpoints/qwen14b/, which holds the text-path run. train.py refuses
# to resume across a prompt change, but the fixed best.pt name would still be overwritten.
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b-run8-whitened
LORA=1                                  # rank 16 on q/k/v, per config.py. k_proj stays in the
                                        # target list although this run has no neighbour
                                        # tokens: it is harmless with none, and leaving it
                                        # means run 2 changes only --neighbour-tokens
# A new name, not a reuse: checkpoints/encoder-gatv2.pt was pretrained on the RAW features and
# without the VICReg terms, and loading it here would silently start run 8 from step 7's
# collapsed encoder. The two files have identical shapes, so nothing else would catch it.
# The -vicreg suffix must match the --out the stage 10a loop below writes. Job 14473284 died
# at stage 10b because it did not: the loop wrote encoder-gatv2.pt and train.py opened
# encoder-gatv2-vicreg.pt. Changing one of these two lines means changing both.
ENCODER=$BASE_DIR/checkpoints/encoder-gatv2-vicreg.pt   # written by stage 10a in this same job

mkdir -p "$CKPT_DIR" $BASE_DIR/logs

# LoRA is recorded in every checkpoint and train.py refuses to resume across a change of it,
# so a LoRA run needs its own CKPT_DIR — the resume below picks up whatever epoch file is
# newest, which would otherwise be another run's.
[ "$LORA" = 1 ] && export SINGULARITYENV_LORA="--lora"

# Built by stage 10a inside the job below when it is missing. Exported as the /app path,
# which is what both the existence check and train.py see inside the container.
export SINGULARITYENV_ENCODER="/app${ENCODER#$BASE_DIR}"

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
    # one-pass forward training uses and the two-pass prefill generation uses. Kept although
    # this run sets --neighbour-tokens 0, because run 2 turns them back on and a break would
    # otherwise surface there instead of here.
    # NOTE: no apostrophes inside this block — it is one single-quoted argument to
    # bash -lc, and an apostrophe silently ends the quote and splits the script.
    python verify_neighbour_tokens.py || exit 1

    # And the new loss. A ranking loss that scores the wrong rows, averages over padding or
    # never reaches the encoder produces a falling number and a plausible curve — it would
    # look exactly like the run working. Seconds on CPU.
    python verify_rank.py || exit 1

    # --- run 8 gate A: whiten the node features ------------------------------------------
    # CPU only, a few minutes, and it exits non-zero if pooled-reference retrieval on the eval
    # set does not clear config.WHITEN_GATE. That check is the cheap half of the run 8
    # hypothesis: failing it here costs minutes, discovering it after stage B costs a day of
    # A100 time and another evaluation job. Skipped when the file already exists so a resubmit
    # after a timeout does not rebuild 1.5 GB — and, more importantly, cannot change the
    # features underneath a partially trained checkpoint.
    if [ -f /app/data/processed/joint/node_embeddings_whitened.npy ]; then
        echo "=== whitening skipped: node_embeddings_whitened.npy already exists ==="
    else
        echo "=== run 8 gate A: ZCA-whitening the node features ==="
        python whiten_features.py || exit 1
    fi

    # --- stage 10a: pretrain the encoder, on the GPU this job already holds ----------------
    # Minutes, no LLM. Skipped when the encoder exists so a resubmit after a timeout does not
    # redo it — and, more importantly, does not overwrite the encoder that a partially trained
    # checkpoint was initialised from.
    if [ -f "$ENCODER" ]; then
        echo "=== stage 10a skipped: $ENCODER already exists ==="
    else
        for CONV in gatv2 sage; do
            echo "=== stage 10a: pretraining the $CONV encoder (InfoNCE, no LLM) ==="
            python pretrain_encoder.py \
                --gnn-layers 2 \
                --gnn-conv "$CONV" \
                --out "/app/checkpoints/encoder-$CONV-vicreg.pt" \
                --wandb \
                --wandb-run "pretrain-$CONV-$SLURM_JOB_ID" || exit 1
        done
    fi

    echo "=== stage 10b: training the adapter against Qwen3-14B ==="
    python train.py \
        --gnn-layers 2 \
        --gnn-conv gatv2 \
        --neighbour-tokens 0 \
        --pretrained-encoder "$ENCODER" \
        --gate-init 0.2 \
        --lambda-rank 1.0 \
        --rank-negatives 8 \
        --rank-tokens 32 \
        --rank-context prefix \
        --rank-warmup 484 \
        --rank-offset random \
        --out-dir "$CKPT_DIR" \
        --wandb \
        --wandb-run "run8-whitened-$SLURM_JOB_ID" \
        ${LORA:-} ${RESUME:-}
'
