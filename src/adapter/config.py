import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# --- Where things live -------------------------------------------------------
JOINT_DIR = os.path.join(ROOT, "data", "processed", "joint")   # nodes/edges/embeddings
OGBN_DIR = os.path.join(ROOT, "data", "raw", "ogbn_arxiv")     # the original download
FROZEN_DIR = os.path.join(ROOT, "data", "frozen")              # eval_nodes, splits_eligible, training_sets
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")            # one directory per run
ENV_FILE = os.path.join(ROOT, ".env")

# The frozen LLM, everywhere: train.py, generate.py, and the three scripts that only need
# its tokenizer to measure prompt lengths (text.py, prepare_data_capped.py,
# verify_capped_prompts.py). It was written out in all five before.
#
# Stage 7 (2026-09-03): was "lmsys/vicuna-7b-v1.5". The Vicuna runs are reproduced by checking
# out the commit before this one — not by flipping this back, because the architecture
# constants below move with it. Which references a prompt shows is *not* tokenizer-dependent
# any more; that was frozen under the Vicuna tokenizer in data/frozen/prompt_refs_capped.json
# and freeze_prompt_refs.py pins its own tokenizer rather than reading this.
BASE_MODEL = "Qwen/Qwen3-14B"

# --- Stage 1: eligible nodes and the frozen evaluation set -------------------
MIN_OUT_DEGREE = 5
MAX_OUT_DEGREE = 15
EVAL_PER_DEGREE = 45

# --- Stage 1b: the capped-prompt node sets -----------------------------------
# Eligibility is widened to out-degree 5-100; the prompt is capped at 15 references
# by centroid distance (neighbour_cap.py), so degree no longer has to fit the context.
# Quotas are ceilings, not guarantees: above out-degree ~34 the graph simply has fewer
# nodes than that, and those degrees contribute everything they have.
CAPPED_MIN_OUT_DEGREE = 5
CAPPED_MAX_OUT_DEGREE = 100
CAPPED_TRAIN_PER_DEGREE = 200
CAPPED_VALID_PER_DEGREE = 10
CAPPED_EVAL_PER_DEGREE = 10

# --- Stage 2: neighbourhood sampling -----------------------------------------
# Sampling depth is not a free parameter: a SAGEConv layer whose inputs come from an
# unsampled hop aggregates incomplete neighbourhoods and fails silently, so the hop count
# is derived from GNN_LAYERS rather than written down separately. -1 = take every neighbour
# at that hop; measured sizes are in docs/adapter.md.
PROMPT_MAX_REFS = 15        # ceiling on references shown in the prompt; the GNN still sees all
# Tokens held back for the abstract, = generate.py's --max-new-tokens. Raised from 256 on
# 2026-08-19: 317 of the 945 capped eval golds are longer than 256 tokens (median gold is 170
# words, ~230 tokens), so a third of the targets could not be reproduced even in principle, and
# 238 of the adapter's 945 outputs ended mid-sentence. 320 covers the 150-180 words the prompt
# now asks for (~245 tokens) with slack. It is charged against the prompt, so a larger reserve
# means the centroid cap shows *fewer* references, not more.
GENERATION_RESERVE = 320

# --- Stage 3a: graph encoder -------------------------------------------------
IN_CHANNELS = 1024          # voyage-4-large embedding dim
HIDDEN_CHANNELS = 1024
OUT_CHANNELS = 1024
DROPOUT = 0.2
GNN_LAYERS = 2              # default depth; --gnn-layers overrides it per run

# Stage 9 (2026-09-25): was GraphSAGE's mean aggregation, in every run up to step 5.
# GATv2 weights each reference instead of averaging them, which is the point: a paper's
# 52 median references are not equally informative about its own abstract, and mean
# aggregation cannot say so. "sage" reproduces every earlier run and is what --gnn-conv
# takes to do it. Recorded in the checkpoint; train.py refuses to resume across a change,
# because the two build different parameter names and a mismatch would half-load.
GNN_CONV = "gatv2"
GNN_HEADS = 4               # per-layer attention heads; each emits width // heads and they
                            # are concatenated, so the layer width is unchanged at 1024


def num_neighbors(gnn_layers=GNN_LAYERS):
    """Sampling fan-out for a `gnn_layers`-deep encoder: every neighbour at every hop."""
    return [-1] * gnn_layers

# --- Stage 3b: prefix adapter ------------------------------------------------
# Qwen3-14B: 40 layers, 40 query heads served by 8 KV heads (5:1 GQA), head_dim 128.
# Was Vicuna-7B-v1.5 (Llama-2-7b): 32 layers, 32 heads, head_dim 128, no GQA.
#
# NUM_KV_HEADS was called NUM_HEADS, which was a name that lied: it has always been the
# *key/value* head count in the cache layout, and only equalled the query-head count because
# Llama-2 has no GQA. The prefix is a KV tensor, so its width is set by the KV heads.
NUM_LAYERS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128              # read off config.head_dim, which Qwen3 states explicitly; it
                            # equals hidden_size // num_attention_heads here by coincidence
# Was 10. Holding T at 10 would NOT hold the method fixed: GQA narrows the graph's channel
# into the LLM by 3.2x as a side effect, and a fallen gap could not be attributed to the LLM
# rather than the narrower prefix. T = 32 restores the prefix *volume* exactly —
#   Vicuna, T=10:  32 x 2 x 32 x 10 x 128 = 2,621,440 values/node
#   Qwen3,  T=32:  40 x 2 x  8 x 32 x 128 = 2,621,440   (identical)
# What it cannot restore is addressability: 8 KV heads serve 40 query heads, so the prefix can
# no longer say different things to different heads within a group. No workaround exists for a
# GQA model — it is recorded as a limitation, not fixed. See docs/model_swap_qwen.md §3.
# Costs 22 more of the 4,096 context tokens, and only in the adapter arm.
NUM_PREFIX_TOKENS = 32
# Stage 8 step 2 (2026-09-17): were 768 and 96, with one decoder shared by keys and values.
# 1024 means nothing is compressed between the encoder output and the codes (768 kept 99.83% of
# the variance of reference centroids, so this closes a question more than a measured loss);
# 192 doubles the directions each (layer, role) can use. Adapter 12,494,672 -> 32,065,616
# parameters. See docs/soft_prompt_only.md, step 2.
TRUNK_DIM = 1024
CODE_DIM = 192
BASE_INIT_STD = 0.02        # the base model's own initializer_range; 0.02 for both Llama-2
                            # and Qwen3, so this value is unchanged by the swap

# --- Stage 3c: one input token per 1-hop neighbour ---------------------------
# Stage 9 (2026-09-25). Every run up to step 5 compressed the whole neighbourhood into ONE
# 1024-d vector before the LLM saw any of it. Step 5 measured what that costs: with the
# channel demonstrably open (the oracle reached 11.96% top-1 identifiability through the
# very same 32 tokens), the graph arm reached 0.95% and collapsed to 453 distinct abstracts
# out of 945 — the LLM learned to ignore an uninformative prefix and emit a template.
# These tokens remove the compression: each reference arrives as its own token.
#
# They enter as **input embeddings**, not as KV prefix. A KV-prefix token is something
# attention can look at; an input-embedding token is something the 40 layers compute on,
# and comparing references against each other is exactly the operation this run is testing.
# It is also 16x cheaper per token — hidden_size 5,120 against a KV token's
# 40 x 2 x 8 x 128 = 81,920 values.
# Stage 10 (2026-09-28): was 64, chosen to match the text prompt's centroid cap. That cap was
# inherited from an arm where a reference costs ~230 prompt tokens; here it costs ONE, so the
# constraint never applied. Worse, the centroid ranking kept the references *closest* to the
# mean — the ones already carried by the pooled prefix — and dropped the distinctive tail the
# token channel exists to deliver. Measured on the 345 eval nodes where the cap bit: nearest-64
# 66.38% top-1, random-64 69.57%, farthest-64 71.88%, all references 70.43%.
#
# 128 is a non-binding safety bound, not a filter: CAPPED_MAX_OUT_DEGREE caps seed out-degree at
# 100, so every node now contributes all of its references. It cannot simply be deleted —
# generate.py charges it against the token budget and prefix_generation.patch_for_prefix
# range-checks the cache with it — so a ceiling above the data's maximum is the way to say
# "no cap". Consequence to report, not to prevent: k now equals out-degree exactly, so sequence
# length is a perfect readout of degree, on top of the known degree confound.
NUM_NEIGHBOUR_TOKENS = 128  # ceiling only; a node contributes min(out_degree, 128) = its degree
HIDDEN_SIZE = 5120          # Qwen3-14B's hidden_size = 40 query heads x 128. NOT 4096, which
                            # was Vicuna's, and not the k/v width (1024) — these are ordinary
                            # input embeddings, so they are as wide as the residual stream.

# --- Stage 4: training -------------------------------------------------------
# Held at Vicuna's context on purpose. Qwen3-14B natively supports 40,960, but raising it would
# dissolve the experiment: the centroid cap is what gives the GNN information the prompt does
# not have — the GNN reads every reference, the prompt shows ~13. If every reference fit the
# prompt, the graph's unique signal would collapse to 2-hop structure alone. Lifting the context
# is a separate experiment, and is expected to *shrink* the gap. See model_swap_qwen.md §2.
MAX_LENGTH = 4096           # prefix tokens count against it
TRAIN_SUBSET = 4000         # first-run training set, sampled from the eligible train nodes
VALID_PER_DEGREE = 45       # 45 x 11 degrees = 495 validation nodes, as in the eval set
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0          # decaying base/gate would pull the prefix toward the ablation
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
SEED = 42

DEFAULT_SETS = "capped"     # which frozen node sets; --sets legacy reproduces the old runs

# The run schedule, previously literals in hpc/komondor/jobs/run_train_adapter.sh.
EPOCHS = 10                 # upper bound; early stopping usually ends the run sooner
PATIENCE = 2                # epochs without a better validation loss before stopping
BATCH_SIZE = 1              # micro-batch; 1 x 16 keeps the effective batch at 16, which is the
GRAD_ACCUM = 16             # invariant. Was 2 x 8 — halved because Qwen3-14B is 29.6 GB of
                            # bf16 weights on a 40 GB A100, against Vicuna-7B's 13.5 GB.
LOG_EVERY = 10              # optimizer steps
EVAL_EVERY = 250            # optimizer steps; at 100, validation cost more tokens than training

WANDB_PROJECT = "dytag-adapter"

# --- Stage 4b: LoRA on the frozen LLM ----------------------------------------
# Off unless train.py is given --lora, and off for the Stage 7 Qwen run — era 1 had no LoRA.
# Only q_proj and v_proj, on all 40 layers: 80 modules, 10.5M parameters, against the 36.3M of
# the encoder and adapter together (4.2M + 32.1M, measured by `python model.py`).
# Under GQA the two targets are different sizes — q_proj is 5120 x 5120, v_proj only
# 5120 x 1024 — so per layer it is 16 x (5120+5120) + 16 x (5120+1024) = 262,144.
# Was 8.4M on Vicuna: 64 modules of a uniform 4096 x 4096.
# The prefix reaches the model as extra keys and values, so the query and value projections
# are what decide how much of it is read — see lora.py for why the targets stop there.
# Stage 9 (2026-09-25): r was 16, alpha 32, targets q/v only.
#
# r 16 -> 64 quadruples the adapter. It is raised here and not earlier because rank is only
# worth paying for once the prefix carries something to read: step 5 showed that LoRA given
# an uninformative prefix spends its capacity learning a template instead (judge accuracy
# 1.737 frozen -> 1.490 with r=16). The neighbour tokens are what make the rank useful.
#
# k_proj is added, and that reverses the rule stated below for the KV prefix. The reasoning
# there is sound and still holds for the *prefix*: it supplies its keys and values directly,
# so k_proj and v_proj never touch it and only q_proj decides how much is read. A neighbour
# token is the opposite kind of object — an ordinary input embedding whose key is *computed*
# by k_proj at every layer. With k_proj frozen the model has no way to make the new tokens
# attendable, whatever else it learns. This is the projection the new token type depends on,
# not general capacity; o_proj and the MLP stay frozen for exactly the reason given below.
#
# Stage 10 (2026-09-28): r back to 16. Step 6 measured that decoder capacity is anti-correlated
# with graph use whenever the conditioning signal is weak — same encoder, same prefix, same
# prompt: frozen LLM 2.86% top-1 identifiability, LoRA r=16 1.16%, LoRA r=64 0.32%. Monotonic
# over three runs. Rank is not worth paying for until the channel demonstrably carries
# something, and that is what the ranking loss below is for. r=64 returns only after it does.
LORA_R = 16
LORA_ALPHA = 32             # 2 x rank; peft scales the adapter by alpha / r
LORA_DROPOUT = 0.05
#
# Run 9 (2026-10-07): back to q/v, the targets of step 4 — the only run that carried paper
# identity, and the configuration run 9 extends. k_proj was added for the neighbour tokens,
# which run 9 does not have. generate.py reads the targets back off the checkpoint's LoRA keys,
# so the q/k/v checkpoints of steps 6-8 still load.
LORA_TARGETS = ["q_proj", "v_proj"]

# --- Stage 10: contrastive supervision ---------------------------------------
# Step 6 posted the lowest training loss of any run in the project (1.83, below the oracle and
# below the text runs) together with its worst conditioning. That is not a contradiction: the
# LM objective is cross-entropy over ~230 abstract tokens, which is dominated by generic
# scientific English. Getting the paper right buys a few nats on a handful of content words;
# fluent boilerplate buys more on every token. Nothing in the loss ever asked the prefix to be
# about *this* paper. These constants add an objective that asks for nothing else.

# Stage A — encoder pretraining against the node's own voyage embedding (pretrain_encoder.py).
# Deliberately Voyage-capped and deliberately temporary: it is an initialisation, never a
# result. Justified by measurement rather than assumption — a plain untrained mean of a node's
# reference embeddings already retrieves that node's own embedding at 64.34% top-1 out of 945,
# and the pipeline currently delivers 0.32%, so the ceiling is ~50x above where we are.
PRETRAIN_TAU = 0.05         # InfoNCE temperature
PRETRAIN_LR = 1e-3          # no LLM in the loop, so this can be far above LEARNING_RATE
PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH = 256        # anchors per step; the 14B model is absent, so batches are free
PRETRAIN_NEGATIVES = 4096   # sampled candidates on top of the in-batch positives

# Stage B — the ranking loss carried into LM training (rank.py). No Voyage anywhere: the
# candidates are gold abstracts from the training set, so the ceiling is the LLM's own.
RANK_NEGATIVES = 8          # M wrong abstracts scored against each node's own prefix
RANK_TOKENS = 32            # T. Identity is concentrated in the opening: measured by TF-IDF,
                            # the first ~21 tokens of an abstract already retrieve it at 47.8%
                            # top-1 out of 945, ~32 tokens at 57.4%, ~43 at 63.2%. That is a
                            # lexical floor for what the LLM can read from the same tokens.
                            # T=32 buys more negatives rather than longer ones, which is the
                            # better trade for a contrastive objective.
LAMBDA_RANK = 1.0           # weight on L_rank; L_LM stays primary and unweighted
RANK_HARD_FRACTION = 0.5    # share of negatives drawn from the node's OWN references rather
                            # than at random. Random arXiv negatives separate on topic alone,
                            # which is the exact shortcut every run so far has taken; a paper's
                            # own references are the negatives that force topic -> identity.

# gate=0 made ignoring the graph both the default and the starting point: at step 0 the model
# saw no graph at all while LoRA was already training, and the fastest descent was to learn the
# generic register. The gate=0 ablation is preserved by zeroing at *evaluation* time, which
# costs nothing — it never required training from zero.
GATE_INIT_FRACTION = 0.2    # target ||gate*delta|| / ||base|| at initialisation, calibrated on
                            # a real batch by PrefixAdapter.calibrate_gate rather than guessed

# --- Stage 11 (run 8): whitened node features --------------------------------
# The voyage features are anisotropic before the GNN ever touches them: over the 25,539 nodes
# reachable from the eval set, participation ratio 92.6 of 1024, and the mean of 25k unit
# vectors has norm 0.5975 where isotropy would give ~0.006. Two thirds of the pooled reference
# vector is one direction shared by every node, PrefixAdapter.trunk's LayerNorm normalises per
# sample and cannot remove a between-node component, so the adapter received a near-constant
# and greedy decoding emitted a near-constant. See docs/step8_representation_plan.md.
#
# ZCA whitening of the *inputs* is the correction, measured on the 945 eval nodes by
# pooled-reference retrieval: 64.34% raw -> 78.52% whitened. It is placed here, in the
# features, rather than as a matrix bolted onto the adapter, because the anisotropy belongs to
# the features — so the GNN, the neighbour tokens and the oracle are all treated identically
# and the arms stay comparable.
#
# whiten_features.py writes the file and refuses to if it does not clear the retrieval gate.
# Point NODE_EMBEDDINGS back at node_embeddings.npy to reproduce every run up to step 7.
RAW_NODE_EMBEDDINGS = os.path.join(JOINT_DIR, "node_embeddings.npy")
NODE_EMBEDDINGS = os.path.join(JOINT_DIR, "node_embeddings_whitened.npy")
WHITEN_EPS = 1e-3           # eigenvalue floor as a fraction of the largest eigenvalue. Without
                            # it the smallest directions — which are noise — are amplified by
                            # 1/sqrt(lambda) and the transform inflates exactly what carries
                            # least signal.
# The gate, and it is NOT the retrieval number. Measured with an honest train-only fit:
#
#   eval retrieval      64.34% -> 72.59%      (+8.3, not the +14.2 of the leaky fit)
#   pooled cosine       0.6602 -> 0.0547      (12x less shared direction)
#   participation ratio    122 -> 948 of 1024
#   ||mean direction||  0.5458 -> 0.0078      (isotropic would be ~0.006)
#
# Retrieval was never the diagnosis. Step 7 had 64.34% of retrievable information available and
# delivered 0.11%; what it did not have was a prefix that VARIED between nodes. So the gate is
# that the whitened features must not lose retrievable information, and must decisively fix the
# contrast. The oracle, which generates 944 distinct abstracts out of 945, is fed vectors at
# pairwise cosine 0.378; 0.10 clears it with room.
#
# eps hardly matters: swept over 945 held-out TRAINING nodes (never the eval set) it moves dev
# retrieval between 69.3% and 70.6% across 1e-4..1e-1, which is inside the noise on 945 nodes.
# 1e-3 is kept as the standard choice rather than chasing a 1.3-point difference.
WHITEN_COSINE_GATE = 0.10   # pooled-reference pairwise cosine after whitening

# Stage A gains VICReg's two missing terms (Bardes, Ponce & LeCun 2021). InfoNCE is *invariant*
# to anisotropy — F.normalize followed by tau=0.05 is a 20x amplification of cosine differences
# — which is how stage 10a scored 67% top-1 on a representation the adapter reads as a
# constant. The variance hinge prevents collapse outright; the covariance term decorrelates the
# dimensions. Ratio 25:1 as in the paper, scaled so the InfoNCE term stays primary.
VICREG_VAR = 1.0            # weight on the per-dimension standard-deviation hinge
VICREG_COV = 0.04           # weight on off-diagonal covariance
VICREG_GAMMA = 1.0          # target standard deviation per dimension
ENCODER_COSINE_GATE = 0.40  # Stage A also has to leave the output *spread*, not just ranked.
                            # The oracle, which generates 944 distinct abstracts, is fed
                            # vectors at pairwise cosine 0.378; step 7's graph arm got 0.660.

# L_LM and L_rank genuinely conflict — one rewards likely text, the other discriminative text.
# Step 7 ran at lambda 1.0 from step 0 and took a -0.39 coherence regression at exactly the
# point the rank term started winning. Warming in over the first epoch (7,750 nodes / 16 =
# ~484 optimizer steps) lets the LM term settle first.
RANK_WARMUP_STEPS = 484

# The ranking window is sampled from a random offset during training rather than always being
# the abstract's opening. This is objective alignment, NOT a bug fix: the gold abstracts are
# 98.6% distinct by word 8 (7,645 of 7,750), so the opening was never formulaic. The opening
# states the problem and the middle states the method, and methodology is the stated goal.
# Validation keeps offset 0 so rank_acc stays comparable with step 7's numbers.
RANK_STORED_TOKENS = 256    # tokens kept per candidate so a window can be sliced out of them

# The canary. Steps 5, 6 and 7 each shipped a collapsed model that no logged number flagged
# until an evaluation job had already spent seven GPU hours. This one is not a proxy: it
# generates, greedily, and counts how many distinct strings come back.
CANARY_NODES = 16
CANARY_TOKENS = 48          # past the 8-word opening the collapse is measured on, and ~18 s
                            # per validation on an A100

# --- Stage 12 (run 9): the embedding -> abstract decoder ---------------------
# Run 9 splits the task in two. The LLM side is trained to write a paper's abstract from that
# paper's OWN (whitened) embedding, on every train-split core paper rather than 7,750 of them;
# the graph side — estimating that embedding from the references — needs no LLM and is joined
# only at evaluation. See docs/run9_embedding_decoder.md.
#
# The decoder will be handed an *estimate* at evaluation, so it is trained on corrupted
# embeddings: each one is rotated to a cosine drawn uniformly from [ORACLE_MIN_COS, 1] with the
# true vector. Measured in whitened space, the plain reference mean sits at cosine
# 0.313 / 0.487 / 0.617 (p10 / p50 / p90) from its target, so 0.3 covers what the graph arm
# will feed it while the upper part of the range keeps the clean oracle arm in distribution.
ORACLE_MIN_COS = 0.3

# ~90k training papers is ~5,600 optimizer steps per pass — too long between checkpoints for a
# 48 h walltime. An "epoch" is therefore a fixed-size slice of one seeded permutation of the
# training nodes, so successive epochs walk through all of them before any repeats.
EPOCH_SIZE = 15000

# Model selection and the stop rules read generated text, not loss. At each epoch end the
# decoder writes VAL_ID_TOKENS tokens for VAL_ID_NODES fixed validation papers, twice — from the
# true embedding and from the plain reference mean — and each text is ranked against those
# papers' golds by TF-IDF (identify.py). best.pt is the epoch with the highest reference-mean
# MRR: the arm the thesis reports, measured without ever touching the eval set.
VAL_ID_NODES = 128
VAL_ID_TOKENS = 96          # ~70 words; the first ~32 words of an abstract already identify it
                            # lexically at 63% top-1 out of 945 (soft_prompt_only.md, step 7)

# Stop rules, enforced in code rather than written in a plan (run 8 ran 1,250 steps past a
# prose stop-gate). Checked at every epoch end from VAL_ID_GATE_EPOCH on; the epoch is saved
# first so the failure can be inspected, then the job exits non-zero.
#
# Both rules wait for epoch 2 since run 10a. Run 9 checked the canary at epoch 1 (937 steps);
# step 4, starting its gate at 0, needed ~1,000 steps before the vector moved validation loss.
CANARY_MIN_DISTINCT = 9     # of CANARY_NODES. Step 4's oracle wrote 944 distinct of 945
VAL_ID_GATE_EPOCH = 2       # from this epoch on, the oracle arm's top-1 must be at least
VAL_ID_MIN_LIFT = 10        # this many times chance (1 / VAL_ID_NODES)

# --- Stage 13 (run 10a): back to step 4's recipe, trained on real estimates ----------
# Run 9 stopped at epoch 1 with the vector unused (validation gain 0.0003 nats against step 4's
# 0.022 at the same step; gate x1.2 against x13). It had changed three things about the
# adapter's input at once relative to step 4: whitened features, gate-init 0.2 and synthetic
# noise. Run 10a returns to step 4's raw features and gate 0, and replaces the noise with the
# real estimate: each training paper is fed its reference mean with probability REFS_MIX, its
# own embedding otherwise. See docs/run9_embedding_decoder.md, "Run 10a".
FEATURES = {"raw": RAW_NODE_EMBEDDINGS, "whitened": NODE_EMBEDDINGS}
REFS_MIX = 0.5

# --- Stage 14 (run 11): reader, GNN imitation, joint fine-tune -------------------------
# Three phases, each trained from scratch so the run stands alone: (1) the reader, run 10a's
# recipe; (2) GNNs imitating the paper's raw embedding (pretrain_encoder.py), one winner chosen
# on validation cosine; (3) both fine-tuned together (train.py --reader). The 10a oracle arm
# showed the reader is close to the text baseline given a perfect vector (judge 3.41 vs 3.59),
# so the estimate is the bottleneck. See docs/run9_embedding_decoder.md, "Run 11".

# Phase 3. The cosine term keeps the GNN estimate in the space the reader decodes while the LM
# loss pulls on it; the GNN moves at a tenth of the reader's rate because it arrives trained.
JOINT_LAMBDA_COS = 1.0
JOINT_GNN_LR = 1e-5

# --- Stage 5: generation -----------------------------------------------------
MAX_NEW_TOKENS = GENERATION_RESERVE   # asking for more than the capped sets were frozen
                                      # under would need references the freeze dropped

REPETITION_PENALTY = 1.00
# Run 11: generate.py --no-repeat-ngram bans any repeated n-gram of this length. 17.6% of run
# 10a's oracle outputs repeat a whole sentence, scoring coherence 3.8 against 7.7. 0 disables it
# and reproduces every run before run 11; 6 is long enough that ordinary technical phrasing
# ("the number of ...") is not touched, and short enough to stop a sentence repeating.
NO_REPEAT_NGRAM = 0
