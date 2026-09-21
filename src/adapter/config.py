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

# --- Stage 3a: GraphSAGE encoder ---------------------------------------------
IN_CHANNELS = 1024          # voyage-4-large embedding dim
HIDDEN_CHANNELS = 1024
OUT_CHANNELS = 1024
DROPOUT = 0.2
GNN_LAYERS = 2              # default depth; --gnn-layers overrides it per run


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
LORA_R = 16
LORA_ALPHA = 32             # 2 x rank; peft scales the adapter by alpha / r
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]

# --- Stage 5: generation -----------------------------------------------------
MAX_NEW_TOKENS = GENERATION_RESERVE   # asking for more than the capped sets were frozen
                                      # under would need references the freeze dropped

REPETITION_PENALTY = 1.00
