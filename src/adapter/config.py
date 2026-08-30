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
BASE_MODEL = "lmsys/vicuna-7b-v1.5"

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
# Vicuna-7B-v1.5 is Llama-2-7b: 32 layers, 32 attention heads, head_dim 128, no GQA
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 128
NUM_PREFIX_TOKENS = 10
TRUNK_DIM = 768
CODE_DIM = 96
BASE_INIT_STD = 0.02        # Llama's own initializer_range

# --- Stage 4: training -------------------------------------------------------
MAX_LENGTH = 4096           # Vicuna-7b-v1.5 context; prefix tokens count against it
TRAIN_SUBSET = 4000         # first-run training set, sampled from the eligible train nodes
VALID_PER_DEGREE = 45       # 45 x 11 degrees = 495 validation nodes, as in the eval set
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0          # decaying base/gate would pull the prefix toward the ablation
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
SEED = 42

DEFAULT_SETS = "capped"     # which frozen node sets; --sets legacy reproduces the old runs

# The run schedule, previously literals in hpc/komondor/jobs/run_train_adapter.sh.
EPOCHS = 3                 # upper bound; early stopping usually ends the run sooner
PATIENCE = 2                # epochs without a better validation loss before stopping
BATCH_SIZE = 2              # micro-batch; 2 x 8 keeps the effective batch at 16 while
GRAD_ACCUM = 8              # holding memory at the capped sets' ~3,428-token mean
LOG_EVERY = 10              # optimizer steps
EVAL_EVERY = 250            # optimizer steps; at 100, validation cost more tokens than training

WANDB_PROJECT = "dytag-adapter"

# --- Stage 4b: LoRA on the frozen LLM ----------------------------------------
# Off unless train.py is given --lora. Only q_proj and v_proj, on all 32 layers: 64 modules
# x 2 x (4096 x 16) = 8.4M parameters, against the 5.7M of the encoder and adapter together.
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
