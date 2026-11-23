"""Paths, model choice and hyperparameters shared by every script in src/adapter."""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# --- Which dataset ------------------------------------------------------------
DATASET = os.environ.get("DYTAG_DATASET", "papers")
assert DATASET in ("papers", "children"), f"DYTAG_DATASET={DATASET!r}: papers or children"

# --- Paths -------------------------------------------------------------------
JOINT_DIR = os.path.join(ROOT, "data", "processed", "joint")   # nodes/edges/embeddings
OGBN_DIR = os.path.join(ROOT, "data", "raw", "ogbn_arxiv")     # the original download
FROZEN_DIR = os.path.join(ROOT, "data", "frozen")              # eval_nodes, splits_eligible, training_sets
if DATASET == "children":
    JOINT_DIR = os.path.join(ROOT, "data", "children_books")
    FROZEN_DIR = os.path.join(ROOT, "data", "frozen", "children_books")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")            # one directory per run
ENV_FILE = os.path.join(ROOT, ".env")

BASE_MODEL = "Qwen/Qwen3-14B"

# --- Node sets: out-degree range and nodes per degree -------------------------
MIN_OUT_DEGREE = 5
MAX_OUT_DEGREE = 15
EVAL_PER_DEGREE = 45

CAPPED_MIN_OUT_DEGREE = 5
CAPPED_MAX_OUT_DEGREE = 100
CAPPED_TRAIN_PER_DEGREE = 200
CAPPED_VALID_PER_DEGREE = 10
CAPPED_EVAL_PER_DEGREE = 10

# --- Prompt and neighbourhood ------------------------------------------------
PROMPT_MAX_REFS = 15        # ceiling on references shown in the prompt; the GNN still sees all
GENERATION_RESERVE = 320    # tokens kept free for the generated abstract

# --- GNN estimator -----------------------------------------------------------
IN_CHANNELS = 1024          # voyage-4-large embedding dim
HIDDEN_CHANNELS = 1024
OUT_CHANNELS = 1024
DROPOUT = 0.2
GNN_LAYERS = 2              # default depth; --gnn-layers overrides it per run

GNN_CONV = "gatv2"
GNN_HEADS = 4               # GATv2 heads, concatenated to the layer width


def num_neighbors(gnn_layers=GNN_LAYERS):
    """Sampling fan-out for a `gnn_layers`-deep encoder: every neighbour at every hop."""
    return [-1] * gnn_layers

# --- Prefix adapter (Qwen3-14B shapes) ---------------------------------------
NUM_LAYERS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_PREFIX_TOKENS = 32
TRUNK_DIM = 1024
CODE_DIM = 192
BASE_INIT_STD = 0.02        # the base model's initializer_range

# --- Neighbour tokens (not used in the final model: --neighbour-tokens 0) ----
NUM_NEIGHBOUR_TOKENS = 128
HIDDEN_SIZE = 5120          # Qwen3-14B hidden size

# --- Reader training ---------------------------------------------------------
MAX_LENGTH = 4096           # prefix tokens count against it
TRAIN_SUBSET = 4000
VALID_PER_DEGREE = 45
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
SEED = 42

DEFAULT_SETS = "capped"     # which frozen node sets (--sets); the final run uses run9
if DATASET == "children":
    DEFAULT_SETS = "children"

EPOCHS = 10                 # upper bound; early stopping usually ends the run sooner
PATIENCE = 2                # epochs without a better validation loss before stopping
BATCH_SIZE = 1              # micro-batch; effective batch = 1 x 16
GRAD_ACCUM = 16
LOG_EVERY = 10              # optimizer steps
EVAL_EVERY = 250            # optimizer steps

WANDB_PROJECT = "dytag-adapter"

# --- LoRA --------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32             # 2 x rank; peft scales the adapter by alpha / r
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]

# --- GNN pretraining (pretrain_encoder.py) -----------------------------------
PRETRAIN_TAU = 0.05         # InfoNCE temperature
PRETRAIN_LR = 1e-3
PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH = 256
PRETRAIN_NEGATIVES = 4096   # sampled candidates on top of the in-batch positives

# --- Ranking loss (rank.py; disabled in the final run with --lambda-rank 0) ---
RANK_NEGATIVES = 8          # wrong abstracts scored against each node's prefix
RANK_TOKENS = 32            # opening tokens of each abstract that are scored
LAMBDA_RANK = 1.0
RANK_HARD_FRACTION = 0.5    # share of negatives drawn from the node's own references

GATE_INIT_FRACTION = 0.2    # target ||gate*delta|| / ||base|| at initialisation

# --- Node features -----------------------------------------------------------
RAW_NODE_EMBEDDINGS = os.path.join(JOINT_DIR, "node_embeddings.npy")
NODE_EMBEDDINGS = os.path.join(JOINT_DIR, "node_embeddings_whitened.npy")
WHITEN_EPS = 1e-3
WHITEN_COSINE_GATE = 0.10

# --- GNN pretraining checks ---------------------------------------------------
VICREG_VAR = 1.0            # weight on the per-dimension standard-deviation hinge
VICREG_COV = 0.04           # weight on off-diagonal covariance
VICREG_GAMMA = 1.0          # target standard deviation per dimension
ENCODER_COSINE_GATE = 0.40  # maximum mean pairwise cosine of the GNN outputs

RANK_WARMUP_STEPS = 484

RANK_STORED_TOKENS = 256

# --- Reader validation: collapse canary and identification -------------------
CANARY_NODES = 16
CANARY_TOKENS = 48
ORACLE_MIN_COS = 0.3

EPOCH_SIZE = 15000

VAL_ID_NODES = 128
VAL_ID_TOKENS = 96

CANARY_MIN_DISTINCT = 9     # of CANARY_NODES
VAL_ID_GATE_EPOCH = 2       # from this epoch on, the oracle arm's top-1 must be at least
VAL_ID_MIN_LIFT = 10        # this many times chance (1 / VAL_ID_NODES)

FEATURES = {"raw": RAW_NODE_EMBEDDINGS, "whitened": NODE_EMBEDDINGS}
REFS_MIX = 0.5              # share of training papers fed their reference mean, not own embedding

# --- Joint fine-tuning --------------------------------------------------------
# The cosine term keeps the GNN estimate in the space the reader decodes; the GNN learns at a
# tenth of the reader's rate because it is already trained.
JOINT_LAMBDA_COS = 1.0
JOINT_GNN_LR = 1e-5
JOINT_MAX_SUBGRAPH_EDGES = 60_000 if DATASET == "children" else None

# --- Generation --------------------------------------------------------------
MAX_NEW_TOKENS = GENERATION_RESERVE
if DATASET == "children":
    MAX_NEW_TOKENS = 1024

# The LaTeX cleaner (text.clean_abstract) is only applied to abstracts.
CLEAN_TARGETS = DATASET == "papers"
# The external id generate.py writes beside node_id.
SOURCE_ID = "mag_id" if DATASET == "papers" else "cstag_id"


def check_sets(sets):
    """`--sets children` goes with DYTAG_DATASET=children, and every other set with papers."""
    assert (sets == "children") == (DATASET == "children"), (
        f"--sets {sets} with DYTAG_DATASET={DATASET}: the children's books use --sets children "
        f"and DYTAG_DATASET=children together")

REPETITION_PENALTY = 1.00
NO_REPEAT_NGRAM = 0
