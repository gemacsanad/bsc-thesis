"""
Configuration for the adapter pipeline.

Every tunable constant used by prepare_data.py, graph.py and model.py lives here,
so changing one number does not mean hunting through three files. Values that are
already exposed on the command line (input paths, batch sizes, the sampling seed)
stay as argparse defaults in the scripts that own them.

Paths are resolved from this file's location: ROOT is the repository root.
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# --- Stage 1: eligible nodes and the frozen evaluation set -------------------
MIN_OUT_DEGREE = 5
MAX_OUT_DEGREE = 15
EVAL_PER_DEGREE = 45

# --- Stage 2: neighbourhood sampling -----------------------------------------
NUM_NEIGHBORS = [-1, -1]    # all 1-hop and all 2-hop neighbours

# --- Stage 3a: GraphSAGE encoder ---------------------------------------------
IN_CHANNELS = 1024          # voyage-4-large embedding dim
HIDDEN_CHANNELS = 1024
OUT_CHANNELS = 1024
DROPOUT = 0.2

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
