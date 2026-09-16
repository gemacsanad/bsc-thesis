#!/bin/bash
#SBATCH --account=c_gnn42
#SBATCH --job-name=prompt-sens-qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=06:00:00
#SBATCH --output=/project/c_gnn42/csanad_thesis/baseline/logs/prompt_sens_qwen_%j.out
#SBATCH --error=/project/c_gnn42/csanad_thesis/baseline/logs/prompt_sens_qwen_%j.err
#SBATCH --mail-type=END,FAIL

# The prompt-sensitivity run under Qwen3-14B — four prompt eras, 48 nodes each, no graph.
#
#   experiments/prompt_sensitivity_qwen/0N_eraN/generations.csv
#
# This is the Qwen twin of experiments/prompt_sensitivity_gpt/, and it reads the *same*
# 0N_eraN/test_set.csv files, copied byte for byte. The question is whether the four
# instructions that moved Vicuna-7B downward and gpt-5.6-luna upward (+1.23, era 1 -> era 4)
# do anything under the model this project actually uses.
#
# There is no adapter and no checkpoint here: all four arms are the frozen-LLM condition, the
# same one experiments/<date>_qwen14b-baseline/ generates. Nothing is trained, nothing is
# written to W&B, no .env is needed.
#
# All four arms run back to back in ONE job, on the same node and the same GPU, so nothing but
# the prompt differs between the four files.
#
# Prerequisites:
#   - adapter-qwen-gpu.sif built (hpc/komondor/containers/adapter-qwen-gpu.def) — the same
#     image the generation job uses; generate.py imports nothing src/adapter/generate.py did not
#   - Qwen/Qwen3-14B already in $BASE_DIR/.hf_cache — compute nodes do not re-download it
#   - data/processed/joint/{nodes.csv,edges.csv} and data/frozen/prompt_refs_capped.json, for
#     the pre-flight check only; generation itself reads nothing but the test sets
#
# Walltime: 4 x 48 = 192 generations. The 24 h generation job produced 945 x 2 rows, so a node
# costs at most ~45 s; 192 of them is ~2.4 h plus the model load. 6 h is ~2x that. Generation
# is resumable, so a timeout costs a queue wait, not work.

BASE_DIR=/project/c_gnn42/csanad_thesis/baseline
SIF=$BASE_DIR/adapter-qwen-gpu.sif

mkdir -p $BASE_DIR/logs

module load singularity

# Resumable: generate.py appends only the node_ids missing from each arm's output CSV, so a
# resubmit after a timeout costs at most one node and a finished arm costs nothing. Rerunning
# this script is therefore safe.
singularity exec --nv -B $BASE_DIR:/app "$SIF" bash -lc '
    export HF_HOME=/app/.hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /app/experiments/prompt_sensitivity_qwen
    nvidia-smi

    # ~30 s on CPU against a multi-hour job. Every failure it catches is silent — a prompt
    # that lost its Vicuna wrapper, an arm reading a different bibliography, a <think> block
    # that was never suppressed — and each one yields a fluent abstract rather than an error.
    # It also proves the era-1 arm is the 945-node baseline prompt, which is the only thing
    # tying this folder to the run it is compared against.
    python verify_prompts.py || exit 1

    echo "=== four prompt eras, frozen Qwen3-14B, no prefix ==="
    python generate.py --era all
'
