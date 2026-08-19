"""
Stage 4, step 0 — prove the capped prompt path still matches the frozen node sets.

The failure this guards against is silent. `prepare_data_capped.py` chose each node on the
assumption that a particular set of references is shown in its prompt; if `train.py` or
`generate.py` later build a different set, the result is a fluent abstract written from the
wrong sources, not an error. Nothing downstream can tell, and the CSV looks fine.

So the check runs before the GPU hours, not after, and it exercises the real code path rather
than a reimplementation of it: `text.capped_example`, the same function the freeze itself used.

What is checked, on a sample of the frozen train / valid / eval nodes:

  1. train.py's embedding source (a numpy view of the graph's own feature matrix) and
     generate.py's (a memory-map of the same file) produce identical prompts, targets and
     reference lists — the two scripts must not diverge on how the array is loaded;
  2. every node yields at least one reference, so the graph is always shown;
  3. kept references are a subsequence of the node's edges.csv order, so a trimmed prompt
     reads like an untrimmed one;
  4. prompt + target + prefix fits MAX_LENGTH, which is what the freeze asserted;
  5. the prompt alone fits generate.py's budget, so the ERROR path is never reached.

Usage:
    cd src/adapter
    python verify_capped_prompts.py                 # ~1 min on a laptop CPU
    python verify_capped_prompts.py --sample 400    # slower, more thorough
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

from config import (BASE_MODEL, FROZEN_DIR, GENERATION_RESERVE, JOINT_DIR, MAX_LENGTH,
                    NUM_PREFIX_TOKENS)
from text import Corpus, capped_example, encode


def check(name, nodes, tokenizer, corpus, view, mapped, budget):
    """-> the largest prompt seen. Raises on the first disagreement."""
    worst = 0
    for node_id in nodes:
        node_id = int(node_id)
        # train.py's path and generate.py's path, which must not differ
        prompt, target, refs = capped_example(tokenizer, corpus, view, node_id)
        prompt_mm, target_mm, refs_mm = capped_example(tokenizer, corpus, mapped, node_id)

        assert prompt == prompt_mm, f"{name} node {node_id}: prompts differ between paths"
        assert target == target_mm and refs == refs_mm, \
            f"{name} node {node_id}: targets or references differ between paths"
        assert refs, f"{name} node {node_id}: no reference fits, the graph cannot be shown"

        # a subsequence check, not a subset one: edges.csv order is not ascending
        remaining = iter(corpus.refs.get(node_id, []))
        assert all(r in remaining for r in refs), \
            f"{name} node {node_id}: kept references left edges.csv order"

        input_ids, _ = encode(tokenizer, prompt, target)
        assert len(input_ids) + NUM_PREFIX_TOKENS <= MAX_LENGTH, \
            f"{name} node {node_id}: sequence overflows MAX_LENGTH"

        n_prompt = len(tokenizer(prompt).input_ids)
        assert n_prompt <= budget, \
            f"{name} node {node_id}: prompt {n_prompt} tokens exceeds the {budget} budget"
        worst = max(worst, n_prompt)

    print(f"  {name:<6} {len(nodes):>4} nodes ok")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--sample", type=int, default=100,
                    help="nodes drawn per set; 0 checks every frozen node (slow)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    corpus = Corpus()

    path = os.path.join(JOINT_DIR, "node_embeddings.npy")
    view = np.load(path)                      # as train.py sees it, via data.x
    mapped = np.load(path, mmap_mode="r")     # as generate.py sees it

    with open(os.path.join(FROZEN_DIR, "training_sets_capped.json")) as f:
        sets = json.load(f)
    with open(os.path.join(FROZEN_DIR, "eval_nodes_capped.json")) as f:
        sets["eval"] = json.load(f)

    rng = np.random.default_rng(args.seed)
    budget = MAX_LENGTH - NUM_PREFIX_TOKENS - GENERATION_RESERVE

    print(f"Checking the capped prompt path against the frozen sets (budget {budget:,}) …")
    worst = 0
    for name in ("train", "valid", "eval"):
        nodes = sets[name]
        if args.sample and args.sample < len(nodes):
            nodes = rng.choice(nodes, args.sample, replace=False)
        worst = max(worst, check(name, nodes, tokenizer, corpus, view, mapped, budget))

    print(f"\nlargest prompt {worst:,} tokens, {budget - worst:,} under budget")
    print("OK — one prompt path, three callers, no overflow")


if __name__ == "__main__":
    main()
