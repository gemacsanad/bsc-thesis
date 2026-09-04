"""
Stage 4, step 0 — prove the capped prompt path still matches the frozen node sets.

The failure this guards against is silent. `prepare_data_capped.py` chose each node on the
assumption that a particular set of references is shown in its prompt; if `train.py` or
`generate.py` later build a different set, the result is a fluent abstract written from the
wrong sources, not an error. Nothing downstream can tell, and the CSV looks fine.

So the check runs before the GPU hours, not after, and it exercises the real code path rather
than a reimplementation of it: `text.capped_example`, the same function the freeze itself used.

What is checked, on a sample of the frozen train / valid / eval nodes:

  1. the bibliography comes from the frozen file and from nothing else — passing `None` where
     the embedding matrix goes changes no prompt, target or reference list, because on the
     live path that array is never read. This is what makes "both models see the same
     evidence" a fact rather than an intention;
  2. every node yields at least one reference, so the graph is always shown;
  3. kept references are a subsequence of the node's edges.csv order, so a trimmed prompt
     reads like an untrimmed one;
  4. prompt + target + prefix fits MAX_LENGTH, which is what the freeze asserted;
  5. the prompt alone fits generate.py's budget, so the ERROR path is never reached.

Check 1 used to compare train.py's embedding source (a numpy view of the graph's feature
matrix) against generate.py's (a memory-map of the same file). Since Stage 7 step 2 froze the
reference lists, neither script consults the array to build a prompt, so that comparison passed
by construction. Asserting the array is *unused* is the same guarantee, still true.

Usage:
    cd src/adapter
    python verify_capped_prompts.py                 # ~1 min on a laptop CPU
    python verify_capped_prompts.py --sample 400    # slower, more thorough
    python verify_capped_prompts.py --sample 0      # every frozen node
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

from config import (BASE_MODEL, FROZEN_DIR, GENERATION_RESERVE, JOINT_DIR, MAX_LENGTH,
                    NUM_PREFIX_TOKENS)
from text import (INSTRUCTION, SYSTEM, Corpus, build_prompt, capped_example, encode,
                  frozen_refs, user_message)

# The finished chat wrapper, frozen. `build_prompt` asks the tokenizer for this via
# `apply_chat_template`, which is the right way to derive it and the wrong thing to trust
# blindly: a tokenizer_config update would move every prompt in both arms without a word.
# Written out here so the wrapper is reviewable in a diff rather than only at runtime.
#
# The `<think>\n\n</think>\n\n` is not a mistake. Qwen3 is a hybrid thinking model, and
# `enable_thinking=False` does not delete the block — it emits an empty, already-closed one,
# which is how the model is told to answer directly.
CHAT_WRAPPER = ("<|im_start|>system\n{system}<|im_end|>\n"
                "<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def check_chat_wrapper(tokenizer):
    """The delimiters are a calling convention; the text inside them is the experiment."""
    neighbours = [("A Title", "An abstract."), ("Another Title", "Another abstract.")]
    got = build_prompt(tokenizer, neighbours)
    want = CHAT_WRAPPER.format(system=SYSTEM, user=user_message(neighbours))
    assert got == want, (
        "the chat template has moved:\n"
        f"  got  {got[:120]!r}…\n  want {want[:120]!r}…")

    # The experimental content must survive the wrapper untouched. These two strings are the
    # variable the four prompt eras measured; the swap is supposed to change nothing about them.
    assert SYSTEM in got and INSTRUCTION in got, "the prompt text did not survive the wrapper"
    assert got.endswith("</think>\n\n"), "the assistant turn does not open where we think"
    print(f"  wrapper  ChatML, thinking disabled, {len(SYSTEM)}+{len(INSTRUCTION)} chars of "
          f"prompt text intact")


def check(name, nodes, tokenizer, corpus, embeddings, budget):
    """-> the largest prompt seen. Raises on the first disagreement."""
    worst = 0
    for node_id in nodes:
        node_id = int(node_id)
        prompt, target, refs = capped_example(tokenizer, corpus, embeddings, node_id)
        # None where the embedding matrix goes: on the frozen path it is never read, so this
        # must return the same thing. If a recompute ever creeps back in, it raises here
        # rather than quietly showing this model a different bibliography.
        blind = capped_example(tokenizer, corpus, None, node_id)

        assert (prompt, target, refs) == blind, \
            f"{name} node {node_id}: the prompt depends on the embeddings — it is not frozen"
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

    # The memory-map generate.py uses. Kept only so check 1 has something real to be blind to.
    embeddings = np.load(os.path.join(JOINT_DIR, "node_embeddings.npy"), mmap_mode="r")

    with open(os.path.join(FROZEN_DIR, "training_sets_capped.json")) as f:
        sets = json.load(f)
    with open(os.path.join(FROZEN_DIR, "eval_nodes_capped.json")) as f:
        sets["eval"] = json.load(f)

    rng = np.random.default_rng(args.seed)
    budget = MAX_LENGTH - NUM_PREFIX_TOKENS - GENERATION_RESERVE

    print(f"Checking the capped prompt path against the frozen sets (budget {budget:,}) …")
    check_chat_wrapper(tokenizer)

    # Before sampling: a node missing from the freeze raises KeyError deep inside training,
    # hours in. Coverage is cheap to state up front and the sample would not find it.
    wanted = {int(n) for nodes in sets.values() for n in nodes}
    missing = wanted - set(frozen_refs())
    assert not missing, \
        f"{len(missing)} frozen nodes have no entry in prompt_refs_capped.json, e.g. " \
        f"{sorted(missing)[:5]}"
    print(f"  frozen   {len(wanted):,} nodes, all present in prompt_refs_capped.json")

    worst = 0
    for name in ("train", "valid", "eval"):
        nodes = sets[name]
        if args.sample and args.sample < len(nodes):
            nodes = rng.choice(nodes, args.sample, replace=False)
        worst = max(worst, check(name, nodes, tokenizer, corpus, embeddings, budget))

    print(f"\nlargest prompt {worst:,} tokens, {budget - worst:,} under budget")
    print("OK — one prompt path, three callers, no overflow")


if __name__ == "__main__":
    main()
