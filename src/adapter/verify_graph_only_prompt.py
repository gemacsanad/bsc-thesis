"""
Soft-prompt-only, step 0 — prove the prompt train.py and generate.py build is the one intended.

With no references in the prompt there
is no frozen bibliography to check against; what is left to go wrong silently is:

  1. the chat wrapper — `apply_chat_template` derives it from tokenizer_config, so a tokenizer
     update would move the prompt of every node in both arms without a word. The finished
     string is written out below and compared exactly;
  2. the prompt leaking the paper — it must contain no reference block and none of the
     target's own abstract;
  3. the training target — every frozen node needs a non-empty abstract (an empty one trains
     on a lone EOS), the labels must mask exactly the prompt, and prompt + target + prefix
     must fit MAX_LENGTH.

Every frozen train / valid / eval node is checked; with a constant prompt that takes seconds.

Usage:
    cd src/adapter
    python verify_graph_only_prompt.py
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from config import (BASE_MODEL, CLEAN_TARGETS, DATASET, FROZEN_DIR, MAX_LENGTH,
                    MAX_NEW_TOKENS, NUM_PREFIX_TOKENS)
from text import GRAPH_ONLY_SYSTEM, GRAPH_ONLY_USER, Corpus, encode, graph_only_example

# The `<think>\n\n</think>\n\n` is not a mistake: `enable_thinking=False` emits an empty,
# already-closed block, which is how Qwen3 is told to answer directly.
EXPECTED = ("<|im_start|>system\n" + GRAPH_ONLY_SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + GRAPH_ONLY_USER + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=BASE_MODEL)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    corpus = Corpus(clean=CLEAN_TARGETS)

    train_file, eval_file = (("training_sets.json", "eval_nodes.json") if DATASET == "children"
                             else ("training_sets_capped.json", "eval_nodes_capped.json"))
    with open(os.path.join(FROZEN_DIR, train_file)) as f:
        sets = {k: v for k, v in json.load(f).items() if k in ("train", "valid")}
    with open(os.path.join(FROZEN_DIR, eval_file)) as f:
        sets["eval"] = json.load(f)

    print("Checking the graph-only prompt …")
    prompt, _ = graph_only_example(tokenizer, corpus, sets["train"][0])
    assert prompt == EXPECTED, (
        f"the chat template has moved:\n  got  {prompt[:160]!r}…\n  want {EXPECTED[:160]!r}…")
    assert "Title:" not in prompt and "Source publications" not in prompt, \
        "a reference block is in the prompt"
    n_prompt = len(tokenizer(prompt).input_ids)
    print(f"  wrapper  ChatML, thinking disabled, {n_prompt} tokens, identical for every node")

    worst = 0
    for name, nodes in sets.items():
        for node_id in nodes:
            got, target = graph_only_example(tokenizer, corpus, node_id)
            assert got == prompt, f"{name} node {node_id}: the prompt is not constant"
            assert target.strip(), f"{name} node {node_id}: empty target abstract"
            assert target not in got, f"{name} node {node_id}: the target is in the prompt"

            input_ids, labels = encode(tokenizer, got, target)
            assert labels[:n_prompt] == [-100] * n_prompt and labels[n_prompt] != -100, \
                f"{name} node {node_id}: labels do not mask exactly the prompt"
            assert len(input_ids) + NUM_PREFIX_TOKENS <= MAX_LENGTH, \
                f"{name} node {node_id}: sequence overflows MAX_LENGTH"
            worst = max(worst, len(input_ids) + NUM_PREFIX_TOKENS)
            # Books: a target generation could not reproduce within the cap is not in the sets.
            if DATASET == "children":
                assert len(input_ids) - n_prompt <= MAX_NEW_TOKENS, \
                    f"{name} node {node_id}: target + EOS longer than MAX_NEW_TOKENS"
        print(f"  {name:<6} {len(nodes):>5,} nodes ok")

    print(f"\nlongest sequence {worst:,} tokens incl. prefix, {MAX_LENGTH - worst:,} under "
          f"MAX_LENGTH")
    print("OK — one constant prompt, no reference text, every target trainable")


if __name__ == "__main__":
    main()
