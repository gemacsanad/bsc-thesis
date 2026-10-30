"""
The books' text arm — freeze which co-purchased books each evaluation book's prompt lists.

The papers' counterpart is freeze_prompt_refs.py. The selection is the same (neighbour_cap.
fit_refs): the PROMPT_MAX_REFS neighbours closest to the neighbourhood centroid, then the
farthest dropped one at a time until the prompt fits. The budget is the one generate.py checks,
MAX_LENGTH - NUM_PREFIX_TOKENS - MAX_NEW_TOKENS, under the Qwen3 tokenizer the arm runs with,
so no frozen prompt can produce an ERROR row. There is no Vicuna history to pin.

Only the 945 evaluation books: the text arm is frozen Qwen3-14B, nothing is trained on it.

Needs the Qwen3 tokenizer (the DyTAG-qwen env).

Usage:
    cd src/adapter
    DYTAG_DATASET=children python freeze_book_refs.py
"""

import json
import os

import numpy as np
from transformers import AutoTokenizer

from config import (BASE_MODEL, DATASET, FROZEN_DIR, JOINT_DIR, MAX_LENGTH, MAX_NEW_TOKENS,
                    NUM_PREFIX_TOKENS)
from neighbour_cap import fit_refs
from text import BOOK_TEXT_REFS, Corpus, book_text_example


def main():
    assert DATASET == "children", "the text arm is frozen for the books: DYTAG_DATASET=children"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    corpus = Corpus(clean=False)
    embeddings = np.load(os.path.join(JOINT_DIR, "node_embeddings.npy"), mmap_mode="r")
    with open(os.path.join(FROZEN_DIR, "eval_nodes.json")) as f:
        eval_nodes = json.load(f)

    budget = MAX_LENGTH - NUM_PREFIX_TOKENS - MAX_NEW_TOKENS
    print(f"budget {budget:,} prompt tokens = {MAX_LENGTH} - {NUM_PREFIX_TOKENS} prefix "
          f"- {MAX_NEW_TOKENS} generation")

    def measure(kept):
        prompt, _ = book_text_example(tokenizer, corpus, node, refs=kept)
        return len(tokenizer(prompt).input_ids)

    frozen, lengths = {}, []
    for node in eval_nodes:
        kept = fit_refs(corpus.refs[node], embeddings, measure, budget)
        assert kept, f"node {node}: not even one neighbour fits the budget"
        frozen[node] = [int(n) for n in kept]
        lengths.append(measure(kept))

    with open(BOOK_TEXT_REFS, "w") as f:
        json.dump({str(k): v for k, v in sorted(frozen.items())}, f)

    shown = np.array([len(v) for v in frozen.values()])
    degree = np.array([len(corpus.refs[n]) for n in eval_nodes])
    print(f"wrote {BOOK_TEXT_REFS}: {len(frozen)} books")
    print(f"neighbours shown: mean {shown.mean():.2f}, median {np.median(shown):.0f}, "
          f"min {shown.min()}, max {shown.max()}; all shown for {(shown == degree).mean():.1%} "
          f"(degree median {np.median(degree):.0f})")
    print(f"prompt tokens: median {np.median(lengths):.0f}, max {max(lengths):,} "
          f"(budget {budget:,})")

    # The round trip generate.py takes: the frozen file read back gives the prompts measured.
    for node, n_tokens in zip(eval_nodes[:50], lengths[:50]):
        prompt, target = book_text_example(tokenizer, corpus, node)
        assert len(tokenizer(prompt).input_ids) == n_tokens and target not in prompt
    print("round trip: 50 prompts rebuilt from the file match, none contains its target")


if __name__ == "__main__":
    main()
