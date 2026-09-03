"""
Stage 7, step 2 — freeze which references each prompt shows.

`text.capped_example` used to decide a node's bibliography at call time, by dropping the
farthest-from-centroid reference until the *tokenizer's* count fit the budget. That makes the
bibliography a function of the tokenizer. Qwen3's BPE is more efficient on English than
Vicuna's SentencePiece, so under the same 4,096-token budget more references would survive,
and the two models would silently read different evidence — the exact confound the whole
comparison exists to avoid.

Nothing new is decided here. The frozen node sets were selected back in August by running this
same computation, so these lists were already implicit in every run since; they were simply
never written down, because reproducibility came from recomputing them with the same tokenizer.
This makes them explicit, so they stop depending on a tokenizer that is about to be replaced.

The list is computed once under the **Vicuna** tokenizer at GENERATION_RESERVE = 320, and
written to data/frozen/prompt_refs_capped.json. Every model then reads it. The tokenizer is
pinned in this file rather than taken from config.BASE_MODEL, because BASE_MODEL becomes Qwen
in step 4 and re-running this script afterwards must still reproduce the same file.

Covers all 9,501 nodes of the capped sets: 945 eval + 7,750 train + 806 valid.

The checks run after writing, and are the point of the exercise:

  1. coverage — the file's keys are exactly the three frozen sets, nothing more or less;
  2. every node keeps at least one reference, so the graph is always shown;
  3. kept references are a subsequence of the node's edges.csv order;
  4. the era-1 gate — experiments/prompt_sensitivity_gpt/01_era1/test_set.csv stores the
     finished prompt and reference list for 48 nodes, built at this reserve under this
     tokenizer. All 48 must match, in the lists and string-for-string in the prompts. This is
     the same method that folder used to validate era 1 and era 2 (0 mismatches both times);
  5. regression — reading the file back through capped_example(frozen=True) must give
     byte-identical prompts to the recompute path, which is what protects the existing runs.

Check 4 covers 48 of 9,501 nodes. The other 9,453 have no stored prompt to diff against; the
only guarantee for them is that they came from this code path at this reserve. Say so in the
write-up rather than claiming a verification that was not done.

Usage:
    cd src/adapter
    python freeze_prompt_refs.py
    python freeze_prompt_refs.py --regression 400   # wider round-trip sample
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import text
from config import FROZEN_DIR, GENERATION_RESERVE, JOINT_DIR, ROOT
from text import Corpus, capped_example

# Pinned, not config.BASE_MODEL: the freeze is *defined* by this tokenizer, and BASE_MODEL moves.
FREEZE_TOKENIZER = "lmsys/vicuna-7b-v1.5"

ERA1 = os.path.join(ROOT, "experiments", "prompt_sensitivity_gpt", "01_era1", "test_set.csv")


def load_sets():
    """-> {'train': [...], 'valid': [...], 'eval': [...]} from the frozen node lists."""
    with open(os.path.join(FROZEN_DIR, "training_sets_capped.json")) as f:
        sets = json.load(f)
    with open(os.path.join(FROZEN_DIR, "eval_nodes_capped.json")) as f:
        sets["eval"] = json.load(f)
    return {name: [int(n) for n in nodes] for name, nodes in sets.items()}


def freeze(tokenizer, corpus, embeddings, sets):
    """-> {node_id: [ref_node_id, ...]}, computed the old way, once."""
    refs = {}
    for name, nodes in sets.items():
        for i, node_id in enumerate(nodes, 1):
            if i % 500 == 0 or i == len(nodes):
                print(f"  {name:<6} {i:>5}/{len(nodes)}", flush=True)
            _, _, kept = capped_example(tokenizer, corpus, embeddings, node_id, frozen=False)
            refs[node_id] = [int(r) for r in kept]
    return refs


def check_coverage(refs, sets, corpus):
    wanted = {n for nodes in sets.values() for n in nodes}
    assert set(refs) == wanted, \
        f"coverage: {len(set(refs) ^ wanted)} nodes differ between the file and the frozen sets"

    for node_id, kept in refs.items():
        assert kept, f"node {node_id}: no reference fits, the graph cannot be shown"
        # a subsequence check, not a subset one: edges.csv order is not ascending
        remaining = iter(corpus.refs.get(node_id, []))
        assert all(r in remaining for r in kept), \
            f"node {node_id}: kept references left edges.csv order"
    print(f"  coverage:   {len(refs):,} nodes, all with >=1 reference, edges.csv order kept")


def check_era1(tokenizer, corpus, embeddings):
    """The hard gate: 48 stored era-1 prompts, matched string-for-string."""
    df = pd.read_csv(ERA1)
    for _, row in df.iterrows():
        node_id = int(row.node_id)
        prompt, _, kept = capped_example(tokenizer, corpus, embeddings, node_id)
        # space-separated, the way build_test_set.py wrote it and diagnostics.py reads it
        stored = [int(r) for r in str(row.ref_node_ids).split()]
        assert kept == stored, \
            f"era-1 node {node_id}: reference list differs from the stored build"
        assert prompt == row.prompt, \
            f"era-1 node {node_id}: prompt differs from the stored build"
    print(f"  era-1 gate: {len(df)}/{len(df)} prompts and reference lists match")


def check_regression(tokenizer, corpus, embeddings, sets, sample, seed):
    """The frozen path and the recompute path must agree byte for byte."""
    rng = np.random.default_rng(seed)
    for name, nodes in sets.items():
        drawn = rng.choice(nodes, min(sample, len(nodes)), replace=False)
        for node_id in drawn:
            node_id = int(node_id)
            a = capped_example(tokenizer, corpus, embeddings, node_id, frozen=True)
            b = capped_example(tokenizer, corpus, embeddings, node_id, frozen=False)
            assert a == b, f"{name} node {node_id}: frozen and recomputed prompts differ"
        print(f"  regression: {name:<6} {len(drawn):>4} nodes byte-identical")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regression", type=int, default=150,
                    help="nodes per set for the frozen-vs-recomputed round trip")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading tokenizer {FREEZE_TOKENIZER} …")
    tokenizer = AutoTokenizer.from_pretrained(FREEZE_TOKENIZER, use_fast=True)

    print("Loading corpus …")
    corpus = Corpus()
    embeddings = np.load(os.path.join(JOINT_DIR, "node_embeddings.npy"), mmap_mode="r")

    sets = load_sets()
    total = sum(len(n) for n in sets.values())
    print(f"\nFreezing {total:,} nodes at GENERATION_RESERVE = {GENERATION_RESERVE} …")
    refs = freeze(tokenizer, corpus, embeddings, sets)

    with open(text.FROZEN_REFS, "w") as f:
        json.dump({str(k): v for k, v in sorted(refs.items())}, f)
    print(f"\nWrote {text.FROZEN_REFS}")

    print("\nChecking …")
    check_coverage(refs, sets, corpus)
    check_era1(tokenizer, corpus, embeddings)
    check_regression(tokenizer, corpus, embeddings, sets, args.regression, args.seed)

    counts = sorted(len(v) for v in refs.values())
    print(f"\nreferences shown: mean {sum(counts) / len(counts):.2f}, "
          f"median {counts[len(counts) // 2]}, min {counts[0]}, max {counts[-1]}")
    print("OK — one bibliography per node, decided under the Vicuna tokenizer")


if __name__ == "__main__":
    main()
