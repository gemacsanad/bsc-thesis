"""
Stage 1b — the capped-prompt node sets.

Freezes new train / valid / eval node lists for the experiment that lifts the
out-degree ceiling from 15 to 100. Nothing here touches eval_nodes.json,
splits_eligible.json or training_sets.json: the current 495-node evaluation set and
the two runs built on it stay reproducible.

What changes
------------
Old sets: core, 5 <= out_degree <= 15, because the whole reference list went into the
prompt and 15 abstracts were as many as the 4,096-token context would hold.

New sets: core, 5 <= out_degree <= 100. The GNN still reads the complete neighbourhood;
only the prompt is trimmed, to at most PROMPT_MAX_REFS references and then down to
whatever fits the context, choosing by centroid distance (see neighbour_cap.py).
Out-degree therefore stops being a context-window constraint and becomes what it should
be — the amount of structure the graph pathway can exploit.

Sampling
--------
Per out-degree quotas, drawn separately from each split:

    train  CAPPED_TRAIN_PER_DEGREE per degree
    valid  CAPPED_VALID_PER_DEGREE per degree
    eval   CAPPED_EVAL_PER_DEGREE  per degree, from the test split

These are ceilings. Above out-degree ~39 the train split holds fewer than 100 nodes per
degree (82 at 40, 31 at 50, 0 at 99), and the valid split fewer than 10 above degree 72,
so those degrees contribute every node they have. The per-degree table printed at the end
is what any result has to be read against — the high-degree bins are small.

Fitting the context
-------------------
15 full-length abstracts overflow 4,096 tokens about half the time, so 15 is a ceiling and
not a target: references are dropped farthest-from-centroid first until the prompt fits

    MAX_LENGTH - NUM_PREFIX_TOKENS - max(GENERATION_RESERVE, len(target) + 1)

which leaves the node's own abstract room to be trained on, and never less than
GENERATION_RESERVE tokens for generation to write into. Nothing is dropped for overflow;
a node is only skipped if not even its single closest reference fits, which the report
counts as "unusable". What varies with out-degree is now how many references survive,
and that number is in the per-degree table.

Outputs (data/frozen/)
----------------------
splits_eligible_capped.json — {"train": [...], "valid": [...], "test": [...]}, everything eligible
training_sets_capped.json   — {"train": [...], "valid": [...]}, the sampled sets
eval_nodes_capped.json      — the sampled eval nodes

Usage:
    python adapter/prepare_data_capped.py
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from config import (CAPPED_EVAL_PER_DEGREE, CAPPED_MAX_OUT_DEGREE,
                    CAPPED_MIN_OUT_DEGREE, CAPPED_TRAIN_PER_DEGREE,
                    CAPPED_VALID_PER_DEGREE, FROZEN_DIR, GENERATION_RESERVE,
                    JOINT_DIR, MAX_LENGTH, NUM_PREFIX_TOKENS, PROMPT_MAX_REFS)
from neighbour_cap import fit_refs
from text import Corpus, build_prompt, encode

DEGREES = range(CAPPED_MIN_OUT_DEGREE, CAPPED_MAX_OUT_DEGREE + 1)


def example(tokenizer, corpus, embeddings, node_id):
    """-> (prompt, target, kept_refs) for one node, trimmed to fit the context.

    The budget leaves room for the prefix and for the abstract: the node's own abstract
    when it is longer than GENERATION_RESERVE, so the training sequence is complete, and
    never less than GENERATION_RESERVE, so generation has somewhere to write.
    """
    target = corpus.abstract[node_id]
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    reserve = max(GENERATION_RESERVE, len(target_ids) + 1)   # +1 for the EOS token
    budget = MAX_LENGTH - NUM_PREFIX_TOKENS - reserve

    def measure(kept):
        prompt = build_prompt([(corpus.title[n], corpus.abstract[n]) for n in kept])
        return len(tokenizer(prompt).input_ids)

    refs = fit_refs(corpus.refs.get(node_id, []), embeddings, measure, budget)
    return (build_prompt([(corpus.title[n], corpus.abstract[n]) for n in refs]),
            target, refs)


def collect(tokenizer, corpus, embeddings, candidates, wanted, rng):
    """Walk shuffled candidates until `wanted` are collected. -> (nodes, lengths, refs, seen)."""
    pool = list(candidates)
    rng.shuffle(pool)
    kept, lengths, refs, seen = [], [], [], 0
    for node_id in pool:
        seen += 1
        prompt, target, node_refs = example(tokenizer, corpus, embeddings, node_id)
        if not node_refs:
            continue        # not even one reference fits — the graph cannot be shown at all
        input_ids, _ = encode(tokenizer, prompt, target)
        assert len(input_ids) + NUM_PREFIX_TOKENS <= MAX_LENGTH, \
            f"node {node_id}: trimmed sequence still overflows"
        kept.append(int(node_id))
        lengths.append(len(input_ids))
        refs.append(len(node_refs))
        if len(kept) == wanted:
            break
    return kept, lengths, refs, seen


def sample_by_degree(tokenizer, corpus, embeddings, candidates, degree, per_degree, rng, label):
    """Draw up to `per_degree` fitting nodes at every out-degree. -> (nodes, lengths, stats)."""
    by_degree = {}
    for node_id in candidates:
        by_degree.setdefault(int(degree[node_id]), []).append(node_id)

    nodes, lengths, refs, stats = [], [], [], []
    for d in DEGREES:
        pool = sorted(by_degree.get(d, []))       # sorted so the shuffle is reproducible
        kept, kept_len, kept_refs, seen = collect(
            tokenizer, corpus, embeddings, pool, per_degree, rng)
        nodes += kept
        lengths += kept_len
        refs += kept_refs
        stats.append({
            "degree": d,
            "available": len(pool),
            "kept": len(kept),
            "short": per_degree - len(kept),
            "dropped": seen - len(kept),
            "refs_mean": float(np.mean(kept_refs)) if kept_refs else 0.0,
            "tokens_mean": int(np.mean(kept_len)) if kept_len else 0,
        })
    short = [s for s in stats if s["short"] > 0]
    print(f"  {label}: {len(nodes):,} nodes "
          f"({len(short)} of {len(list(DEGREES))} degrees below the {per_degree}-node quota)")
    return nodes, lengths, refs, stats


def report(stats, per_degree):
    print(f"  degree  available  kept/{per_degree}  refs shown  mean tokens")
    for s in stats:
        flag = "" if s["short"] == 0 else f"  short {s['short']}"
        flag += "" if s["dropped"] == 0 else f"  ({s['dropped']} unusable)"
        print(f"  {s['degree']:>6}  {s['available']:>9,}  {s['kept']:>4}      "
              f"{s['refs_mean']:>6.1f}      {s['tokens_mean']:>6,}{flag}")
    kept = sum(s["kept"] for s in stats)
    print(f"  {'total':>6}  {sum(s['available'] for s in stats):>9,}  {kept:>4}      "
          f"{np.average([s['refs_mean'] for s in stats if s['kept']], weights=[s['kept'] for s in stats if s['kept']]):>6.1f}"
          f"           ({sum(s['dropped'] for s in stats)} nodes unusable)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default=os.path.join(JOINT_DIR, "nodes.csv"))
    ap.add_argument("--edges", default=os.path.join(JOINT_DIR, "edges.csv"))
    ap.add_argument("--splits", default=os.path.join(JOINT_DIR, "splits.json"))
    ap.add_argument("--embeddings", default=os.path.join(JOINT_DIR, "node_embeddings.npy"))
    ap.add_argument("--model", default="lmsys/vicuna-7b-v1.5", help="tokenizer to measure with")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=FROZEN_DIR)
    args = ap.parse_args()

    print(f"Loading tokenizer {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("Loading the joint dataset …")
    nodes = pd.read_csv(args.nodes, usecols=["node_id", "core"])
    corpus = Corpus(args.nodes, args.edges)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    print(f"  {len(nodes):,} nodes, embeddings {embeddings.shape}")

    out_degree = corpus.refs.apply(len)
    nodes["out_degree"] = nodes["node_id"].map(out_degree).fillna(0).astype(int)
    degree = nodes.set_index("node_id")["out_degree"]

    eligible = set(nodes.loc[
        nodes["core"]
        & nodes["out_degree"].between(CAPPED_MIN_OUT_DEGREE, CAPPED_MAX_OUT_DEGREE),
        "node_id",
    ])
    print(f"  {len(eligible):,} eligible (core, "
          f"{CAPPED_MIN_OUT_DEGREE}-{CAPPED_MAX_OUT_DEGREE} out-edges)")

    with open(args.splits) as f:
        splits = json.load(f)
    splits_eligible = {name: sorted(eligible & set(ids)) for name, ids in splits.items()}
    for name in ("train", "valid", "test"):
        print(f"    {name:<6} {len(splits_eligible[name]):>7,}")

    # --- sample ------------------------------------------------------------
    rng = random.Random(args.seed)
    print(f"\nSampling (at most {PROMPT_MAX_REFS} references, trimmed to fit "
          f"{MAX_LENGTH} tokens with {GENERATION_RESERVE}+ held back for the abstract) …")
    train, train_len, train_refs, train_stats = sample_by_degree(
        tokenizer, corpus, embeddings, splits_eligible["train"], degree,
        CAPPED_TRAIN_PER_DEGREE, rng, "train")
    valid, valid_len, valid_refs, valid_stats = sample_by_degree(
        tokenizer, corpus, embeddings, splits_eligible["valid"], degree,
        CAPPED_VALID_PER_DEGREE, rng, "valid")
    eval_nodes, eval_len, eval_refs, eval_stats = sample_by_degree(
        tokenizer, corpus, embeddings, splits_eligible["test"], degree,
        CAPPED_EVAL_PER_DEGREE, rng, "eval")

    # --- checks ------------------------------------------------------------
    core_ids = set(nodes.loc[nodes["core"], "node_id"])
    for name, ids in (("train", train), ("valid", valid), ("eval", eval_nodes)):
        assert len(set(ids)) == len(ids), f"duplicate {name} nodes"
        assert set(ids) <= core_ids, f"some {name} nodes are not core nodes"
        assert set(ids) <= eligible, f"some {name} nodes are not eligible"
    assert set(eval_nodes) <= set(splits["test"]), "some eval nodes are not in the test split"
    sets = {"train": set(train), "valid": set(valid), "eval": set(eval_nodes)}
    for a, b in (("train", "valid"), ("train", "eval"), ("valid", "eval")):
        overlap = sets[a] & sets[b]
        assert not overlap, f"{len(overlap)} nodes in both {a} and {b}"

    # --- write -------------------------------------------------------------
    paths = {
        "splits_eligible_capped.json": splits_eligible,
        "training_sets_capped.json": {"train": sorted(train), "valid": sorted(valid)},
        "eval_nodes_capped.json": sorted(eval_nodes),
    }
    for name, payload in paths.items():
        with open(os.path.join(args.out_dir, name), "w") as f:
            json.dump(payload, f)

    # --- report ------------------------------------------------------------
    for name, stats, per_degree in (("train", train_stats, CAPPED_TRAIN_PER_DEGREE),
                                    ("valid", valid_stats, CAPPED_VALID_PER_DEGREE),
                                    ("eval", eval_stats, CAPPED_EVAL_PER_DEGREE)):
        print(f"\n{name}")
        report(stats, per_degree)

    for name, lengths, refs in (("train", train_len, train_refs),
                                ("valid", valid_len, valid_refs),
                                ("eval", eval_len, eval_refs)):
        lengths = sorted(lengths)
        full = sum(r == PROMPT_MAX_REFS for r in refs)
        print(f"\n{name}: {len(lengths):,} nodes, sequence length "
              f"mean {sum(lengths) / len(lengths):,.0f}, "
              f"median {lengths[len(lengths) // 2]:,}, "
              f"p99 {lengths[int(len(lengths) * 0.99)]:,}, max {lengths[-1]:,}")
        print(f"  references shown: mean {np.mean(refs):.1f}, median {int(np.median(refs))}, "
              f"min {min(refs)}, {full / len(refs):.0%} at the {PROMPT_MAX_REFS}-reference cap")

    # The old eval set is a subset of the same test split, so some of it can reappear here.
    with open(os.path.join(FROZEN_DIR, "eval_nodes.json")) as f:
        old_eval = set(json.load(f))
    print(f"\noverlap with the old 495-node eval set: "
          f"{len(old_eval & set(eval_nodes))} nodes (both drawn from the test split)")

    for name in paths:
        print(f"Wrote {os.path.join(args.out_dir, name)}")


if __name__ == "__main__":
    main()
