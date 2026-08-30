"""
Usage:
    python adapter/text.py          # freeze the subsets (needs the Vicuna tokenizer)
"""

import argparse
import json
import os
import random

import pandas as pd
from transformers import AutoTokenizer

from config import (BASE_MODEL, FROZEN_DIR, GENERATION_RESERVE, JOINT_DIR, MAX_LENGTH,
                    MAX_OUT_DEGREE, MIN_OUT_DEGREE, NUM_PREFIX_TOKENS,
                    TRAIN_SUBSET, VALID_PER_DEGREE)
from neighbour_cap import fit_refs

SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


INSTRUCTION = (
    "You are a scientific writing assistant. "
    "Based on the source publications listed below, write a concise abstract "
    "for a new application paper that builds on and combines these works. "
    "Output ONLY the abstract text, without any heading or explanation."
)


def build_prompt(neighbours):
    """`neighbours` is a list of (title, abstract) for the papers the target cites."""
    sources = "\n\n".join(f"- Title: {t}\n  Abstract: {a}" for t, a in neighbours)
    user = (f"{INSTRUCTION}\n\n"
            f"Source publications:\n{sources or '(no neighbouring sources available)'}\n\n"
            f"Generated abstract:")
    return f"{SYSTEM} USER: {user} ASSISTANT:"


def encode(tokenizer, prompt, target):
    """-> (input_ids, labels). Labels are -100 over the prompt, so only the abstract trains."""
    prompt_ids = tokenizer(prompt).input_ids                      # carries BOS
    target_ids = tokenizer(target, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids


def fits(input_ids):
    """The prefix occupies real attention positions, so it counts against the context."""
    return len(input_ids) + NUM_PREFIX_TOKENS <= MAX_LENGTH


class Corpus:
    """Node text and reference lists, for turning a node_id into a training example."""

    def __init__(self, nodes=None, edges=None):
        nodes = nodes or os.path.join(JOINT_DIR, "nodes.csv")
        edges = edges or os.path.join(JOINT_DIR, "edges.csv")

        df = pd.read_csv(nodes, usecols=["node_id", "title", "abstract"]).set_index("node_id")
        self.title = df["title"].fillna("")
        self.abstract = df["abstract"].fillna("")
        self.refs = pd.read_csv(edges).groupby("src_node_id")["dst_node_id"].apply(list)

    def example(self, node_id):
        """-> (prompt, target_abstract) for one node."""
        refs = self.refs.get(node_id, [])
        return (build_prompt([(self.title[n], self.abstract[n]) for n in refs]),
                self.abstract[node_id])

    def out_degree(self, node_id):
        return len(self.refs.get(node_id, []))


def capped_example(tokenizer, corpus, embeddings, node_id):
    """-> (prompt, target, kept_refs) for one node, trimmed to fit the context.

    The single capped-prompt path: prepare_data_capped.py freezes node sets with it,
    train.py builds training prompts with it, and generate.py builds eval prompts with
    it. They must agree exactly — a node was selected on the assumption that these
    references are the ones shown, and the failure mode of a mismatched budget is silent
    (a fluent abstract written from the wrong sources, not an error).

    The budget leaves room for the prefix and for the abstract: the node's own abstract
    when it is longer than GENERATION_RESERVE, so the training sequence is complete, and
    never less than GENERATION_RESERVE, so generation has somewhere to write.
    """
    target = corpus.abstract[node_id]
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    reserve = max(GENERATION_RESERVE, len(target_ids) + 1)   # +1 for the EOS token
    budget = MAX_LENGTH - NUM_PREFIX_TOKENS - reserve

    all_refs = corpus.refs.get(node_id, [])

    def measure(kept):
        prompt = build_prompt([(corpus.title[n], corpus.abstract[n]) for n in kept])
        return len(tokenizer(prompt).input_ids)

    refs = fit_refs(all_refs, embeddings, measure, budget)
    return (build_prompt([(corpus.title[n], corpus.abstract[n]) for n in refs]),
            target, refs)


def collect(tokenizer, corpus, candidates, wanted, rng):
    """Walk shuffled candidates, keeping those that fit, until `wanted` are found."""
    pool = list(candidates)
    rng.shuffle(pool)
    kept, lengths, seen = [], [], 0
    for node_id in pool:
        seen += 1
        input_ids, _ = encode(tokenizer, *corpus.example(node_id))
        if fits(input_ids):
            kept.append(int(node_id))
            lengths.append(len(input_ids))
            if len(kept) == wanted:
                break
    return kept, lengths, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=BASE_MODEL, help="tokenizer to measure with")
    ap.add_argument("--nodes", default=os.path.join(JOINT_DIR, "nodes.csv"))
    ap.add_argument("--edges", default=os.path.join(JOINT_DIR, "edges.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=FROZEN_DIR)
    args = ap.parse_args()

    print(f"Loading tokenizer {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("Loading corpus …")
    corpus = Corpus(args.nodes, args.edges)

    with open(os.path.join(FROZEN_DIR, "splits_eligible.json")) as f:
        splits = json.load(f)

    rng = random.Random(args.seed)

    print(f"\nCollecting {TRAIN_SUBSET:,} training nodes …")
    train, train_len, train_seen = collect(tokenizer, corpus, splits["train"], TRAIN_SUBSET, rng)
    assert len(train) == TRAIN_SUBSET, f"only {len(train)} training nodes fit the context"
    print(f"  {len(train):,} kept from {train_seen:,} examined "
          f"({1 - len(train) / train_seen:.1%} overflowed)")

    print(f"\nCollecting {VALID_PER_DEGREE} validation nodes at each out-degree "
          f"{MIN_OUT_DEGREE}-{MAX_OUT_DEGREE} …")
    by_degree = {}
    for node_id in splits["valid"]:
        by_degree.setdefault(corpus.out_degree(node_id), []).append(node_id)

    valid, valid_len = [], []
    for degree in range(MIN_OUT_DEGREE, MAX_OUT_DEGREE + 1):
        kept, lengths, seen = collect(tokenizer, corpus, by_degree.get(degree, []),
                                      VALID_PER_DEGREE, rng)
        assert len(kept) == VALID_PER_DEGREE, \
            f"only {len(kept)} valid nodes at out-degree {degree} fit the context"
        valid += kept
        valid_len += lengths
        print(f"  degree {degree:>2}: {len(kept)} kept from {seen} examined, "
              f"tokens mean {sum(lengths) / len(lengths):,.0f}")

    # --- checks -----------------------------------------------------------
    with open(os.path.join(FROZEN_DIR, "eval_nodes.json")) as f:
        eval_nodes = json.load(f)
    assert len(set(train)) == len(train) and len(set(valid)) == len(valid), "duplicates"
    assert not set(train) & set(valid), "train and valid overlap"
    for name, ids in (("train", train), ("valid", valid)):
        assert not set(ids) & set(eval_nodes), f"{name} overlaps the frozen eval set"
        assert set(ids) <= set(splits[name]), f"{name} contains ineligible nodes"

    # --- write ------------------------------------------------------------
    path = os.path.join(args.out_dir, "training_sets.json")
    with open(path, "w") as f:
        json.dump({"train": sorted(train), "valid": sorted(valid)}, f)

    # --- report -----------------------------------------------------------
    for name, lengths in (("train", train_len), ("valid", valid_len)):
        lengths = sorted(lengths)
        print(f"\n{name}: {len(lengths):,} nodes, sequence length "
              f"mean {sum(lengths) / len(lengths):,.0f}, "
              f"median {lengths[len(lengths) // 2]:,}, "
              f"p99 {lengths[int(len(lengths) * 0.99)]:,}, max {lengths[-1]:,}")

    # The eval set is frozen and is not filtered — Stage 5 has to handle its overflow.
    over = sum(not fits(encode(tokenizer, *corpus.example(n))[0]) for n in eval_nodes)
    print(f"\neval set: {over}/{len(eval_nodes)} nodes overflow the context "
          f"({over / len(eval_nodes):.1%}) — left in place, Stage 5's problem")

    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
