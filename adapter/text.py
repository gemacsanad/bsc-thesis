"""
Stage 4 — prompts, labels, and the frozen training subsets.

Turns a node into the (prompt, target) pair the LLM is trained on, and freezes the
train and validation node lists so every run reads the same sets.

Prompt
------
Vicuna-v1.1 chat format, carried over verbatim from `scripts/generate_test_csv.py` so
the results stay comparable with the earlier Vicuna and GPT baselines in `test_data/`.

The target paper's **own title is not in the prompt** — the model sees only the titles and
abstracts of the papers it cites, and has to infer the topic from the neighbourhood alone.
Loss is computed on the target abstract only; the prompt is masked with -100.

Subsets
-------
Nodes whose prompt + target + prefix exceeds the 4,096-token context are dropped, so every
training sequence is complete. Candidates are shuffled and walked until enough survivors are
collected, which is distributionally the same as sampling from the fitting set but tokenises
a few thousand nodes instead of all 48,000.

train  — TRAIN_SUBSET nodes from the eligible train split
valid  — VALID_PER_DEGREE nodes at each out-degree 5-15, matching how the eval set was built

The 495-node evaluation set in `eval_nodes.json` is **not** touched; its overflow rate is
reported so Stage 5 knows what it is dealing with.

Usage:
    python adapter/text.py          # freeze the subsets (needs the Vicuna tokenizer)
"""

import argparse
import json
import os
import random

import pandas as pd
from transformers import AutoTokenizer

from config import (MAX_LENGTH, MAX_OUT_DEGREE, MIN_OUT_DEGREE, NUM_PREFIX_TOKENS,
                    ROOT, SCRIPT_DIR, TRAIN_SUBSET, VALID_PER_DEGREE)

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
        nodes = nodes or os.path.join(ROOT, "joint_dataset", "nodes.csv")
        edges = edges or os.path.join(ROOT, "joint_dataset", "edges.csv")

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
    ap.add_argument("--model", default="lmsys/vicuna-7b-v1.5", help="tokenizer to measure with")
    ap.add_argument("--nodes", default=os.path.join(ROOT, "joint_dataset", "nodes.csv"))
    ap.add_argument("--edges", default=os.path.join(ROOT, "joint_dataset", "edges.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=SCRIPT_DIR)
    args = ap.parse_args()

    print(f"Loading tokenizer {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("Loading corpus …")
    corpus = Corpus(args.nodes, args.edges)

    with open(os.path.join(SCRIPT_DIR, "splits_eligible.json")) as f:
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
    with open(os.path.join(SCRIPT_DIR, "eval_nodes.json")) as f:
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
