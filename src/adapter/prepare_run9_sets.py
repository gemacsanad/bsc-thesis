"""
Run 9 — the node set the embedding -> abstract decoder is trained on.

Every train-split core paper, instead of the 7,750 sampled into training_sets_capped.json.
Those 7,750 were sampled to a context budget and an out-degree quota, both inherited from the
text arm; the decoder reads one embedding, so neither applies. The step-4 oracle saw 7,750
papers ten times each; run 9 sees ~90k papers about once, at roughly the same compute, so the
run measures variety rather than more training. See docs/run9_embedding_decoder.md.

Removed, and counted:
  * the 945 eval and 806 validation seeds (none are train-split anyway; asserted, not assumed)
  * abstracts under 30 words — rank.MIN_CANDIDATE_WORDS, the stub threshold run 8 already uses
  * near-duplicates of an eval paper: same normalised title, or raw voyage cosine > 0.95 to
    one. A training target that IS an eval paper under another id would be a leak the oracle
    arm could not be blamed for.

The validation nodes are kept from training_sets_capped.json unchanged, so the canary and the
identifiability set are the papers every earlier run validated on.

CPU only, ~1-2 minutes, ~0.6 GB: the 90k embedding rows are read through a memmap.

Usage:
    cd src/adapter
    python prepare_run9_sets.py            # writes data/frozen/training_sets_run9.json
"""

import json
import os
import re

import numpy as np
import pandas as pd

from config import FROZEN_DIR, JOINT_DIR, RAW_NODE_EMBEDDINGS
from rank import MIN_CANDIDATE_WORDS

DUPLICATE_COSINE = 0.95
OUT = os.path.join(FROZEN_DIR, "training_sets_run9.json")


def normalised(title):
    return re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()


def main():
    with open(os.path.join(JOINT_DIR, "splits.json")) as f:
        splits = json.load(f)
    with open(os.path.join(FROZEN_DIR, "eval_nodes_capped.json")) as f:
        eval_nodes = json.load(f)
    with open(os.path.join(FROZEN_DIR, "training_sets_capped.json")) as f:
        valid = json.load(f)["valid"]

    held = set(eval_nodes) | set(valid)
    train = [n for n in splits["train"] if n not in held]
    assert len(train) == len(splits["train"]), "an eval or validation seed is in the train split"

    # Only the rows needed: train-split papers and the eval papers they are checked against.
    keep = set(train) | set(eval_nodes)
    parts = [chunk[chunk.node_id.isin(keep)] for chunk in
             pd.read_csv(os.path.join(JOINT_DIR, "nodes.csv"),
                         usecols=["node_id", "title", "abstract"], chunksize=50_000)]
    nodes = pd.concat(parts).set_index("node_id")

    words = nodes["abstract"].reindex(train).fillna("").str.split().str.len()
    stubs = set(words.index[words < MIN_CANDIDATE_WORDS])

    eval_titles = {normalised(t) for t in nodes["title"].reindex(eval_nodes)}
    same_title = {n for n in train if normalised(nodes["title"].get(n, "")) in eval_titles}

    x = np.load(RAW_NODE_EMBEDDINGS, mmap_mode="r")
    targets = np.asarray(x[eval_nodes], dtype=np.float32)
    close = set()
    for i in range(0, len(train), 20_000):
        block = train[i:i + 20_000]
        similarity = np.asarray(x[block], dtype=np.float32) @ targets.T
        close |= {n for n, s in zip(block, similarity.max(axis=1)) if s > DUPLICATE_COSINE}

    dropped = stubs | same_title | close
    kept = [n for n in train if n not in dropped]
    print(f"train-split core papers   {len(train):,}")
    print(f"  under {MIN_CANDIDATE_WORDS} words          {len(stubs):,}")
    print(f"  same title as an eval paper  {len(same_title):,}")
    print(f"  voyage cosine > {DUPLICATE_COSINE} to one  {len(close):,}")
    print(f"kept for training         {len(kept):,}  (training_sets_capped.json had 7,750)")
    print(f"validation (unchanged)    {len(valid):,}")

    with open(OUT, "w") as f:
        json.dump({"train": kept, "valid": valid}, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
