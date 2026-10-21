"""
Freeze which books each split may use: the children's-books counterpart of
src/adapter/prepare_run9_sets.py. CS-TAG has no fixed split, so this draws one.

Every set keeps only books whose description + EOS fits MAX_TARGET_TOKENS Qwen3 tokens
(= src/adapter/config.py MAX_NEW_TOKENS for the books): a longer target could not be generated
in full, and the longest (27k tokens) would not fit the 4,096-token context to train on.

Evaluation (945) and validation (806) books, the sizes of run 11, are drawn at random
(seed 42) from the books that can be judged fairly:
  * a description of >= 80 words that is not a seller note;
  * >= 3 co-purchased neighbours, so there is a neighbourhood to read;
  * a description no other book shares (build_children.py, `shared_description`);
  * no neighbour with the same normalised title: another edition of the book would hand
    the model its own text.

Training is every other book, as in run 9, minus:
  * descriptions under 30 words (rank.MIN_CANDIDATE_WORDS), seller notes and shared
    descriptions;
  * books with the same normalised title as an evaluation or validation book;
  * books whose voyage embedding has cosine > 0.95 to one. This needs step 3's
    embeddings: without them the filter is skipped, the output says so, and the script has
    to be re-run after embedding. The evaluation and validation books do not depend on it,
    so re-running leaves them unchanged.

Output (versioned, like data/frozen/):
    data/frozen/children_books/eval_nodes.json      [node_id, ...]
    data/frozen/children_books/training_sets.json   {"train": [...], "valid": [...],
                                                     "cosine_filtered": bool}

Needs the Qwen3 tokenizer, so it runs in the DyTAG-qwen env (transformers 4.51).

Usage:
    python prepare_sets.py
"""

import json
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from build_children import DATA_DIR, ROOT, normalised

FROZEN_DIR = os.path.join(ROOT, "data", "frozen", "children_books")
EMBEDDINGS = os.path.join(DATA_DIR, "node_embeddings.npy")
SEED = 42
N_EVAL, N_VALID = 945, 806
MIN_EVAL_WORDS, MIN_NEIGHBOURS = 80, 3
MIN_TRAIN_WORDS = 30            # rank.MIN_CANDIDATE_WORDS
DUPLICATE_COSINE = 0.95
BASE_MODEL = "Qwen/Qwen3-14B"
MAX_TARGET_TOKENS = 1024        # description + EOS


def main():
    nodes = pd.read_csv(os.path.join(DATA_DIR, "nodes.csv"), keep_default_na=False)
    edges = pd.read_csv(os.path.join(DATA_DIR, "edges.csv"))
    assert (nodes["node_id"] == range(len(nodes))).all()
    title = nodes["title"].map(normalised).to_numpy()
    deg = edges.groupby("src_node_id").size().reindex(nodes["node_id"], fill_value=0).to_numpy()

    same_title_edge = title[edges["src_node_id"]] == title[edges["dst_node_id"]]
    has_edition = np.zeros(len(nodes), dtype=bool)
    has_edition[edges["src_node_id"][same_title_edge]] = True

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokens = np.array([len(ids) + 1 for ids in tokenizer(nodes["abstract"].tolist(),
                                                         add_special_tokens=False).input_ids])
    fits = tokens <= MAX_TARGET_TOKENS
    clean = (~nodes["seller_note"].to_numpy() & ~nodes["shared_description"].to_numpy()
             & fits)
    words = nodes["words"].to_numpy()
    pool = np.flatnonzero(clean & (words >= MIN_EVAL_WORDS) & (deg >= MIN_NEIGHBOURS)
                          & ~has_edition)
    drawn = np.random.default_rng(SEED).choice(pool, N_EVAL + N_VALID, replace=False)
    eval_nodes = sorted(drawn[:N_EVAL].tolist())
    valid = sorted(drawn[N_EVAL:].tolist())

    held = set(eval_nodes) | set(valid)
    held_titles = set(title[list(held)])
    rest = np.array([n for n in range(len(nodes)) if n not in held])
    short = rest[words[rest] < MIN_TRAIN_WORDS]
    unclean = rest[~clean[rest]]
    same_title = rest[np.isin(title[rest], list(held_titles))]

    cosine_filtered = os.path.exists(EMBEDDINGS)
    close = np.array([], dtype=int)
    if cosine_filtered:
        x = np.load(EMBEDDINGS, mmap_mode="r")
        assert x.shape[0] == len(nodes), "embeddings do not match nodes.csv"
        targets = np.asarray(x[sorted(held)], dtype=np.float32)
        best = np.concatenate([(np.asarray(x[rest[i:i + 20_000]], dtype=np.float32) @ targets.T)
                               .max(axis=1) for i in range(0, len(rest), 20_000)])
        close = rest[best > DUPLICATE_COSINE]

    dropped = set(short) | set(unclean) | set(same_title) | set(close)
    train = [int(n) for n in rest if n not in dropped]

    print(f"over {MAX_TARGET_TOKENS} tokens          {(~fits).sum():,} books "
          f"({(~fits & (words >= MIN_EVAL_WORDS)).sum():,} of them >= {MIN_EVAL_WORDS} words)")
    print(f"eligible for eval/valid   {len(pool):,}  (of {len(nodes):,} books; "
          f"{has_edition.sum():,} dropped for a same-title neighbour)")
    print(f"evaluation                {len(eval_nodes):,}")
    print(f"validation                {len(valid):,}")
    print(f"other books               {len(rest):,}")
    print(f"  under {MIN_TRAIN_WORDS} words            {len(short):,}")
    print(f"  seller note / shared / too long {len(unclean):,}")
    print(f"  same title as held-out  {len(same_title):,}")
    if cosine_filtered:
        print(f"  cosine > {DUPLICATE_COSINE} to held-out {len(close):,}")
    else:
        print(f"  cosine > {DUPLICATE_COSINE} to held-out SKIPPED: no {EMBEDDINGS}; "
              f"re-run after step 3")
    print(f"training                  {len(train):,}  "
          f"({(words[train] >= MIN_EVAL_WORDS).mean():.1%} >= {MIN_EVAL_WORDS} words, "
          f"{(deg[train] >= 1).sum():,} with a neighbour for the GNN)")
    for name, s in [("eval", eval_nodes), ("train", train)]:
        print(f"  {name:5s} words median {np.median(words[s]):.0f}, "
              f"tokens median {np.median(tokens[s]):.0f}, max {tokens[s].max()}, "
              f"degree median {np.median(deg[s]):.0f}")

    os.makedirs(FROZEN_DIR, exist_ok=True)
    with open(os.path.join(FROZEN_DIR, "eval_nodes.json"), "w") as f:
        json.dump(eval_nodes, f)
    with open(os.path.join(FROZEN_DIR, "training_sets.json"), "w") as f:
        json.dump({"train": train, "valid": valid, "cosine_filtered": cosine_filtered}, f)
    print(f"wrote {FROZEN_DIR}/{{eval_nodes,training_sets}}.json")


if __name__ == "__main__":
    main()
