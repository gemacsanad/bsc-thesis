"""
Stage 1 — data prep for the graph-conditioned adapter.

Defines which nodes of the joint dataset are usable for adapter training and which
are held out for evaluation, and writes both to disk so every later stage reads
the same fixed sets.

Eligible node
-------------
core == True  and  5 <= out_degree <= 15, where out_degree counts outgoing edges
in data/processed/joint/edges.csv (src cites dst, so out-edges are the paper's own
references — the temporally honest signal). Intersected with splits.json, which
only covers the 169,343 core nodes.

Evaluation nodes
----------------
495 nodes drawn from the eligible test nodes, 45 at each out-degree from 5 to 15,
so quality can be read against neighbour count without the degree bins differing
in size. Sampling is seeded and the frozen list is committed, so the set never
moves between runs.

The earlier 200-node set in data/processed/ogbn_subset/nodes_subset.csv is not reused:
it was stratified over ogbn-arxiv out-degree, which counts only citations between
arXiv CS papers. the joint dataset adds Semantic Scholar bibliographies, under which
those 200 nodes span out-degree 1-65 and half of them overflow the context window.

Caveat carried by this set: an S2 bibliography exists for only some core nodes, and
coverage rises with out-degree (39% at degree 5, 65% at degree 15). Nodes without one
show only their arXiv-CS-internal references, so a low out-degree can mean either a
short bibliography or an unseen one. The eval set is not filtered on coverage; the
per-degree breakdown is printed at freeze time so the confound stays visible.

Outputs
-------
adapter/eval_nodes.json      — list of 495 node_ids
adapter/splits_eligible.json — {"train": [...], "valid": [...], "test": [...]}

Usage:
    python adapter/prepare_data.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from config import (EVAL_PER_DEGREE, FROZEN_DIR, JOINT_DIR, MAX_OUT_DEGREE,
                    MIN_OUT_DEGREE, OGBN_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default=os.path.join(JOINT_DIR, "nodes.csv"))
    ap.add_argument("--edges", default=os.path.join(JOINT_DIR, "edges.csv"))
    ap.add_argument("--splits", default=os.path.join(JOINT_DIR, "splits.json"))
    ap.add_argument("--ogbn-edges", default=os.path.join(OGBN_DIR, "edges.csv"),
                    help="Only used to report S2 bibliography coverage of the eval set")
    ap.add_argument("--seed", type=int, default=42, help="Seed for the eval node sample")
    ap.add_argument("--out-dir", default=FROZEN_DIR)
    args = ap.parse_args()

    print("Loading the joint dataset …")
    nodes = pd.read_csv(args.nodes, usecols=["node_id", "core"])
    edges = pd.read_csv(args.edges)
    print(f"  {len(nodes):,} nodes, {len(edges):,} edges")

    out_degree = edges.groupby("src_node_id").size()
    nodes["out_degree"] = nodes["node_id"].map(out_degree).fillna(0).astype(int)

    eligible = set(nodes.loc[
        nodes["core"]
        & nodes["out_degree"].between(MIN_OUT_DEGREE, MAX_OUT_DEGREE),
        "node_id",
    ])
    print(f"  {len(eligible):,} eligible (core, {MIN_OUT_DEGREE}-{MAX_OUT_DEGREE} out-edges)")

    with open(args.splits) as f:
        splits = json.load(f)
    splits_eligible = {name: sorted(eligible & set(ids)) for name, ids in splits.items()}

    # --- freeze the evaluation set ----------------------------------------
    degree = nodes.set_index("node_id")["out_degree"]
    rng = np.random.default_rng(args.seed)
    eval_nodes = []
    for d in range(MIN_OUT_DEGREE, MAX_OUT_DEGREE + 1):
        # sorted() so the candidate order — and thus the sample — is reproducible
        candidates = sorted(n for n in splits_eligible["test"] if degree[n] == d)
        assert len(candidates) >= EVAL_PER_DEGREE, \
            f"only {len(candidates)} test nodes at out-degree {d}, need {EVAL_PER_DEGREE}"
        eval_nodes += [int(n) for n in rng.choice(candidates, EVAL_PER_DEGREE, replace=False)]
    eval_nodes = sorted(eval_nodes)

    # --- checks -----------------------------------------------------------
    core_ids = set(nodes.loc[nodes["core"], "node_id"])
    assert len(eval_nodes) == len(set(eval_nodes)), "duplicate eval nodes"
    assert set(eval_nodes) <= core_ids, "some eval nodes are not core nodes"
    assert set(eval_nodes) <= set(splits["test"]), "some eval nodes are not in the test split"
    assert set(eval_nodes) <= eligible, "some eval nodes are not eligible"
    for name in ("train", "valid"):
        overlap = set(eval_nodes) & set(splits_eligible[name])
        assert not overlap, f"{len(overlap)} eval nodes leaked into {name}"

    # --- write ------------------------------------------------------------
    eval_path = os.path.join(args.out_dir, "eval_nodes.json")
    splits_path = os.path.join(args.out_dir, "splits_eligible.json")
    with open(eval_path, "w") as f:
        json.dump(eval_nodes, f)
    with open(splits_path, "w") as f:
        json.dump(splits_eligible, f)

    # --- report -----------------------------------------------------------
    print("\nEligible nodes per split:")
    for name in ("train", "valid", "test"):
        print(f"  {name:<6} {len(splits_eligible[name]):>7,}")

    # An S2 bibliography adds references beyond the arXiv-CS-internal ones ogbn records.
    ogbn_degree = pd.read_csv(args.ogbn_edges).groupby("src_node_id").size()
    covered = {n: degree[n] > ogbn_degree.get(n, 0) for n in eval_nodes}

    print(f"\nEval nodes: {len(eval_nodes)} "
          f"({EVAL_PER_DEGREE} per out-degree, seed {args.seed}, no overlap with train or valid)")
    print("  degree  nodes  S2 bibliography")
    for d in range(MIN_OUT_DEGREE, MAX_OUT_DEGREE + 1):
        at_d = [n for n in eval_nodes if degree[n] == d]
        n_cov = sum(covered[n] for n in at_d)
        print(f"  {d:>6}  {len(at_d):>5}  {n_cov:>3}/{len(at_d)} ({n_cov / len(at_d):.0%})")
    total_cov = sum(covered.values())
    print(f"  {'total':>6}  {len(eval_nodes):>5}  {total_cov:>3}/{len(eval_nodes)} "
          f"({total_cov / len(eval_nodes):.0%})")

    print(f"\nWrote {eval_path}")
    print(f"Wrote {splits_path}")


if __name__ == "__main__":
    main()
