"""
For each test node (is_test=1), compute a relevance score for each of its
reference nodes (the papers it cites) based on how similar each reference is
to all other references in the same citation set.

Steps per test node:
  1. Collect all reference node IDs  (dst_node_id where src = test_node)
  2. Build an N×N cosine similarity matrix between those references
  3. Sum each column  → raw score per reference node
  4. Normalise with one or more methods (see --normalization)
  5. Write score(s) to edges_subset.csv on the matching (test_node, ref) edge

Normalization options:
  min_max  : (x - min) / (max - min)                → [0, 1]
             column: cosine_score_min_max
  mean     : col_sum / N  (average cosine sim with all other refs) → [0, 1]
             column: cosine_score_mean
  softmax  : exp(x) / sum(exp(x))                   → (0, 1), sums to 1
             column: cosine_score_softmax
  all      : compute all three (default)

Usage:
    python cosine_filter/reference_cosine_score.py
    python cosine_filter/reference_cosine_score.py --normalization min_max
    python cosine_filter/reference_cosine_score.py --normalization mean
    python cosine_filter/reference_cosine_score.py --normalization softmax
    python cosine_filter/reference_cosine_score.py --normalization all
"""

import argparse
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'ogbn_arxiv_dataset')
NODES_PATH    = os.path.join(DATA_DIR, 'nodes_subset.csv')
EDGES_PATH    = os.path.join(DATA_DIR, 'edges_subset.csv')
FEATURES_PATH = os.path.join(DATA_DIR, 'node_subset_features.npy')

NORM_COLUMNS = {
    'min_max': 'cosine_score_min_max',
    'mean':    'cosine_score_mean',
    'softmax': 'cosine_score_softmax',
}


# ---------------------------------------------------------------------------
# Normalisation functions  (all operate on a 1-D numpy array)
# ---------------------------------------------------------------------------

def norm_min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.ones_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def norm_mean(x: np.ndarray) -> np.ndarray:
    """col_sum / N — average cosine similarity with all other references."""
    n = len(x)
    if n == 0:
        return x.astype(np.float32)
    return (x / n).astype(np.float32)


def norm_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: exp(x - max) / sum(exp(x - max))."""
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float32)


NORM_FNS = {
    'min_max': norm_min_max,
    'mean':    norm_mean,
    'softmax': norm_softmax,
}


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--normalization',
        choices=['min_max', 'mean', 'softmax', 'all'],
        default='all',
        help='Normalisation method to apply (default: all)',
    )
    args = parser.parse_args()

    methods = list(NORM_FNS.keys()) if args.normalization == 'all' else [args.normalization]

    nodes_df = pd.read_csv(NODES_PATH)
    edges_df = pd.read_csv(EDGES_PATH)
    features = np.load(FEATURES_PATH)

    print(f"Nodes    : {len(nodes_df)}")
    print(f"Edges    : {len(edges_df)}")
    print(f"Features : {features.shape}")
    print(f"Methods  : {methods}\n")

    # node_id -> row index in features matrix
    node_id_to_idx = {int(nid): i for i, nid in enumerate(nodes_df['node_id'])}

    # L2-normalise once so cosine similarity = dot product
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = (features / norms).astype(np.float32)

    # initialise output columns
    for method in methods:
        col = NORM_COLUMNS[method]
        if col not in edges_df.columns:
            edges_df[col] = np.nan

    test_node_ids = nodes_df.loc[nodes_df['is_test'] == 1, 'node_id'].astype(int).tolist()
    print(f"Test nodes: {len(test_node_ids)}")

    for test_id in test_node_ids:
        src_mask = edges_df['src_node_id'] == test_id
        ref_ids  = edges_df.loc[src_mask, 'dst_node_id'].astype(int).tolist()

        if len(ref_ids) == 0:
            continue

        # embeddings for reference nodes, shape (N, D)
        idx = [node_id_to_idx[r] for r in ref_ids]
        emb = normed[idx]

        # N×N cosine similarity matrix → column sums = raw score
        col_sums = (emb @ emb.T).sum(axis=0)  # (N,)

        scores_by_method = {m: NORM_FNS[m](col_sums) for m in methods}

        # write scores to matching edges
        for ref_id, pos in zip(ref_ids, range(len(ref_ids))):
            edge_mask = src_mask & (edges_df['dst_node_id'] == ref_id)
            for method in methods:
                edges_df.loc[edge_mask, NORM_COLUMNS[method]] = float(scores_by_method[method][pos])

        summary = '  '.join(
            f"{m}=[{scores_by_method[m].min():.3f},{scores_by_method[m].max():.3f}]"
            for m in methods
        )
        print(f"  test_node {test_id}: {len(ref_ids)} refs | {summary}", flush=True)

    edges_df.to_csv(EDGES_PATH, index=False)
    for method in methods:
        col = NORM_COLUMNS[method]
        filled = edges_df[col].notna().sum()
        print(f"\n{col}: {filled} edges scored")
    print(f"\nSaved → {EDGES_PATH}")


if __name__ == '__main__':
    main()
