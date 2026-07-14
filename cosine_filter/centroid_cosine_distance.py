"""
For each test node (is_test=1), compute how far each of its reference nodes
(the papers it cites) sits from the centroid of that citation set, in cosine
distance.

Steps per test node:
  1. Collect all reference node IDs  (dst_node_id where src = test_node)
  2. Compute the centroid of those reference embeddings (mean vector)
  3. Measure the cosine distance between each reference and the centroid
  4. Write the distance to edges_subset.csv on the matching (test_node, ref) edge

Distance definition:
    cosine_distance = (1 - cosine_similarity(candidate, centroid)) / 2

Since embeddings are L2-normalised, cosine_similarity lies in [-1, 1], so the
distance lies in [0, 1]: 0 means the candidate points in the same direction as
the centroid, 1 means it points in the exact opposite direction.
    column: centroid_cosine_distance

Usage:
    python cosine_filter/centroid_cosine_distance.py
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'ogbn_arxiv_dataset')
NODES_PATH    = os.path.join(DATA_DIR, 'nodes_subset.csv')
EDGES_PATH    = os.path.join(DATA_DIR, 'edges_subset.csv')
FEATURES_PATH = os.path.join(DATA_DIR, 'node_subset_features.npy')

DISTANCE_COLUMN = 'centroid_cosine_distance'


def main():
    nodes_df = pd.read_csv(NODES_PATH)
    edges_df = pd.read_csv(EDGES_PATH)
    features = np.load(FEATURES_PATH)

    print(f"Nodes    : {len(nodes_df)}")
    print(f"Edges    : {len(edges_df)}")
    print(f"Features : {features.shape}\n")

    # node_id -> row index in features matrix
    node_id_to_idx = {int(nid): i for i, nid in enumerate(nodes_df['node_id'])}

    # L2-normalise once so cosine similarity = dot product
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = (features / norms).astype(np.float32)

    if DISTANCE_COLUMN not in edges_df.columns:
        edges_df[DISTANCE_COLUMN] = np.nan

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

        # centroid of the reference set, re-normalised to a unit direction
        centroid = emb.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            centroid_unit = centroid
        else:
            centroid_unit = centroid / centroid_norm

        # cosine similarity of each candidate to the centroid direction
        sims = emb @ centroid_unit  # (N,), in [-1, 1]

        # map to a [0, 1] distance: 0 = aligned with centroid, 1 = opposite
        distances = np.abs((1.0 - sims) / 2.0).astype(np.float32)

        # write distances to matching edges
        for ref_id, pos in zip(ref_ids, range(len(ref_ids))):
            edge_mask = src_mask & (edges_df['dst_node_id'] == ref_id)
            edges_df.loc[edge_mask, DISTANCE_COLUMN] = float(distances[pos])

        print(
            f"  test_node {test_id}: {len(ref_ids)} refs | "
            f"distance=[{distances.min():.3f},{distances.max():.3f}]",
            flush=True,
        )

    edges_df.to_csv(EDGES_PATH, index=False)
    filled = edges_df[DISTANCE_COLUMN].notna().sum()
    print(f"\n{DISTANCE_COLUMN}: {filled} edges scored")
    print(f"\nSaved → {EDGES_PATH}")


if __name__ == '__main__':
    main()
