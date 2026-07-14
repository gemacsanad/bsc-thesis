"""
Add cosine_similarity column to edges_subset.csv.

For each edge (src, dst), computes the cosine similarity between the
voyage-4 abstract embeddings of the two nodes.

Embeddings in node_subset_features.npy are ordered by row position in
nodes_subset.csv, so a node_id -> row-index mapping is built first.

Usage:
    python cosine_filter/cosine_similarity.py
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'ogbn_arxiv_dataset')
NODES_PATH = os.path.join(DATA_DIR, 'nodes_subset.csv')
EDGES_PATH = os.path.join(DATA_DIR, 'edges_subset.csv')
FEATURES_PATH = os.path.join(DATA_DIR, 'node_subset_features.npy')


def main():
    nodes_df = pd.read_csv(NODES_PATH)
    edges_df = pd.read_csv(EDGES_PATH)
    features = np.load(FEATURES_PATH)

    print(f"Nodes  : {len(nodes_df)}")
    print(f"Edges  : {len(edges_df)}")
    print(f"Features shape: {features.shape}")

    # node_id -> row index in the features matrix
    node_id_to_idx = {int(nid): i for i, nid in enumerate(nodes_df['node_id'])}

    # L2-normalise all embeddings once so cosine similarity = dot product
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    normed = features / norms

    src_ids = edges_df['src_node_id'].to_numpy(dtype=int)
    dst_ids = edges_df['dst_node_id'].to_numpy(dtype=int)

    src_idx = np.array([node_id_to_idx[i] for i in src_ids])
    dst_idx = np.array([node_id_to_idx[i] for i in dst_ids])

    similarities = (normed[src_idx] * normed[dst_idx]).sum(axis=1)

    edges_df['cosine_similarity'] = similarities.astype(np.float32)
    edges_df.to_csv(EDGES_PATH, index=False)

    print(f"cosine_similarity — min: {similarities.min():.4f}  "
          f"mean: {similarities.mean():.4f}  max: {similarities.max():.4f}")
    print(f"Saved → {EDGES_PATH}")


if __name__ == '__main__':
    main()
