"""
Download ogbn-arxiv Text-Attributed Graph (TA) dataset.

Saves to ./data/raw/ogbn_arxiv/:
  - nodes.csv         : node_id, mag_id, title, abstract, label, year
  - edges.csv         : src_node_id, dst_node_id
  - node_features.npy : (169343, 128) float32 word2vec embeddings
  - splits.json       : train / val / test node ID lists

Requirements:
    pip install ogb torch torch_geometric requests
"""

import os
import json
import gzip
import shutil
import requests
import numpy as np
import pandas as pd

# output directory
OUT_DIR = "./data/raw/ogbn_arxiv"
os.makedirs(OUT_DIR, exist_ok=True)

#  download graph structure, features, labels via OGB
print("Downloading graph structure via OGB …")

from ogb.nodeproppred import NodePropPredDataset

dataset = NodePropPredDataset(name="ogbn-arxiv", root=os.path.join(OUT_DIR, "_ogb_cache"))

graph, labels = dataset[0]
split_idx = dataset.get_idx_split()

edge_index  = graph["edge_index"]
node_feat   = graph["node_feat"]
node_year   = graph["node_year"].flatten()
labels_flat = labels.flatten()

print(f"  Nodes      : {node_feat.shape[0]:,}")
print(f"  Edges      : {edge_index.shape[1]:,}")
print(f"  Node feat  : {node_feat.shape}")

# download raw title / abstract text
print("Downloading title and abstract text")

TEXT_URL  = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
GZ_PATH   = os.path.join(OUT_DIR, "_titleabs.tsv.gz")
TSV_PATH  = os.path.join(OUT_DIR, "_titleabs.tsv")

if not os.path.exists(TSV_PATH):
    print(f"  Fetching {TEXT_URL} …")
    r = requests.get(TEXT_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(GZ_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    with gzip.open(GZ_PATH, "rb") as f_in, open(TSV_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(GZ_PATH)
    print("  Done.")
else:
    print("  Already downloaded, skipping.")

text_df = pd.read_csv(
    TSV_PATH,
    sep="\t",
    header=None,
    names=["mag_id", "title", "abstract"],
    skiprows=1,
)
print(f"  Text rows  : {len(text_df):,}")

# Load OGB node index → MAG paper ID mapping
MAP_PATH = os.path.join(OUT_DIR, "_ogb_cache", "ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz")
nodeidx2paperid = pd.read_csv(MAP_PATH, names=["node_id", "mag_id"], header=0)
print(f"  Mapping rows: {len(nodeidx2paperid):,}")

# Join node mapping with text on MAG paper ID
text_df["mag_id"] = pd.to_numeric(text_df["mag_id"], errors="coerce")
text_df = text_df.dropna(subset=["mag_id"])
text_df["mag_id"] = text_df["mag_id"].astype(np.int64)
nodeidx2paperid["mag_id"] = nodeidx2paperid["mag_id"].astype(np.int64)
merged = nodeidx2paperid.merge(text_df, on="mag_id", how="left")
merged = merged.sort_values("node_id").reset_index(drop=True)

assert len(merged) == node_feat.shape[0], (
    f"Row count mismatch after merge: merged={len(merged)}, graph={node_feat.shape[0]}"
)

nodes_df = pd.DataFrame({
    "node_id" : merged["node_id"].astype(np.int32),
    "mag_id"  : merged["mag_id"].values,
    "title"   : merged["title"].values,
    "abstract": merged["abstract"].values,
    "label"   : labels_flat.astype(np.int16),
    "year"    : node_year.astype(np.int16),
})

nodes_path = os.path.join(OUT_DIR, "nodes.csv")
nodes_df.to_csv(nodes_path, index=False)
print(f"  Saved → {nodes_path}  ({len(nodes_df):,} rows)")

# Step 4: build edges.csv
print("Step 4 – Building edges.csv")

edges_df = pd.DataFrame({
    "src_node_id": edge_index[0].astype(np.int32),
    "dst_node_id": edge_index[1].astype(np.int32),
})

edges_path = os.path.join(OUT_DIR, "edges.csv")
edges_df.to_csv(edges_path, index=False)
print(f"  Saved → {edges_path}  ({len(edges_df):,} edges)")

# save node_features.npy

feat_path = os.path.join(OUT_DIR, "node_features.npy")
np.save(feat_path, node_feat.astype(np.float32))
print(f"  Saved → {feat_path}  shape={node_feat.shape}")

# Step 6: save splits.json
splits = {
    "train": split_idx["train"].tolist(),
    "valid": split_idx["valid"].tolist(),
    "test" : split_idx["test"].tolist(),
}

splits_path = os.path.join(OUT_DIR, "splits.json")
with open(splits_path, "w") as f:
    json.dump(splits, f)

print(f"  Train nodes: {len(splits['train']):,}")
print(f"  Valid nodes: {len(splits['valid']):,}")
print(f"  Test  nodes: {len(splits['test']):,}")
print(f"  Saved → {splits_path}")
print("files saved to:", os.path.abspath(OUT_DIR))
