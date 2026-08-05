"""
Join the ogbn-arxiv dataset (data/raw/ogbn_arxiv/{nodes,edges}.csv) with the
S2-expanded subset (data/processed/s2/{papers_filtered,references_subset}.parquet)
into a single data/processed/joint/{nodes,edges}.csv pair, in the same CSV format
as the ogbn-arxiv files.

Node id scheme
--------------
Original ogbn-arxiv node_ids (0..169342) are kept exactly as-is (same
row order too), so data/processed/joint/nodes.csv stays index-compatible with
existing node_id-keyed artifacts (node_features.npy, splits.json, ...)
for the "core" subset. New S2-only nodes (papers referenced by ogbn-arxiv
nodes but not themselves ogbn-arxiv nodes) get fresh sequential ids
starting at 169343, assigned in sorted mag_id order for reproducibility.

Node attribute priority
------------------------
`label` is dropped (not needed downstream). A new boolean `core` column
marks whether the node is one of the original ogbn-arxiv nodes. Where a
mag_id exists in both datasets, ogbn-arxiv's own title/abstract/year are
kept (ogbn takes priority over S2) - the S2 copy of that paper is not
used at all.

Edge direction
--------------
Verified empirically (matching a sample of ogbn edges.csv through mag_id
into references_subset.parquet) that both files use the same citation
direction: (src, dst) / (citing_mag_id, cited_mag_id) both mean
"src/citing cites dst/cited". So no direction flip is needed - edges are
just unioned after mapping references_subset's mag_id pairs to node_ids
via the combined mag_id -> node_id mapping built above.

Usage:
    python build_joint.py
    python build_joint.py --ogbn-dir ../../data/raw/ogbn_arxiv
"""

import argparse
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ogbn-dir", default=os.path.join(DATA_DIR, "raw", "ogbn_arxiv"),
                     help="Directory containing ogbn-arxiv nodes.csv / edges.csv")
    ap.add_argument("--processed-dir", default=os.path.join(DATA_DIR, "processed", "s2"),
                     help="Directory containing papers_filtered.parquet / references_subset.parquet")
    ap.add_argument("--out-dir", default=os.path.join(DATA_DIR, "processed", "joint"),
                     help="Where to write the joint nodes.csv / edges.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ogbn_nodes_path = os.path.join(args.ogbn_dir, "nodes.csv")
    ogbn_edges_path = os.path.join(args.ogbn_dir, "edges.csv")
    papers_path = os.path.join(args.processed_dir, "papers_filtered.parquet")
    refs_path = os.path.join(args.processed_dir, "references_subset.parquet")

    print("[1/6] Loading inputs...")
    ogbn_nodes = pd.read_csv(ogbn_nodes_path)
    ogbn_edges = pd.read_csv(ogbn_edges_path)
    papers = pd.read_parquet(papers_path)
    refs = pd.read_parquet(refs_path)
    print(f"      ogbn nodes: {len(ogbn_nodes):,}  ogbn edges: {len(ogbn_edges):,}")
    print(f"      s2 papers_filtered: {len(papers):,}  s2 references_subset: {len(refs):,}")

    assert not ogbn_nodes["mag_id"].duplicated().any(), "duplicate mag_id in ogbn nodes.csv"
    assert not ogbn_nodes["node_id"].duplicated().any(), "duplicate node_id in ogbn nodes.csv"
    assert not papers["mag_id"].duplicated().any(), "duplicate mag_id in papers_filtered.parquet"

    print("[2/6] Building the core (ogbn-arxiv) node set...")
    core_mag_ids = set(ogbn_nodes["mag_id"])
    core_nodes = ogbn_nodes.drop(columns=["label"]).copy()
    core_nodes["core"] = True
    print(f"      {len(core_nodes):,} core nodes (node_id 0..{core_nodes['node_id'].max()})")

    print("[3/6] Building the S2-only node set (papers referenced but not in ogbn-arxiv)...")
    s2_only = papers[~papers["mag_id"].isin(core_mag_ids)].sort_values("mag_id").reset_index(drop=True)
    next_id = int(core_nodes["node_id"].max()) + 1
    s2_only_nodes = pd.DataFrame({
        "node_id": range(next_id, next_id + len(s2_only)),
        "mag_id": s2_only["mag_id"],
        "title": s2_only["title"],
        "abstract": s2_only["abstract"],
        "year": s2_only["year"],
        "core": False,
    })
    print(f"      {len(s2_only_nodes):,} S2-only nodes (node_id {next_id}..{next_id + len(s2_only_nodes) - 1})")

    print("[4/6] Combining node sets...")
    joint_nodes = pd.concat([core_nodes, s2_only_nodes], ignore_index=True)
    joint_nodes = joint_nodes[["node_id", "mag_id", "title", "abstract", "year", "core"]]
    joint_nodes["year"] = joint_nodes["year"].astype("Int64")
    assert not joint_nodes["mag_id"].duplicated().any(), "duplicate mag_id after combining node sets"
    assert not joint_nodes["node_id"].duplicated().any(), "duplicate node_id after combining node sets"
    print(f"      {len(joint_nodes):,} total nodes "
          f"({joint_nodes['core'].sum():,} core, {(~joint_nodes['core']).sum():,} S2-only)")

    print("[5/6] Building the joint edge set...")
    mag_to_node_id = pd.Series(joint_nodes["node_id"].values, index=joint_nodes["mag_id"].values)

    # ogbn-arxiv edges are already in the (unchanged) core node_id space - use as-is.
    ogbn_edge_pairs = ogbn_edges.rename(
        columns={"src_node_id": "src", "dst_node_id": "dst"})[["src", "dst"]]

    # references_subset edges are in mag_id space - map both endpoints to node_id.
    ref_src = refs["citing_mag_id"].map(mag_to_node_id)
    ref_dst = refs["cited_mag_id"].map(mag_to_node_id)
    assert ref_src.notna().all() and ref_dst.notna().all(), \
        "references_subset contains a mag_id not present in the joint node set"
    ref_edge_pairs = pd.DataFrame({"src": ref_src.astype("int64"), "dst": ref_dst.astype("int64")})

    joint_edges = pd.concat([ogbn_edge_pairs, ref_edge_pairs], ignore_index=True)
    n_before_dedup = len(joint_edges)
    joint_edges = joint_edges.drop_duplicates().reset_index(drop=True)
    joint_edges = joint_edges.rename(columns={"src": "src_node_id", "dst": "dst_node_id"})
    n_self_loops = (joint_edges["src_node_id"] == joint_edges["dst_node_id"]).sum()
    print(f"      {n_before_dedup:,} edges before dedup -> {len(joint_edges):,} distinct edges "
          f"({n_before_dedup - len(joint_edges):,} duplicates removed across the two sources, "
          f"{n_self_loops:,} self-loops kept)")

    print("[6/6] Writing data/processed/joint/nodes.csv / edges.csv...")
    nodes_out = os.path.join(args.out_dir, "nodes.csv")
    edges_out = os.path.join(args.out_dir, "edges.csv")
    joint_nodes.to_csv(nodes_out, index=False)
    joint_edges.to_csv(edges_out, index=False)
    print(f"      wrote {nodes_out} ({len(joint_nodes):,} rows)")
    print(f"      wrote {edges_out} ({len(joint_edges):,} rows)")


if __name__ == "__main__":
    main()
