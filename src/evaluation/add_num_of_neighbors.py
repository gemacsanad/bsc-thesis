"""
add_num_of_neighbors.py

Fills the num_of_neighbors column of already-evaluated result CSVs.

evaluate_results.py joins num_of_neighbors from data/processed/ogbn_subset/nodes_subset.csv,
which only covers the old 200-node test set — for the 495-node adapter eval set the
join produced an all-empty column. This script recomputes it from the joint dataset,
using the same definition as src/adapter/prepare_data.py: out-degree, i.e. the number of
outgoing edges in data/processed/joint/edges.csv (src cites dst, so out-edges are the
paper's own references).

The eval set was frozen at 45 nodes per out-degree from 5 to 15, so the resulting
distribution is checked against that and the script refuses to write if it disagrees.

Usage:
    python scripts/add_num_of_neighbors.py
    python scripts/add_num_of_neighbors.py path/to/evaluated_results.csv ...
"""

import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FILES = [
    os.path.join(ROOT, "experiments", "2026-08-04_vicuna-baseline", "scored.csv"),
    os.path.join(ROOT, "experiments", "2026-08-04_adapter-ep2", "scored.csv"),
]
EXPECTED_PER_DEGREE = 45
MIN_OUT_DEGREE, MAX_OUT_DEGREE = 5, 15

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("files", nargs="*", default=DEFAULT_FILES,
                help="Result CSVs to update in place (default: the two adapter eval CSVs)")
ap.add_argument("--edges", default=os.path.join(ROOT, "data", "processed", "joint", "edges.csv"))
args = ap.parse_args()

print(f"Loading {args.edges} …")
edges = pd.read_csv(args.edges, usecols=["src_node_id"])
out_degree = edges.groupby("src_node_id").size()
print(f"  {len(edges):,} edges, {len(out_degree):,} nodes with at least one out-edge")

for path in args.files:
    df = pd.read_csv(path)
    degrees = df["node_id"].map(out_degree).fillna(0).astype(int)

    missing = (degrees == 0).sum()
    counts = degrees.value_counts().sort_index()
    print(f"\n{path}")
    print(f"  {len(df)} rows | {missing} nodes with no out-edges")
    print("  distribution: " + ", ".join(f"{d}:{c}" for d, c in counts.items()))

    expected = {d: EXPECTED_PER_DEGREE for d in range(MIN_OUT_DEGREE, MAX_OUT_DEGREE + 1)}
    if counts.to_dict() != expected:
        print(f"  MISMATCH — expected {EXPECTED_PER_DEGREE} nodes at each degree "
              f"{MIN_OUT_DEGREE}-{MAX_OUT_DEGREE}. Not written.")
        continue

    df["num_of_neighbors"] = degrees
    df.to_csv(path, index=False)
    print("  num_of_neighbors written.")
