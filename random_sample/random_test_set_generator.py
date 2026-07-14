import json
import numpy as np
import pandas as pd

with open("./ogbn_arxiv_dataset/splits.json") as f:
    splits = json.load(f)
test_ids = set(splits["test"])

edges_df = pd.read_csv("./ogbn_arxiv_dataset/edges.csv")
nodes_df = pd.read_csv("./ogbn_arxiv_dataset/nodes.csv")

# Out-degree for every node that has at least one outgoing edge
all_out_degree = edges_df.groupby("src_node_id").size().rename("num_of_neighbors")

# Build a frame of test nodes with their out-degree (0 if not in edges at all)
test_degree_df = pd.DataFrame({"node_id": list(test_ids)})
test_degree_df = test_degree_df.join(all_out_degree, on="node_id")
test_degree_df["num_of_neighbors"] = test_degree_df["num_of_neighbors"].fillna(0).astype(int)

# Keep only degrees 1-20 and sample up to 10 per group
filtered = test_degree_df[(test_degree_df["num_of_neighbors"] >= 1) & (test_degree_df["num_of_neighbors"] <= 20)]
sampled = (
    filtered.groupby("num_of_neighbors", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), 10), random_state=42))
    .reset_index(drop=True)
)

sampled_node_ids = set(sampled["node_id"])

# Build random edges: for each sampled node keep the same out-degree but pick
# random destination nodes from the full graph (excluding the source itself).
all_node_ids = nodes_df["node_id"].values
rng = np.random.default_rng(42)

random_edge_rows = []
for _, row in sampled.iterrows():
    src = int(row["node_id"])
    n   = int(row["num_of_neighbors"])
    pool = all_node_ids[all_node_ids != src]
    dsts = rng.choice(pool, size=n, replace=False)
    for dst in dsts:
        random_edge_rows.append({"src_node_id": src, "dst_node_id": int(dst)})

edges_subset = pd.DataFrame(random_edge_rows)

# All nodes to include: sampled + their random neighbours
all_subset_ids = sampled_node_ids | set(edges_subset["dst_node_id"])

# Build single nodes CSV with is_test flag
nodes_subset = (
    nodes_df[nodes_df["node_id"].isin(all_subset_ids)][["node_id", "mag_id", "title", "abstract"]]
    .copy()
    .join(all_out_degree, on="node_id")
)
nodes_subset["num_of_neighbors"] = nodes_subset["num_of_neighbors"].fillna(0).astype(int)
nodes_subset["is_test"] = nodes_subset["node_id"].isin(sampled_node_ids).astype(int)
nodes_subset = nodes_subset.sort_values("node_id").reset_index(drop=True)

# Save
nodes_subset.to_csv("random_sample/nodes_subset_random_sampled.csv", index=False)
edges_subset.to_csv("random_sample/edges_subset_random_sampled.csv", index=False)

print(f"Sampled test nodes : {len(sampled_node_ids)}")
print(f"Neighbour nodes    : {len(all_subset_ids) - len(sampled_node_ids)}")
print(f"Total nodes saved  : {len(nodes_subset)}")
print(f"Edges saved        : {len(edges_subset)}")
print(f"Degree groups      : {sorted(sampled['num_of_neighbors'].unique())}")
