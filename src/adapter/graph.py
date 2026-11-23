"""Graph tensors, the citation-graph neighbourhoods and the dataloaders."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import coalesce, remove_self_loops

from config import (FROZEN_DIR, GNN_LAYERS, JOINT_DIR, NODE_EMBEDDINGS,
                    NUM_NEIGHBOUR_TOKENS, num_neighbors)


def load_graph(embeddings=None, edges=None):
    """Build the directed reference graph with voyage embeddings as node features."""
    embeddings = embeddings or NODE_EMBEDDINGS
    edges = edges or os.path.join(JOINT_DIR, "edges.csv")
    assert os.path.exists(embeddings), (
        f"{embeddings} does not exist — run `python whiten_features.py` first, or point "
        f"config.NODE_EMBEDDINGS back at node_embeddings.npy to reproduce a pre-run-8 run")

    x = torch.from_numpy(np.load(embeddings))
    citing, cited = torch.from_numpy(
        np.loadtxt(edges, delimiter=",", skiprows=1, dtype=np.int64).T
    )

    # reversed: messages flow cited -> citing, so a node aggregates its references
    edge_index = torch.stack([cited, citing])
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=x.size(0))

    return Data(x=x, edge_index=edge_index, num_nodes=x.size(0))


def make_loader(data, seed_nodes, batch_size, shuffle, num_workers=0,
                gnn_layers=GNN_LAYERS, hops=None):
    """NeighborLoader over `seed_nodes`, returning their full `gnn_layers`-hop neighbourhoods."""
    depth = gnn_layers if hops is None else hops
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors(depth) if depth else [0],
        input_nodes=torch.as_tensor(seed_nodes, dtype=torch.long),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def with_references(data, nodes):
    """-> the nodes of `nodes` that cite at least one paper in the graph, order kept."""
    has = torch.zeros(data.num_nodes, dtype=torch.bool)
    has[data.edge_index[1]] = True
    return [n for n in nodes if has[n]]


def within_subgraph(data, nodes, max_edges, gnn_layers):
    """-> the nodes of `nodes` whose full `gnn_layers`-hop subgraph has at most `max_edges` edges,
    order kept. The count is the loader's: a node's in-edges, plus at 2 hops the in-edges of each of
    its in-neighbours (an upper bound, since it does not deduplicate).
    """
    deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    edges = deg.clone()
    if gnn_layers >= 2:
        edges += torch.zeros_like(deg).scatter_add_(0, data.edge_index[1],
                                                    deg[data.edge_index[0]])
    assert gnn_layers <= 2, "the subgraph count is written for at most 2 hops"
    return [n for n in nodes if edges[n] <= max_edges]


def centroid_distances(x):
    """-> (N,) cosine distance of each row to the centroid direction of the set."""
    normed = F.normalize(x.float(), dim=1)
    centroid = F.normalize(normed.mean(0), dim=0)
    return ((1.0 - normed @ centroid) / 2.0).abs()


def select_neighbour_rows(batch, max_tokens=NUM_NEIGHBOUR_TOKENS, seed_row=0):
    """-> batch-local rows of the seed's 1-hop neighbours, closest to the centroid first."""
    assert batch.batch_size == 1, (
        f"select_neighbour_rows handles batch_size 1, got {batch.batch_size}; k varies "
        f"per node, so a wider batch needs padding and a mask")
    rows = batch.edge_index[0][batch.edge_index[1] == seed_row]
    if rows.numel() <= 1:
        return rows
    order = torch.argsort(centroid_distances(batch.x[rows]), stable=True)
    return rows[order[:max_tokens]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edges", default=os.path.join(JOINT_DIR, "edges.csv"))
    ap.add_argument("--embeddings",
                    default=os.path.join(JOINT_DIR, "node_embeddings.npy"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gnn-layers", type=int, default=GNN_LAYERS,
                    help="encoder depth; sets the number of sampled hops")
    args = ap.parse_args()

    print("Building graph …")
    data = load_graph(args.embeddings, args.edges)
    print(data)

    ei = data.edge_index
    assert (ei[0] != ei[1]).all(), "self-loops survived"
    pairs = ei[0] * data.num_nodes + ei[1]
    assert pairs.unique().numel() == pairs.numel(), "duplicate edges survived"

    refs = torch.bincount(ei[1], minlength=data.num_nodes)   # in-edges = own references
    print(f"  {ei.size(1):,} directed edges, no self-loops, no duplicates")
    print(f"  references per node: mean {refs.float().mean():.1f}, max {refs.max()}")

    with open(os.path.join(FROZEN_DIR, "splits_eligible.json")) as f:
        splits = json.load(f)
    with open(os.path.join(FROZEN_DIR, "eval_nodes.json")) as f:
        eval_nodes = json.load(f)

    print("\nLoaders")
    for name, seeds in [("train", splits["train"]), ("valid", splits["valid"]),
                        ("eval", eval_nodes)]:
        print(f"  {name:<6} {len(seeds):>7,} seed nodes")

    loader = make_loader(data, splits["train"], args.batch_size, shuffle=False,
                         gnn_layers=args.gnn_layers)
    batch = next(iter(loader))
    # The seed nodes must be the first rows: the [MASK] vector is placed at x[:batch_size].
    assert batch.n_id[:batch.batch_size].tolist() == splits["train"][:args.batch_size], \
        "seed nodes are not the first rows of the batch"

    print(f"\nFirst train batch (batch_size {args.batch_size}): "
          f"{batch.num_nodes:,} nodes, {batch.num_edges:,} edges")
    print(f"  x {tuple(batch.x.shape)}, seed nodes at rows 0..{batch.batch_size - 1}")

    # Check the direction: a seed aggregates exactly the papers it cites.
    import pandas as pd
    seed = splits["train"][0]
    one = next(iter(make_loader(data, [seed], batch_size=1, shuffle=False)))
    sampled = set(one.n_id[one.edge_index[0][one.edge_index[1] == 0]].tolist())
    raw = pd.read_csv(args.edges)
    expected = set(raw.loc[raw["src_node_id"] == seed, "dst_node_id"])
    assert sampled == expected, (
        f"seed {seed} aggregates {len(sampled)} nodes, expected its {len(expected)} references"
    )
    print(f"  seed {seed} aggregates exactly its {len(expected)} references (direction correct)")


if __name__ == "__main__":
    main()
