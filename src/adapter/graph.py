"""
Stage 2 — graph tensors and dataloaders for GraphSAGE training.

Builds the PyG `Data` object the GNN runs on, and `NeighborLoader`s that feed it
complete neighbourhoods.

Graph
-----
All 384,756 joint-dataset nodes. Features are the voyage-4-large embeddings of
`title + "\\n\\n" + abstract` (row i == node_id i, L2-normalised). Edges come from
data/processed/joint/edges.csv with the 20 self-loops dropped.

The graph is **directed, references only**. edges.csv stores (citing, cited); PyG
aggregates along in-edges, so edge_index is stored reversed as (cited, citing) and
messages flow from a paper's references into the paper. A node therefore learnss
only from work it cites — never from papers citing it, which are published later
and would not exist at generation time.

Hop 1 is then exactly the node's reference list, so the graph matches the 5-15
out-degree filter in prepare_data.py rather than cutting across it.

Sampling
--------
`num_neighbors=[-1] * gnn_layers` — every neighbour at every hop, nothing dropped.
The depth follows the encoder's layer count: a conv layer reading from an unsampled
hop would aggregate incomplete neighbourhoods without erroring. Neighbourhoods stay
small enough not to need a cap (capped train seeds: mean 199 nodes at 2 hops, 876 at
3, 2,645 at 4). Seed nodes occupy rows 0..batch_size-1 of each batch.

`Data` stays on CPU; sampling runs there and only the sampled subgraph moves to GPU.

Usage:
    python adapter/graph.py     # build the graph and print stats
"""

import argparse
import json
import os

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import coalesce, remove_self_loops

from config import FROZEN_DIR, GNN_LAYERS, JOINT_DIR, num_neighbors


def load_graph(embeddings=None, edges=None):
    """Build the directed reference graph with voyage embeddings as node features."""
    embeddings = embeddings or os.path.join(JOINT_DIR, "node_embeddings.npy")
    edges = edges or os.path.join(JOINT_DIR, "edges.csv")

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
                gnn_layers=GNN_LAYERS):
    """NeighborLoader over `seed_nodes`, returning their full `gnn_layers`-hop neighbourhoods."""
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors(gnn_layers),
        input_nodes=torch.as_tensor(seed_nodes, dtype=torch.long),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


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
    # Stage 3 substitutes a [MASK] vector at x[:batch_size], which is only correct
    # if the seed nodes really are the first rows of the batch.
    assert batch.n_id[:batch.batch_size].tolist() == splits["train"][:args.batch_size], \
        "seed nodes are not the first rows of the batch"

    print(f"\nFirst train batch (batch_size {args.batch_size}): "
          f"{batch.num_nodes:,} nodes, {batch.num_edges:,} edges")
    print(f"  x {tuple(batch.x.shape)}, seed nodes at rows 0..{batch.batch_size - 1}")

    # Direction is the thing that could silently be backwards: what the seed
    # aggregates must be exactly the papers it cites in edges.csv.
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
