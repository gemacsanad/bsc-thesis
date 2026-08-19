"""
Stage 3 — GraphSAGE encoder and prefix adapter.

Encodes a target node's reference neighbourhood into a single vector, and turns
that vector into the key/value prefixes injected into every layer of a frozen
Vicuna-7B.

`GNN_LAYERS` SAGEConv layers (all 1024 wide, mean aggregation) over the batches
produced by graph.py, so the encoder sees the node's references at layer 1 and
the references of those references at layer 2. The default is 2; `--gnn-layers`
on train.py and graph.py runs deeper encoders, and the loader's hop count follows
it — a layer reading from an unsampled hop would aggregate incomplete
neighbourhoods and produce a plausible result rather than an error.

Seed masking
------------
The seed node's own feature is the voyage embedding of the very abstract we are
trying to generate, so it is replaced with a learned [MASK] vector before any
message passing. The vector is shared across all seeds and trained with the rest
of the model, letting it settle on a representation that means "this node is the
unknown target". It is initialised to unit norm, matching the L2-normalised
voyage features it stands in for.

Masking relies on the seeds occupying rows 0..batch_size-1 of the batch, which
graph.py asserts.

Prefix adapter
--------------
Prefix tuning, so the graph enters at all 32 layers rather than only at the input.
That means 32 layers x 2 roles x 10 tokens x 4096 = 2.6M numbers per node, and a
plain linear map from the 1024-d encoder output would cost 2.68B parameters.

The cost lives in the output, not the input, so the output is factorised: a shared
trunk compresses the graph vector, one small code is emitted per (layer, role), and
a single decoder shared by all 64 of them expands each code to that layer's prefix.
Each layer sees the graph through a 96-d projection, but 64 x 96 = 6144 code
dimensions across all layers, so collectively they can still read all 1024.

The graph supplies an offset, not the whole prefix. `base` is a node-independent
learned prefix — classic prefix tuning — and `gate` starts at zero, so training
begins as prefix tuning with the graph disconnected. Setting gate to zero at eval
time is therefore the "no graph" ablation on the very same weights.

Usage:
    python adapter/model.py                  # smoke-test a forward pass on one batch
    python adapter/model.py --gnn-layers 3   # same checks on a deeper encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from config import (BASE_INIT_STD, CODE_DIM, DROPOUT, GNN_LAYERS, HEAD_DIM,
                    HIDDEN_CHANNELS, IN_CHANNELS, NUM_HEADS, NUM_LAYERS,
                    NUM_PREFIX_TOKENS, OUT_CHANNELS, TRUNK_DIM)


class GraphSAGEEncoder(nn.Module):
    """`num_layers` SAGEConv layers, all 1024 wide. Layer i aggregates hop i, so the loader
    must sample at least `num_layers` hops — config.num_neighbors() derives that."""

    def __init__(self, in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS,
                 out_channels=OUT_CHANNELS, dropout=DROPOUT, num_layers=GNN_LAYERS):
        super().__init__()
        assert num_layers >= 1, "the encoder needs at least one conv layer"
        widths = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            SAGEConv(widths[i], widths[i + 1], aggr="mean") for i in range(num_layers)
        )
        self.dropout = dropout
        # unit norm, like the voyage features it replaces
        self.mask = nn.Parameter(torch.randn(in_channels) / in_channels ** 0.5)

    def forward(self, x, edge_index, batch_size):
        """Returns (batch_size, out_channels) — one embedding per seed node."""
        h = torch.cat([self.mask.expand(batch_size, -1), x[batch_size:]])
        for conv in self.convs[:-1]:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index)      # no activation on the output layer
        return h[:batch_size]


class PrefixAdapter(nn.Module):
    """Graph embedding -> per-layer key/value prefixes for a frozen Vicuna-7B.

    Returns (batch, num_layers, 2, num_heads, num_prefix_tokens, head_dim), where
    index 0 of dim 2 is keys and index 1 is values. Stage 4 turns that into the
    `past_key_values` the model expects; nothing here imports transformers.
    """

    def __init__(self, in_channels=OUT_CHANNELS, trunk_dim=TRUNK_DIM, code_dim=CODE_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, head_dim=HEAD_DIM,
                 num_prefix_tokens=NUM_PREFIX_TOKENS):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.code_dim = code_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_channels, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
        )
        # one matrix for all 2 * num_layers heads, split apart by the reshape below
        self.codes = nn.Linear(trunk_dim, num_layers * 2 * code_dim)
        # shared by every (layer, role) 
        self.decoder = nn.Linear(code_dim, num_prefix_tokens * num_heads * head_dim)

        # node-independent prefix: plain prefix tuning, and what gate=0 falls back to
        self.base = nn.Parameter(
            torch.randn(num_layers, 2, num_heads, num_prefix_tokens, head_dim) * BASE_INIT_STD
        )
        self.gate = nn.Parameter(torch.zeros(num_layers, 2))

    def forward(self, g):
        b = g.size(0)
        z = self.trunk(g)
        c = self.codes(z).view(b, self.num_layers, 2, self.code_dim)
        # (tokens, heads, head_dim) matches how Llama reshapes a 4096-d projection
        delta = self.decoder(c).view(
            b, self.num_layers, 2, self.num_prefix_tokens, self.num_heads, self.head_dim
        )
        delta = delta.transpose(3, 4)          # -> (heads, tokens), the cache layout
        return self.base + self.gate.view(1, self.num_layers, 2, 1, 1, 1) * delta


if __name__ == "__main__":
    import argparse
    import json
    import os

    from config import FROZEN_DIR
    from graph import load_graph, make_loader

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnn-layers", type=int, default=GNN_LAYERS)
    args = ap.parse_args()

    data = load_graph()
    with open(os.path.join(FROZEN_DIR, "splits_eligible.json")) as f:
        seeds = json.load(f)["train"]

    model = GraphSAGEEncoder(num_layers=args.gnn_layers)
    print(model)
    print(f"\ntrainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch = next(iter(make_loader(data, seeds, batch_size=8, shuffle=False,
                                  gnn_layers=args.gnn_layers)))
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch_size)

    print(f"\nbatch: {batch.num_nodes} nodes, {batch.num_edges} edges, "
          f"batch_size {batch.batch_size}")
    print(f"output: {tuple(out.shape)}, norms {out.norm(dim=1).min():.3f}"
          f"–{out.norm(dim=1).max():.3f}")
    assert out.shape == (batch.batch_size, OUT_CHANNELS)

    # The seed's own embedding must not reach the output — that is the whole point
    # of the mask, and a silent failure here would leak the target abstract.
    perturbed = batch.x.clone()
    perturbed[:batch.batch_size] = torch.randn_like(perturbed[:batch.batch_size])
    with torch.no_grad():
        out2 = model(perturbed, batch.edge_index, batch.batch_size)
    assert torch.equal(out, out2), "seed features leak into the output"
    print("seed features do not affect the output (mask works)")

    # --- prefix adapter ---------------------------------------------------
    adapter = PrefixAdapter()
    print(f"\nadapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")

    adapter.eval()
    with torch.no_grad():
        prefix = adapter(out)
    assert prefix.shape == (batch.batch_size, NUM_LAYERS, 2, NUM_HEADS,
                            NUM_PREFIX_TOKENS, HEAD_DIM)
    print(f"prefix: {tuple(prefix.shape)}, {prefix[0].numel():,} values per node")

    # gate starts at zero, so the graph must not reach the prefix yet: every node
    # gets the same base prefix. This is exactly the "no graph" ablation.
    with torch.no_grad():
        elsewhere = adapter(torch.randn_like(out))
    assert torch.equal(prefix, elsewhere), "gate is zero but the graph still moves the prefix"
    assert torch.equal(prefix[0], adapter.base), "zero-gate output is not the base prefix"
    print("gate=0: prefix is node-independent and equals base (prefix-tuning fallback)")

    with torch.no_grad():
        adapter.gate.fill_(1.0)
        opened = adapter(out)
    assert not torch.equal(prefix, opened), "opening the gate changed nothing"
    print("gate=1: the graph embedding moves the prefix")

    # End to end, with the gate open so the trunk actually sits on the path — a
    # shape or dtype break between the two halves shows up here.
    model.train()
    adapter.train()
    adapter(model(batch.x, batch.edge_index, batch.batch_size)).square().mean().backward()
    missing = [n for n, p in [*model.named_parameters(), *adapter.named_parameters()]
               if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    print("gradients reach every parameter of both modules")
