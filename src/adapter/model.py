"""The GNN estimator, the prefix adapter and the wrapper that connects them to the LLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv

from config import (BASE_INIT_STD, CODE_DIM, DROPOUT, GATE_INIT_FRACTION, GNN_CONV, GNN_HEADS,
                    GNN_LAYERS, HEAD_DIM, HIDDEN_CHANNELS, HIDDEN_SIZE, IN_CHANNELS,
                    NUM_KV_HEADS, NUM_LAYERS, NUM_NEIGHBOUR_TOKENS, NUM_PREFIX_TOKENS,
                    OUT_CHANNELS, TRUNK_DIM)


def _conv(kind, in_dim, out_dim, heads):
    """One message-passing layer of `kind`, `out_dim` wide whichever kind it is."""
    if kind == "sage":
        return SAGEConv(in_dim, out_dim, aggr="mean")
    assert out_dim % heads == 0, f"{out_dim} channels do not divide into {heads} heads"
    return GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True, dropout=0.0)


class GraphEncoder(nn.Module):
    """`num_layers` message-passing layers, all 1024 wide. Layer i aggregates hop i, so the loader must
    sample at least `num_layers` hops — config.num_neighbors() derives that.
    """

    def __init__(self, in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS,
                 out_channels=OUT_CHANNELS, dropout=DROPOUT, num_layers=GNN_LAYERS,
                 conv=GNN_CONV, heads=GNN_HEADS):
        super().__init__()
        assert num_layers >= 1, "the encoder needs at least one conv layer"
        assert conv in ("sage", "gatv2"), f"unknown conv {conv!r}"
        widths = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            _conv(conv, widths[i], widths[i + 1], heads) for i in range(num_layers)
        )
        self.conv_kind = conv
        self.dropout = dropout
        # unit norm, like the voyage features it replaces
        self.mask = nn.Parameter(torch.randn(in_channels) / in_channels ** 0.5)

    def forward(self, x, edge_index, batch_size):
        """Returns (num_nodes, out_channels) — every row of the batch, seeds first."""
        h = torch.cat([self.mask.expand(batch_size, -1), x[batch_size:]])
        for conv in self.convs[:-1]:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.convs[-1](h, edge_index)   # no activation on the output layer


class EmbeddingEstimator(nn.Module):
    """GraphEncoder + linear head -> a unit-norm estimate of the paper's own embedding."""

    def __init__(self, num_layers=GNN_LAYERS, conv=GNN_CONV):
        super().__init__()
        self.encoder = GraphEncoder(num_layers=num_layers, conv=conv)
        self.head = nn.Linear(OUT_CHANNELS, OUT_CHANNELS)

    def forward(self, x, edge_index, batch_size):
        """Returns (num_nodes, out_channels); only the seed rows 0..batch_size-1 are estimates."""
        return F.normalize(self.head(self.encoder(x, edge_index, batch_size)), dim=-1)


class NeighbourTokenAdapter(nn.Module):
    """One 1-hop neighbour embedding (1024) -> one Qwen3 input embedding (5120)."""

    def __init__(self, in_channels=OUT_CHANNELS, hidden_size=HIDDEN_SIZE,
                 max_tokens=NUM_NEIGHBOUR_TOKENS):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.in_norm = nn.LayerNorm(in_channels)
        self.proj = nn.Linear(in_channels, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.base = nn.Parameter(torch.randn(hidden_size) * BASE_INIT_STD)
        self.gate = nn.Parameter(torch.zeros(1))
        self.register_buffer("scale", torch.ones(()))

    @torch.no_grad()
    def match_embeddings(self, embed_weight):
        """Set `scale` so a token leaves this module at the norm of a real token embedding."""
        mean_norm = embed_weight.detach().float().norm(dim=1).mean()
        self.scale.fill_(float(mean_norm / self.hidden_size ** 0.5))

    @torch.no_grad()
    def calibrate_gate(self, neighbour_vectors, fraction=GATE_INIT_FRACTION):
        """The neighbour-token twin of PrefixAdapter.calibrate_gate — same reasoning."""
        self.gate.zero_()
        delta = self.proj(self.in_norm(neighbour_vectors))
        mean_norm = delta.float().norm(dim=1).mean()
        self.gate.fill_(float(fraction * self.base.norm() / mean_norm))
        return float(self.gate)

    def forward(self, h):
        """(k, in_channels) -> (k, hidden_size). k may be zero."""
        delta = self.proj(self.in_norm(h))
        return self.scale * self.out_norm(self.base + self.gate * delta)


def oracle_vector(batch):
    """The ceiling experiment's stand-in for the encoder: the seed's OWN voyage embedding."""
    return batch.x[:batch.batch_size]


def corrupt_to_cosine(v, min_cos):
    """Rotate each row of `v` to a cosine drawn uniformly from [min_cos, 1] with itself."""
    v = F.normalize(v.float(), dim=-1)
    cos = min_cos + (1.0 - min_cos) * torch.rand(v.size(0), 1, device=v.device)
    noise = torch.randn_like(v)
    noise = F.normalize(noise - (noise * v).sum(-1, keepdim=True) * v, dim=-1)
    return cos * v + (1.0 - cos ** 2).clamp(min=0).sqrt() * noise


def mix_in_refs(batch, v, fraction):
    """Swap each seed's vector for its reference mean with probability `fraction`."""
    refs = refs_mean_vector(batch)
    use = ((torch.rand(v.size(0), 1, device=v.device) < fraction)
           & (refs.abs().sum(-1, keepdim=True) > 0))
    return torch.where(use, refs.to(v.dtype), v)


def refs_mean_vector(batch):
    """The normalised mean of each seed's references' embeddings."""
    out = torch.zeros(batch.batch_size, batch.x.size(1), device=batch.x.device)
    for seed in range(batch.batch_size):
        rows = batch.edge_index[0][batch.edge_index[1] == seed]
        if rows.numel():
            out[seed] = F.normalize(batch.x[rows].float().mean(0), dim=0)
    return out


class PrefixAdapter(nn.Module):
    """Graph embedding -> per-layer key/value prefixes for a frozen Qwen3-14B."""

    def __init__(self, in_channels=OUT_CHANNELS, trunk_dim=TRUNK_DIM, code_dim=CODE_DIM,
                 num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                 num_prefix_tokens=NUM_PREFIX_TOKENS):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
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
        # index 0 decodes keys, index 1 values; each is shared across all layers of its role
        self.decoders = nn.ModuleList(
            nn.Linear(code_dim, num_prefix_tokens * num_kv_heads * head_dim) for _ in range(2)
        )

        # node-independent prefix: plain prefix tuning, and what gate=0 falls back to
        self.base = nn.Parameter(
            torch.randn(num_layers, 2, num_kv_heads, num_prefix_tokens, head_dim) * BASE_INIT_STD
        )
        self.gate = nn.Parameter(torch.zeros(num_layers, 2))

    @torch.no_grad()
    def calibrate_gate(self, graph_vectors, fraction=GATE_INIT_FRACTION):
        """Open the gate so the graph supplies `fraction` of the prefix norm at step 0."""
        self.gate.zero_()
        base_norm = self.base.norm()
        z = self.trunk(graph_vectors)
        c = self.codes(z).view(graph_vectors.size(0), self.num_layers, 2, self.code_dim)
        delta = torch.stack([self.decoders[role](c[:, :, role]) for role in range(2)], dim=2)
        mean_norm = delta.float().flatten(1).norm(dim=1).mean()
        self.gate.fill_(float(fraction * base_norm / mean_norm))
        return float(self.gate.flatten()[0])

    def forward(self, g):
        b = g.size(0)
        z = self.trunk(g)
        c = self.codes(z).view(b, self.num_layers, 2, self.code_dim)
        delta = torch.stack([self.decoders[role](c[:, :, role]) for role in range(2)], dim=2)
        delta = delta.view(
            b, self.num_layers, 2, self.num_prefix_tokens, self.num_kv_heads, self.head_dim
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
    ap.add_argument("--gnn-conv", choices=["sage", "gatv2"], default=GNN_CONV)
    args = ap.parse_args()

    data = load_graph()
    with open(os.path.join(FROZEN_DIR, "splits_eligible.json")) as f:
        seeds = json.load(f)["train"]

    model = GraphEncoder(num_layers=args.gnn_layers, conv=args.gnn_conv)
    print(model)
    print(f"\ntrainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch = next(iter(make_loader(data, seeds, batch_size=8, shuffle=False,
                                  gnn_layers=args.gnn_layers)))
    model.eval()
    with torch.no_grad():
        h = model(batch.x, batch.edge_index, batch.batch_size)
    out = h[:batch.batch_size]

    print(f"\nbatch: {batch.num_nodes} nodes, {batch.num_edges} edges, "
          f"batch_size {batch.batch_size}")
    print(f"output: {tuple(h.shape)} (all rows), seeds {tuple(out.shape)}, "
          f"norms {out.norm(dim=1).min():.3f}–{out.norm(dim=1).max():.3f}")
    assert h.shape == (batch.num_nodes, OUT_CHANNELS)

    # The seed's own embedding must not reach its output, or the target would leak.
    perturbed = batch.x.clone()
    perturbed[:batch.batch_size] = torch.randn_like(perturbed[:batch.batch_size])
    with torch.no_grad():
        out2 = model(perturbed, batch.edge_index, batch.batch_size)[:batch.batch_size]
    assert torch.equal(out, out2), "seed features leak into the output"
    print("seed features do not affect the seed output (mask works)")

    oracle = oracle_vector(batch)
    assert oracle.shape == (batch.batch_size, IN_CHANNELS)
    assert torch.equal(oracle, data.x[batch.n_id[:batch.batch_size]]), \
        "oracle_vector is not the seed nodes' own stored embedding"
    assert not torch.allclose(oracle, out), "oracle_vector returned the encoder output"
    print(f"oracle: {tuple(oracle.shape)}, the seeds' own embeddings, "
          f"cosine to encoder output {F.cosine_similarity(oracle, out).mean():.3f}")

    # --- prefix adapter ---------------------------------------------------
    adapter = PrefixAdapter()
    print(f"\nadapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")

    adapter.eval()
    with torch.no_grad():
        prefix = adapter(out)
    assert prefix.shape == (batch.batch_size, NUM_LAYERS, 2, NUM_KV_HEADS,
                            NUM_PREFIX_TOKENS, HEAD_DIM)
    print(f"prefix: {tuple(prefix.shape)}, {prefix[0].numel():,} values per node")

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

    for role, name in enumerate(("keys", "values")):
        bias = adapter.decoders[role].bias
        with torch.no_grad():
            reference, saved = adapter(out), bias.clone()
            bias.add_(1.0)
            moved = adapter(out)
            bias.copy_(saved)          # not sub_: +1 then -1 does not round-trip in float
        changed = (moved != reference).flatten(3).any(-1).any(0).any(0)   # per role
        assert changed.tolist() == [role == 0, role == 1], \
            f"the {name} decoder moved {changed.tolist()} (keys, values)"
    print("separate decoders: the key decoder moves only keys, the value decoder only values")

    # --- neighbour token adapter -------------------------------------------
    from graph import select_neighbour_rows

    ntoken = NeighbourTokenAdapter()
    print(f"\nneighbour token adapter parameters: "
          f"{sum(p.numel() for p in ntoken.parameters()):,}")

    # A stand-in for Qwen3's embedding matrix; the real one is passed in train.py.
    ntoken.match_embeddings(torch.randn(1000, HIDDEN_SIZE) * 0.02)
    one = next(iter(make_loader(data, seeds[:1], batch_size=1, shuffle=False,
                                gnn_layers=args.gnn_layers)))
    with torch.no_grad():
        h_one = model(one.x, one.edge_index, one.batch_size)
        rows = select_neighbour_rows(one)
        tokens = ntoken(h_one[rows])
    print(f"seed {int(one.n_id[0])}: {len(rows)} neighbour tokens, {tuple(tokens.shape)}, "
          f"norms {tokens.norm(dim=1).min():.2f}–{tokens.norm(dim=1).max():.2f}")
    assert tokens.shape == (len(rows), HIDDEN_SIZE)

    assert torch.allclose(tokens, tokens[:1].expand_as(tokens), atol=1e-6), \
        "gate is zero but the neighbours still move their tokens"
    print("gate=0: every neighbour token is the same learned vector")

    with torch.no_grad():
        ntoken.gate.fill_(1.0)
        opened = ntoken(h_one[rows])
    assert not torch.allclose(opened, opened[:1].expand_as(opened), atol=1e-6), \
        "opening the gate left every neighbour token identical"
    print("gate=1: each neighbour token differs")

    model.train()
    adapter.train()
    ntoken.train()
    h = model(batch.x, batch.edge_index, batch.batch_size)
    loss = adapter(h[:batch.batch_size]).square().mean() + ntoken(h[batch.batch_size:]).square().mean()
    loss.backward()
    missing = [n for n, p in [*model.named_parameters(), *adapter.named_parameters(),
                              *ntoken.named_parameters()] if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    print("gradients reach every parameter of all three modules")
