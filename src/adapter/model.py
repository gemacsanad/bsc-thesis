"""
Stage 3 — graph encoder, prefix adapter and neighbour-token adapter.

Encodes a target node's reference neighbourhood, and turns it into the two channels
that reach a frozen Qwen3-14B:

  * the **pooled** seed vector -> key/value prefixes injected at every one of the 40
    layers (PrefixAdapter), unchanged since step 2;
  * each **1-hop neighbour** -> one ordinary input embedding prepended to the prompt
    (NeighbourTokenAdapter), added in stage 9.

`GNN_LAYERS` message-passing layers (all 1024 wide) over the batches produced by
graph.py, so the encoder sees the node's references at layer 1 and the references
of those references at layer 2. The default is 2; `--gnn-layers` on train.py and
graph.py runs deeper encoders, and the loader's hop count follows it — a layer
reading from an unsampled hop would aggregate incomplete neighbourhoods and produce
a plausible result rather than an error.

`GNN_CONV` picks the aggregation. It is "gatv2" since stage 9 — learned attention
over the references rather than GraphSAGE's unweighted mean — and "sage" reproduces
every run up to step 5.

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
Prefix tuning, so the graph enters at all 40 layers rather than only at the input.
That means 40 layers x 2 roles x 32 tokens x (8 kv heads x 128) = 2.6M numbers per
node, and a plain linear map from the 1024-d encoder output would cost 2.68B
parameters.

Both figures are unchanged from the Vicuna-7B configuration (32 layers x 2 x 10
tokens x 4096), and that is deliberate rather than coincidental: Qwen3 serves 40
query heads from only 8 KV heads, which would have narrowed the prefix 3.2x at the
old 10 tokens. T was raised to 32 to hold the graph pathway's capacity fixed, since
that is the quantity that has to be constant for the two runs to be comparable.

The cost lives in the output, not the input, so the output is factorised: a shared
trunk transforms the graph vector at full 1024 width, one small code is emitted per
(layer, role), and two decoders — one for keys, one for values — expand each code to
that layer's prefix. Each is shared across all 40 layers of its role.

Keys and values are decoded separately because they do different jobs in different
spaces: keys decide where attention goes (a real Qwen3 key is k_norm-ed per head and
rotated by RoPE), values carry what is written back. One shared decoder forced both
into the same subspace. Each (layer, role) reads the graph through a 192-d code, and
80 x 192 = 15,360 code dimensions collectively cover all 1024.

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
from torch_geometric.nn import GATv2Conv, SAGEConv

from config import (BASE_INIT_STD, CODE_DIM, DROPOUT, GATE_INIT_FRACTION, GNN_CONV, GNN_HEADS,
                    GNN_LAYERS, HEAD_DIM, HIDDEN_CHANNELS, HIDDEN_SIZE, IN_CHANNELS,
                    NUM_KV_HEADS, NUM_LAYERS, NUM_NEIGHBOUR_TOKENS, NUM_PREFIX_TOKENS,
                    OUT_CHANNELS, TRUNK_DIM)


def _conv(kind, in_dim, out_dim, heads):
    """One message-passing layer of `kind`, `out_dim` wide whichever kind it is.

    GATv2 emits `heads` x `out_dim // heads` and concatenates, so swapping the kind does not
    also change the width — the comparison against the GraphSAGE runs stays about aggregation.
    """
    if kind == "sage":
        return SAGEConv(in_dim, out_dim, aggr="mean")
    assert out_dim % heads == 0, f"{out_dim} channels do not divide into {heads} heads"
    return GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True, dropout=0.0)


class GraphEncoder(nn.Module):
    """`num_layers` message-passing layers, all 1024 wide. Layer i aggregates hop i, so the
    loader must sample at least `num_layers` hops — config.num_neighbors() derives that.

    `conv` picks the aggregation: "gatv2" weights each reference by learned attention,
    "sage" takes the unweighted mean and reproduces every run up to step 5.

    forward() returns **all** rows, not just the seeds. Stage 9 needs two things out of one
    pass: the seed's pooled embedding for the KV prefix, and the 1-hop neighbours' own
    embeddings for the input tokens. Returning `h` and letting the caller slice keeps them
    consistent by construction — the neighbour tokens are always from the same forward pass
    as the prefix beside them.
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


class NeighbourTokenAdapter(nn.Module):
    """One 1-hop neighbour embedding (1024) -> one Qwen3 input embedding (5120).

    Weights are shared across all neighbours, so k is free to vary per node and nothing is
    tied to a position. The structure mirrors PrefixAdapter deliberately:

        token = scale * LayerNorm(base + gate * proj(h))

    `base` is a node-independent learned token and `gate` starts at zero, so at
    initialisation every neighbour slot holds the *same* generic vector and training begins
    as ordinary soft-prompt tuning. Setting gate to zero at eval keeps the k slots and
    removes only their content, which separates "does it help to have k tokens" from "does
    it help that they carry neighbour information" — a distinction no other ablation gives.

    `scale` is the mean row norm of the LLM's own embedding matrix, set by `match_embeddings`
    before training. An unscaled projection of GNN outputs is out of distribution for the
    embedding layer, and the resulting damage looks like the idea not working rather than
    like a bug.
    """

    def __init__(self, in_channels=OUT_CHANNELS, hidden_size=HIDDEN_SIZE,
                 max_tokens=NUM_NEIGHBOUR_TOKENS):
        super().__init__()
        self.hidden_size = hidden_size
        # how many neighbours get a token; carried here rather than threaded through
        # step_loss so training and generation read it from the same place
        self.max_tokens = max_tokens
        self.in_norm = nn.LayerNorm(in_channels)
        self.proj = nn.Linear(in_channels, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.base = nn.Parameter(torch.randn(hidden_size) * BASE_INIT_STD)
        self.gate = nn.Parameter(torch.zeros(1))
        # a buffer, not a constant: it rides in the checkpoint, so generation cannot
        # silently rescale the tokens the run was trained with
        self.register_buffer("scale", torch.ones(()))

    @torch.no_grad()
    def match_embeddings(self, embed_weight):
        """Set `scale` so a token leaves this module at the norm of a real token embedding.

        LayerNorm leaves ~unit variance per dimension, so the raw output norm is ~sqrt(H);
        dividing it out and multiplying by the measured mean puts the two on the same scale.
        """
        mean_norm = embed_weight.detach().float().norm(dim=1).mean()
        self.scale.fill_(float(mean_norm / self.hidden_size ** 0.5))

    @torch.no_grad()
    def calibrate_gate(self, neighbour_vectors, fraction=GATE_INIT_FRACTION):
        """The neighbour-token twin of PrefixAdapter.calibrate_gate — same reasoning.

        Measured against `base` rather than against the post-LayerNorm output, so the number
        means the same thing in both modules: what share of the pre-norm token the graph
        supplies.
        """
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
    """The ceiling experiment's stand-in for the encoder: the seed's OWN voyage embedding.

    This deliberately leaks the answer. A node's feature is the embedding of that node's
    abstract — the very abstract the model is asked to write — which is why
    GraphEncoder masks it before any message passing. Here the mask is skipped and the
    encoder bypassed entirely, so the adapter receives a perfect 1,024-d description of the
    target and no graph at all.

    It measures one thing the graph runs cannot separate: whether a frozen Qwen3-14B can be
    steered by a KV prefix decoded from a single dense vector *at all*. If even this writes
    generic abstracts, the prefix->LLM channel is the bottleneck and no better graph,
    aggregator or embedding model can help; if it writes the right paper, the channel works
    and the graph side is what to fix.

    Its output is a ceiling, never a result: the number is not comparable to any arm that
    does not see the target, and must never be reported beside one without that label.

    Relies on the seeds occupying rows 0..batch_size-1, which graph.py asserts — the same
    invariant the mask relies on.
    """
    return batch.x[:batch.batch_size]


class PrefixAdapter(nn.Module):
    """Graph embedding -> per-layer key/value prefixes for a frozen Qwen3-14B.

    Returns (batch, num_layers, 2, num_kv_heads, num_prefix_tokens, head_dim), where
    index 0 of dim 2 is keys and index 1 is values. Stage 4 turns that into the
    `past_key_values` the model expects; nothing here imports transformers.

    The head count is the **key/value** one, not the query one — the prefix is a KV
    tensor. Under GQA the two differ (Qwen3-14B: 40 query heads, 8 KV heads), and
    using the query count here would produce a cache the model silently misreads.
    """

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
        """Open the gate so the graph supplies `fraction` of the prefix norm at step 0.

        Stage 10. `gate = 0` is a free ablation but a bad initialisation: it makes ignoring the
        graph both the default and the starting point, and step 6 measured what that costs —
        the model learns the generic register while the gate is still shut, after which the
        gradient pushing it open is small. The ablation is kept by zeroing at *evaluation*
        time, which never required training from zero.

        Calibrated rather than set to a constant because the right value is not knowable in
        advance: `delta`'s scale depends on the encoder's output distribution, and step 6
        reached graph_fraction 0.70 at a gate of only 0.0076. A fixed literal would silently
        mean something different for a different encoder.
        """
        self.gate.zero_()
        base_norm = self.base.norm()
        # gate=0, so this is the pure node-dependent part: forward() returns base + gate*delta
        # and we need ||delta|| alone. Re-deriving it here keeps the two definitions in sync.
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
        # (tokens, kv_heads, head_dim) matches how the model reshapes a k/v projection.
        # For Qwen3-14B that projection is 8 x 128 = 1024 wide, not hidden_size (5120) —
        # under GQA the k/v projections are narrower than q.
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

    # The seed's own embedding must not reach the seed's output — that is the whole point
    # of the mask, and a silent failure here would leak the target abstract.
    perturbed = batch.x.clone()
    perturbed[:batch.batch_size] = torch.randn_like(perturbed[:batch.batch_size])
    with torch.no_grad():
        out2 = model(perturbed, batch.edge_index, batch.batch_size)[:batch.batch_size]
    assert torch.equal(out, out2), "seed features leak into the output"
    print("seed features do not affect the seed output (mask works)")

    # The oracle must be the seed's own feature and nothing else — if it silently returned
    # the masked or aggregated vector, the ceiling run would quietly measure the graph again.
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

    # Each decoder must write only its own role. A stack or view along the wrong axis would
    # interleave keys and values and still produce the right shape.
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

    # gate=0 is the ablation that keeps the slots and removes their content, so every
    # token must be the *same* vector — the neighbour-token twin of the prefix's gate=0.
    assert torch.allclose(tokens, tokens[:1].expand_as(tokens), atol=1e-6), \
        "gate is zero but the neighbours still move their tokens"
    print("gate=0: every neighbour token is the same learned vector")

    with torch.no_grad():
        ntoken.gate.fill_(1.0)
        opened = ntoken(h_one[rows])
    assert not torch.allclose(opened, opened[:1].expand_as(opened), atol=1e-6), \
        "opening the gate left every neighbour token identical"
    print("gate=1: each neighbour token differs")

    # End to end, with both gates open so the trunk and the projection sit on the path —
    # a shape or dtype break between the halves shows up here.
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
