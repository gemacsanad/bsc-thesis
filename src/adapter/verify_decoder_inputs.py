"""
Run 9 — CPU checks for everything new between the graph and the decoder.

Each of these, if wrong, produces a run that trains and generates fluent text: a noise model
that misses its cosine, a "reference mean" that includes the paper itself (a silent leak), an
epoch slicer that repeats nodes, a scorer that ranks an empty string first. So each is checked
on a synthetic graph, in seconds, before any GPU time — with a negative control where the check
could otherwise pass vacuously.

Usage:
    cd src/adapter && python verify_decoder_inputs.py      # exits non-zero on any failure
"""

import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from config import OUT_CHANNELS
from generate import load_stage_a
from graph import make_loader
from identify import ranks
from lora import targets_of
from model import GraphEncoder, corrupt_to_cosine, mix_in_refs, oracle_vector, refs_mean_vector
from train import epoch_nodes

torch.manual_seed(0)


def check(ok, what):
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        raise SystemExit(1)


# --- the noise model ---------------------------------------------------------------------
v = F.normalize(torch.randn(4000, 1024), dim=1)
noisy = corrupt_to_cosine(v, 0.3)
cos = (noisy * v).sum(1)
check(torch.allclose(noisy.norm(dim=1), torch.ones(4000), atol=1e-5), "noisy vectors are unit norm")
check(cos.min() >= 0.3 - 1e-5 and cos.max() <= 1 + 1e-5,
      f"cosine to the truth stays in [0.3, 1] (got {cos.min():.3f}..{cos.max():.3f})")
check(abs(cos.mean() - 0.65) < 0.02 and abs(cos.median() - 0.65) < 0.03,
      f"cosine is uniform over the range (mean {cos.mean():.3f}, median {cos.median():.3f})")
check((corrupt_to_cosine(v, 1.0) - v).abs().max() < 1e-5, "min_cos 1.0 is the identity")

# --- a synthetic reference graph -----------------------------------------------------------
# Paper 0 cites 1, 2, 3; paper 1 cites 4; paper 5 cites nothing. 4 also cites 0 (a mutual
# citation two hops out), which must not pull the seed's own row into its reference mean.
citing_cited = [(0, 1), (0, 2), (0, 3), (1, 4), (4, 0)]
x = F.normalize(torch.randn(6, 1024), dim=1)
edge_index = torch.tensor([[c for _, c in citing_cited], [s for s, _ in citing_cited]])
data = Data(x=x, edge_index=edge_index, num_nodes=6)       # stored cited -> citing, as graph.py

batch = next(iter(make_loader(data, [0], 1, shuffle=False, hops=1)))
expected = F.normalize(x[[1, 2, 3]].mean(0), dim=0)
check(torch.allclose(refs_mean_vector(batch)[0], expected, atol=1e-6),
      "refs-mean is the normalised mean of exactly the papers the seed cites")
check(not torch.allclose(refs_mean_vector(batch)[0], F.normalize(x[[0, 1, 2, 3]].mean(0), dim=0)),
      "negative control: including the seed's own row gives a different vector")
check(torch.equal(oracle_vector(batch)[0], x[0]), "oracle is the seed's own row")

two = next(iter(make_loader(data, [0], 1, shuffle=False, hops=2)))
check(torch.allclose(refs_mean_vector(two)[0], expected, atol=1e-6),
      "a deeper sample does not change the reference mean (hop 2 and the citer 4 are ignored)")
none = next(iter(make_loader(data, [5], 1, shuffle=False, hops=1)))
check(torch.equal(refs_mean_vector(none)[0], torch.zeros(1024)),
      "a paper with no references gets a zero vector, not NaN")

seeds_only = next(iter(make_loader(data, [0, 1], 2, shuffle=False, hops=0)))
check(seeds_only.num_nodes == 2 and seeds_only.edge_index.numel() == 0,
      "hops=0 returns the seeds alone (oracle training loads no neighbourhoods)")

# --- run 10a: half the training papers are fed their reference mean -------------------------
both = next(iter(make_loader(data, [0, 5], 2, shuffle=False, hops=1)))
own = oracle_vector(both)
draws = [mix_in_refs(both, own, 0.5) for _ in range(400)]
took_refs = sum(torch.allclose(d[0], refs_mean_vector(both)[0], atol=1e-6) for d in draws)
check(all(torch.allclose(d[0], own[0]) or torch.allclose(d[0], refs_mean_vector(both)[0], atol=1e-6)
          for d in draws), "a mixed vector is either the paper's own embedding or its reference mean")
check(150 < took_refs < 250, f"the reference mean is drawn about half the time ({took_refs}/400)")
check(all(torch.equal(d[1], own[1]) for d in draws),
      "a paper with no references keeps its own embedding, never a zero vector")
check(torch.equal(mix_in_refs(both, own, 0.0), own), "fraction 0 is the plain oracle")

# --- the Stage-A arm loads the head, and normalises -------------------------------------------
encoder, head = GraphEncoder(), nn.Linear(OUT_CHANNELS, OUT_CHANNELS)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "stage_a.pt")
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(),
                "gnn_layers": 2, "gnn_conv": "gatv2", "epoch": 1, "output_cosine": 0.0}, path)
    vector, layers = load_stage_a(path, "cpu", "node_embeddings_whitened.npy")
with torch.no_grad():
    encoder.eval()
    got = vector(two)
    want = F.normalize(head(encoder(two.x, two.edge_index, two.batch_size)[:1]), dim=-1)
    raw = encoder(two.x, two.edge_index, two.batch_size)[:1]
check(layers == 2 and torch.allclose(got, want, atol=1e-5), "stage-a returns normalised head(GNN)")
check(not torch.allclose(got, F.normalize(raw, dim=-1), atol=1e-3),
      "negative control: the head is applied (run 8's bug was skipping it)")

# --- epoch slicing --------------------------------------------------------------------------
nodes = list(range(100, 1100))
slices = [epoch_nodes(nodes, e, 250) for e in range(4)]
check(sorted(sum(slices, [])) == nodes, "four epochs of 250 cover 1,000 nodes exactly once")
check(epoch_nodes(nodes, 2, 250) == slices[2], "an epoch's slice is reproducible on resume")
check(epoch_nodes(nodes, 4, 250) == slices[0], "epoch 5 wraps around to the first slice")
check(epoch_nodes(nodes, 3, None) is nodes, "no --epoch-size is the whole set, as before run 9")

# --- the identifiability scorer ------------------------------------------------------------
golds = ["graph neural networks for citation recommendation on arxiv",
         "speech recognition with transformers on noisy audio",
         "reinforcement learning for robot grasping in clutter"]
check(list(ranks(golds, golds)) == [1, 1, 1], "a text identical to its gold ranks first")
check(list(ranks(["", "", ""], golds)) == [3, 3, 3], "an empty text ranks last, not first")
swapped = ranks([golds[1], golds[0], golds[2]], golds)
check(swapped[0] > 1 and swapped[1] > 1 and swapped[2] == 1, "swapped texts lose their rank")

# --- LoRA targets are read off the checkpoint -----------------------------------------------
keys = {"model.layers.0.self_attn.q_proj.lora_A.default.weight": 0,
        "model.layers.0.self_attn.k_proj.lora_B.default.weight": 0,
        "model.layers.3.self_attn.v_proj.lora_A.default.weight": 0}
check(targets_of(keys) == ["k_proj", "q_proj", "v_proj"], "a step 6-8 q/k/v checkpoint reads as q/k/v")

print("\nall run-9 / run-10a input checks passed")
