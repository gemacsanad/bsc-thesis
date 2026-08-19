# Adapter — graph-conditioned abstract generation

> **Note — the repository was reorganised on 2026-08-05.** Paths below are as they were when
> each section was written. The code is now under `src/`, datasets under `data/`, results under
> `experiments/`, and job scripts under `hpc/komondor/`. See [`README.md`](../README.md) for the
> current mapping. The reasoning and the numbers are unaffected.

Generate a paper's abstract from its citation neighbourhood. A GNN encodes the neighbourhood,
an adapter maps that encoding to soft prefix tokens, and the prefix is injected into a frozen
Vicuna-7B. The prefix is the only trainable channel into the LLM — there is no LoRA in this phase.

```
neighbourhood → GraphSAGE → adapter → k prefix tokens → frozen Vicuna-7B → abstract
```

**Stages:** 1 data prep · 2 graph tensors · 3 model · 4 training · 5 generation + eval · 6 ablations

Stage 5 onwards is documented in [`adapter_evaluation.md`](adapter_evaluation.md).

---

## Stage 1 — Data prep (`prepare_data.py`)

**Date:** 2026-07-27

### Goal

Fix, once and in writing, which nodes the adapter may train on and which are held out for
evaluation, so that every later stage reads the same sets and no result depends on an
undocumented sampling decision.

### Decisions

**Eligible node = `core == True` and `5 ≤ out_degree ≤ 15`.**
Out-degree counts outgoing edges in `joint_dataset/edges.csv`, where `src` cites `dst` — so
out-edges are the paper's own references. References are chosen over citations because they
exist at publication time; papers *citing* the target are published later, and conditioning on
them would leak the future. The 5–15 band keeps the neighbour text inside Vicuna's 4,096-token
context. `core == True` restricts to the original ogbn-arxiv nodes, which are the only ones
`splits.json` covers.

**Evaluation set = 495 nodes, 45 at each out-degree from 5 to 15**, drawn from the eligible
test nodes with `numpy` seed 42. Equal-sized bins mean quality can be plotted against neighbour
count without bin size confounding the trend. 495 rather than 500 because 500 does not divide
evenly across the 11 degree values, and exactly even bins are worth more than a round total.

### The S2 coverage finding

The eval set was originally going to be the 200 `is_test` nodes of
`ogbn_arxiv_dataset/nodes_subset.csv` (10 nodes at each out-degree 1–20), to stay comparable
with the earlier Vicuna-7B and GPT-5.4-mini runs in `test_data/`. Checking those nodes against
the joint graph showed why that does not work, and turned up a property of the dataset that
affects the whole project:

**ogbn-arxiv records only citations between arXiv CS papers. `joint_dataset` adds Semantic
Scholar bibliographies, which are the papers' real reference lists.** Among the 169,343 core
nodes, 94,756 (56%) have an S2 record and 74,587 (44%) do not:

| | count | mean out-degree |
|---|---|---|
| S2 bibliography present | 94,756 (56%) | 17.0 |
| ogbn-internal citations only | 74,587 (44%) | 6.9 |

Two consequences:

1. **The old 200-node set is unusable.** Stratified over ogbn degree 1–20, those nodes span
   out-degree 1–65 in the joint graph, and about half of them produce prompts that overflow the
   4,096-token context. Comparability with the earlier baselines is given up; all arms will be
   rerun on the new eval set.
2. **Out-degree does not mean the same thing for every node.** For an S2-covered paper it
   approximates the true reference count; for an uncovered one it is only the arXiv-CS slice of
   the bibliography. A low out-degree therefore means either a short bibliography or an unseen
   one — and 37% of the 65,609 eligible nodes are uncovered, so this is not a rare edge case.

Coverage was deliberately **not** made an eligibility criterion; the per-degree breakdown is
printed whenever the set is frozen so the confound stays visible in the results.

### Inputs / outputs

| | |
|---|---|
| In | `joint_dataset/nodes.csv`, `joint_dataset/edges.csv`, `joint_dataset/splits.json` (+ `ogbn_arxiv_dataset/edges.csv`, coverage report only) |
| Out | `adapter/eval_nodes.json` (495 node_ids), `adapter/splits_eligible.json` (eligible node_ids per split) |
| Run | `python adapter/prepare_data.py` |

### Results

Joint dataset: 384,756 nodes, 3,844,151 edges, 169,343 of them `core`.
**65,609 nodes are eligible.** Intersected with the splits:

| Split | Eligible nodes |
|---|---|
| train | 36,093 |
| valid | 12,020 |
| test | 17,496 |

Evaluation set: 495 nodes, 45 per degree. S2 coverage rises with out-degree, from 31% at
degree 5 to 64% at degree 15, 56% overall — the confound described above, now quantified on the
exact set that will be scored.

### Verification

Six assertions run on every invocation and abort the script on failure — all passed:
no duplicate eval nodes; every eval node is `core`, in the test split, and eligible; no eval
node appears in the eligible train or valid sets; and every degree bin has enough candidates
to sample from (the smallest, degree 15, has 1,306 available for 45 slots).

The disjointness of the splits makes the train/valid check structural rather than lucky.

Sampling determinism was checked by hand: seed 42 reproduces a byte-identical
`eval_nodes.json` across runs, seed 7 produces a different one, and returning to 42 restores
the original.

### Limitations and open questions

- **S2 coverage confound**, described above — measured, not removed.
- **Data artifacts in `joint_dataset/edges.csv`**: 20 self-loops, and mutual citation pairs
  (≈1.5% of a 500k-edge sample had their reverse also present). Neither is possible in a strict
  citation DAG. Harmless for message passing, but they weaken a "references only, temporally
  honest" claim and should be settled in Stage 2.
- **Message-passing direction is undecided.** Blocks Stage 2. Raw `edge_index` sends messages
  from citing to cited paper, so a 2-layer GraphSAGE on the unmodified direction would aggregate
  papers that cite the target — reintroducing exactly the temporal leak the out-degree filter
  was chosen to avoid. Out-edges only, reversed, or undirected has to be settled before
  `NeighborLoader` means anything.

### Correction to the original plan

The plan recorded 12,028 valid nodes. The measured count is **12,020**.

---

## Stage 2 — Graph tensors and dataloaders (`graph.py`)

**Date:** 2026-07-27

### Goal

Build the PyG graph the GNN trains on, and the loaders that feed it neighbourhoods.

### Decisions

**The graph is directed: references only.** `edges.csv` stores (citing, cited), and PyG
aggregates along in-edges, so `edge_index` is stored **reversed** as (cited, citing). Messages
flow from a paper's references into the paper, and a node learns only from work it cites.

This matches the situation at generation time: when a new paper's abstract is written, its
reference list exists but nothing has cited it yet. Aggregating incoming citations would feed
the model papers published *after* the target — information that would not be available in real
use. It also makes hop 1 exactly the node's reference list, so the graph agrees with the 5–15
out-degree filter from Stage 1 instead of cutting across it.

**All 1-hop and all 2-hop neighbours** (`num_neighbors=[-1, -1]`), no sampling cap.

**The 20 self-loops are dropped**, and edges are coalesced. A paper citing itself is a data
artifact, and with the seed's features masked in Stage 3 a self-loop would feed the mask vector
into its own aggregation.

**`Data` stays on CPU**; sampling runs there and only the sampled subgraph moves to the GPU.
Resident 1.6 GB features on the GPU would buy nothing for batches touching ~1,400 nodes.

### Why not undirected — a detour worth recording

Undirected edges were implemented first and then reverted. Two problems, both measured:

1. **Temporal leak.** Papers citing the target became 1-hop neighbours despite being published
   later.
2. **A heavy computational tail.** Undirected 1-hop pulled in incoming citations from hub
   papers (max degree 16,513), so full 2-hop neighbourhoods ran to 29,650 nodes at p99 and
   94,721 at worst — roughly 200× variation in batch memory.

An exact workaround was explored for problem 2: because SAGEConv layer 1 is linear in the input
features and those features are frozen, the mean of each node's raw neighbour features can be
precomputed once, letting the exact 2-layer output be computed from 1-hop neighbours alone
(verified to ~1e-6 against the full path). Switching to reference-only edges made it
unnecessary — the tail was an artifact of undirecting, not of the task — and avoids the
constraints it would have imposed (2 layers only, mean/sum aggregation only). Recorded because
the technique would apply again if a much denser graph is ever used.

### Inputs / outputs

| | |
|---|---|
| In | `joint_dataset/node_embeddings.npy`, `joint_dataset/edges.csv`, `adapter/splits_eligible.json`, `adapter/eval_nodes.json` |
| Out | none — a module: `load_graph()` and `make_loader()` are imported by later stages |
| Run | `python adapter/graph.py` builds the graph and prints stats |

### Results

Graph: 384,756 nodes, **3,844,131 directed edges**, features `(384756, 1024)` float32 (1.58 GB).
References per node: mean 10.0, max 832.

Loaders seed from train (36,093), valid (12,020) and the frozen eval set (495).

Neighbourhood size for eligible train seeds, measured over 3,000 samples:

| | mean | median | p99 | max |
|---|---|---|---|---|
| 1-hop (references) | 8.6 | 8 | 15 | 15 |
| 2-hop (references of references) | 50.6 | 36 | 215 | 370 |

Roughly 60 nodes per seed. A batch of 32 produced 1,406 nodes and 1,918 edges — no cap needed
anywhere, and batch size is limited by the LLM in Stage 4, not by the graph.

Hop 2 is genuinely populated: S2-only nodes carry their own references (mean out-degree 8.0),
10% of 1-hop neighbours are dead ends, and only 0.3% of seeds have an empty 2-hop neighbourhood.
This matters for the architecture — the prefix is supposed to carry 2-hop information the
1-hop prompt text does not.

### Verification

Four assertions run on build and abort on failure — all passed:

1. No self-loops survive.
2. No duplicate edges survive.
3. **Seed nodes occupy rows 0..batch_size-1** of each batch — what Stage 3's `[MASK]`
   substitution depends on.
4. **A seed aggregates exactly the papers it cites in `edges.csv`** — the direction check.
   Edge direction is the one thing here that could be silently backwards and still run, so it is
   asserted against the raw file rather than assumed.

### Limitations and open questions

- **10% of 1-hop neighbours have no references of their own**, contributing nothing at hop 2.
- **Local environment.** `NeighborLoader` needs `pyg-lib` or `torch-sparse`; neither ships a
  macOS ARM wheel, so `torch-sparse` and `torch-scatter` were built from source into the
  `DyTAG` conda env. The Komondor container needs the same dependency.
- **Row alignment of `node_embeddings.npy` is assumed, not re-verified here.** `embed_nodes.ipynb`
  asserts `node_id == row index` at write time. Re-embedding a node through Voyage would confirm
  it end-to-end, but the `VOYAGE_API_KEY` in `.env` is currently rejected as invalid.

---

## Stage 3a — GraphSAGE encoder (`model.py`)

**Date:** 2026-07-27

### Goal

Encode a target node's reference neighbourhood into a single vector. This is the encoder only —
the adapter that turns this vector into soft prefix tokens is a separate piece, still to come.

### Decisions

**Two SAGEConv layers, 1024 → 1024 → 1024, mean aggregation.** Constant width keeps the
representation as wide as the input; input dim is fixed at 1024 by the voyage embeddings. Layer 1
sees the node's references, layer 2 the references of those references — matching the design
where prompt text carries 1-hop and the prefix carries 2-hop.

**A learned `[MASK]` vector replaces the seed's own features.** The seed's embedding is the
voyage encoding of the very abstract being generated, so leaving it visible would let the model
copy the answer. A trainable parameter shared across seeds lets the model learn a representation
meaning "this node is the unknown target", which zeros could not express. Initialised to unit
norm to match the L2-normalised voyage features it stands in for.

**Mean aggregation** (PyG's `SAGEConv` default). It suits this data unusually well: the features
are unit-norm semantic embeddings, so the mean of a node's ~9 references is a genuine centroid
of what those papers are about, and it is invariant to set size, which matters when degree
varies 5–15. `sum` was rejected because it entangles reference count with content — output norm
would vary 3× across the degree band, feeding an adapter that has to emit prefix vectors at a
stable scale. The order-dependent aggregators (`lstm`, `gru`, `sort`) were rejected because
references have no canonical order.

**Dropout 0.2** between the layers — lighter than the 0.5 in
`GraphSAGE_classification/train_graphsage_full.py`, because the GNN is small relative to 36,093
training nodes and heavy dropout on a ~9-neighbour aggregation is a large perturbation.

**No activation on the output layer.** A final ReLU would force the vector handed to the adapter
to be non-negative, discarding half the representational space for no benefit.

### Inputs / outputs

| | |
|---|---|
| In | batches from `graph.py` (`batch.x`, `batch.edge_index`, `batch.batch_size`) |
| Out | none — a module: `GraphSAGEEncoder` returns `(batch_size, 1024)`, one embedding per seed |
| Run | `python adapter/model.py` runs a forward pass on one batch |

### Results

**4,197,376 trainable parameters.** A batch of 8 seeds (362 nodes, 504 edges) produced a
`(8, 1024)` output with vector norms in 0.774–0.802.

### Verification

Two assertions, both passed:

1. Output shape is `(batch_size, 1024)`.
2. **Randomising the seed rows of `batch.x` leaves the output bit-identical** — proof that the
   mask is actually applied and the target's own embedding cannot reach the result. This is the
   failure that would be most damaging and least visible: without it the model could copy the
   answer and every metric would look excellent.

### Limitations and open questions

- **Masking depends on seeds occupying rows 0..batch_size-1**, which `graph.py` asserts. If the
  loader is ever changed, that assertion is what protects this.
- **The mask is one shared vector**, so every seed's layer-1 root term is identical; a seed's
  representation comes entirely from its neighbourhood. That is intended, but it means the model
  cannot condition on anything about the target itself, not even its year or venue.
- **Uniform averaging weights irrelevant references equally.** The `cosine_filter` work labelled
  2,100 edges with GPT and found 845 relevant vs 1,255 *not* directly relevant — so roughly 60%
  of references may be noise that mean aggregation treats the same as the rest. Measured on the
  old ogbn-internal subset, so indicative rather than exact. This is the concrete motivation for
  the attentional-aggregation ablation, and it connects to the neighbourhood-filter extension in
  the project plan.
- **Aggregation is a one-line ablation** for Stage 6: `AttentionalAggregation` (learned weight per
  reference, motivated above) and `aggr=['mean','max']` are the two candidates worth measuring
  against the mean baseline.

---

## Stage 3a addendum — what the `[MASK]` vector is, and seed–seed collisions

**Date:** 2026-07-28

### What the mask is, and what it is not

It is an **input-side placeholder**, not an initialisation or prior for the target's embedding.
The model never predicts an embedding of the abstract being generated: there is no reconstruction
loss, the encoder output is a *neighbourhood* encoding, and the abstract itself is produced
downstream by the LLM. There is no "target embedding" for the mask to initialise.

Its job follows from the `SAGEConv` update, which is a root term plus a neighbour term:

> h_i = W_root · x_i + W_neigh · mean_{j ∈ N(i)} x_j

For a seed, `x_i` *is* the voyage embedding of the abstract being generated, so `W_root` would hand
the model the answer directly. Replacing that row removes the path. Layer 2 is deliberately **not**
re-masked: `h_seed` from layer 1 is already free of the target's features, so `conv2`'s root term is
both safe and — unlike layer 1's — genuinely informative, since it carries the 1-hop aggregate.

### What it learns

Because one vector is shared across all seeds, its whole contribution to layer 1 is the constant
`W_root · mask`, identical for every seed in every batch. **It cannot carry node-specific
information.** Its function is to be a learned "this row is the unknown target" flag — the only
thing distinguishing a seed row from an ordinary node row at layer 1 — and to place the seed's
pre-activation in a useful region of the ReLU. Zeros could not do this: they leave only `conv1`'s
bias, which applies identically to every row. The upside is real but bounded; expect a small
effect, not a representational one.

### Masking is batch-local

`model.py` slices on `batch.batch_size`, which is the number of seeds **in the current batch**, not
in the split. A train seed sitting in some other batch appears here as an ordinary sampled
neighbour with its features intact.

`NeighborLoader` deduplicates node rows, so when a seed is also a neighbour of another seed *in the
same batch*, both share one row — the seed row — and the neighbour therefore aggregates the mask
instead of that paper's real embedding. Verified on a 6-node synthetic graph where node 1 is a
reference of node 0 and both are seeds: at `batch_size=2` node 1 lands at row 1 and is masked; at
`batch_size=1`, same node and same row, it falls outside the seed range and keeps its real feature.

Collision rate over 400 shuffled batches of eligible train seeds, counting 1-hop and 2-hop:

| batch size | batches with a collision | seed appearances affected |
|---|---|---|
| 8 | 1.2% | 0.19% |
| 32 | 19.2% | 0.79% |
| 64 | 60.5% | 1.79% |

### Both consequences of masking by row position

**The cost.** A seed loses one real reference to a constant vector, and the forward pass becomes
dependent on batch composition. This is a pure loss with no compensating benefit — a paper cited by
the target genuinely exists when the target is written, so its features are legitimate signal. It
errs in the safe direction, though: information is removed, never added, so it cannot leak. Note
the mask is a *constant*, not noise, so the effect is a slight fixed bias in the centroid rather
than added gradient variance. At `batch_size=32` it touches 0.79% of seed appearances per epoch,
well inside the noise floor of `DROPOUT = 0.2`.

**The benefit.** It closes the self-leak path opened by the mutual citation pairs recorded in
Stage 1 (≈1.5% of a 500k-edge sample). If A cites B and B cites A, then A sits at its own 2-hop;
row-based masking means B aggregates the mask rather than `x_A`, so the target's own embedding
still cannot reach it. Masking by "hop-0 node" instead of by row would have left that open.

**Decision: not fixed.** The cheap alternative — masking by node id rather than row position —
would trade a sub-1% input loss for reopening the mutual-citation leak, which is the more damaging
of the two. Recorded rather than engineered around.

### Limitations

- **The cost scales with batch size** — 60.5% of batches at 64. If Stage 4's LLM memory budget
  forces larger graph batches, this needs re-measuring.
- **The `model.py` smoke test cannot catch this.** It randomises seed rows and asserts the output
  is unchanged; a collision only makes that invariant hold *harder*.
- **Collision rates are measured on the train split** with one shuffle seed, as an order of
  magnitude rather than an exact figure.

---

## Stage 3b — the prefix adapter

**Date:** 2026-07-30

### Goal

Turn the encoder's 1024-d neighbourhood vector into the key/value prefixes injected into every
layer of a frozen Vicuna-7B-v1.5. Prefix tuning was chosen over input-only soft tokens so the graph
signal enters at all 32 layers rather than having to survive propagation from the embedding layer.

### Why a naive adapter is impossible

Vicuna-7B is Llama-2-7b: 32 layers, 32 attention heads, `head_dim` 128, **no GQA**, so keys and
values are each a full 4096 per token per layer. At 10 prefix tokens:

| | |
|---|---|
| numbers per (layer, role) | 10 x 4096 = 40,960 |
| (layer, role) pairs | 32 x 2 = 64 |
| total prefix numbers per node | 2,621,440 |
| naive: 64 independent `Linear(1024 -> 40960)` | **2.68 B parameters** |

The cost is in the **output**, not the input: any linear map ending in 2.6M numbers costs
`bottleneck x 2.6M`, so shrinking the 1024-d input alone saves nothing. Even a single shared
`Linear(1024 -> 40960)` is 42M, and a bottleneck of 8 still costs 21M.

### Design

Three factorisation axes, of which the layer axis does the real work:

```
g (1024)
 -> trunk    Linear(1024->768) -> GELU -> LayerNorm        z (768)
 -> codes    Linear(768 -> 32*2*96), reshape               c[layer, role] (96)   x64
 -> decoder  Linear(96 -> 10*4096)  SHARED across all 64   delta[layer, role]

prefix[layer, role] = base[layer, role] + gate[layer, role] * delta[layer, role]
```

Giving each of the 64 heads a small code and expanding it with **one shared decoder** replaces
`64 x (768 x 40960)` with `64 x (768 x 96)` plus a single `96 x 40960`.

`base` is a node-independent learned prefix — classic prefix tuning, carrying the task format the
graph should not have to re-derive 64 times. `gate` is one scalar per (layer, role), **initialised
to zero**, so training starts as plain prefix tuning with the graph disconnected.

Output shape is `(batch, 32, 2, 32, 10, 128)`, keys at index 0 of dim 2. Stage 4 converts it to
`past_key_values`; `model.py` deliberately does not import transformers.

### Parameters

| block | params |
|---|---|
| trunk | 788,736 |
| codes | 4,724,736 |
| decoder (shared) | 3,973,120 |
| base | 2,621,440 |
| gate | 64 |
| **adapter** | **12,108,096** |
| GraphSAGE encoder | 4,197,376 |
| **trainable total** | **16,305,472** |

221x smaller than the naive design, against 36,093 training nodes.

### Decisions and their reasoning

- **`trunk_dim` 768, `code_dim` 96.** Chosen to lose less of the 1024-d input than the 512/64
  variant first considered, which came to 7.9M. Each layer reads the graph through only a 96-d
  projection, but 64 x 96 = 6144 code dimensions across all layers, comfortably above 1024 — so
  *collectively* the layers can read the whole encoder output, and nothing is lost globally.
- **One decoder shared by all 64 heads**, rather than per-layer or per-group decoders. Per-layer
  decoders would cost 64 x 3.97M = 254M. Layer specialisation has to come through the code and the
  base instead.
- **One `codes` matrix for all heads** rather than a layer embedding fed to a shared MLP. The
  embedding version costs ~37k instead of 4.7M, but the single matmul is simpler; noted as the
  obvious place to cut if capacity turns out to be the problem.
- **`base` initialised at std 0.02**, Llama's own `initializer_range`. These vectors live in
  post-RoPE key space, which has no natural scale to match, so the model's own init constant is the
  least arbitrary choice available.
- **Zero-init gate.** At step 0 the graph cannot affect generation at all, so training starts from
  a stable known regime; and the main ablation ("is the graph doing anything, or is the LLM
  paraphrasing the 1-hop abstracts already in the prompt?") becomes `gate = 0` on the *same*
  weights, not a separately trained model.
- **10 prefix tokens.** Not a capacity constraint: `g` has 1024 dimensions, so every prefix the
  adapter can emit lies on a <=1024-dim manifold regardless of token count, and 40,960 slots per
  layer is already 40x over-provisioned. What the token count buys is the number of distinct
  positions attention can point at.

### Verification

`python adapter/model.py` runs a real 8-seed batch end to end and asserts: output shape; that at
`gate=0` two different graph vectors give a bit-identical prefix equal to `base`; that opening the
gate changes it; and that gradients reach every parameter of both modules. All pass.

### Open questions and limitations

- **The `past_key_values` mechanics are unverified.** `transformers==4.39.3` sits mid-transition
  between the legacy tuple-of-tuples cache and `DynamicCache`, and `generate()` deriving
  `cache_position` from a pre-populated cache is the classic place `position_ids` end up offset by
  `prefix_len`. This is Stage 4's first job and must be tested, not assumed.
- **The prefix is position-free.** HF Llama caches already-rotated keys, so an injected prefix
  bypasses RoPE entirely. Standard for prefix tuning, and harmless since the vectors are learned in
  that space — but it means the 10 tokens have no order.
- **`gate=0` is prefix tuning, not a true no-op.** With `base` near zero the values contribute
  almost nothing, but the 10 prefix positions still absorb some attention mass, slightly diluting
  the real tokens. The ablation is honest as "graph off", not as "prefix absent".
- **PEFT cannot be used.** `PrefixTuningConfig` builds its prefix from an unconditioned embedding
  table, so the cache has to be constructed by hand.
- **No regularisation in the adapter.** The encoder has `DROPOUT = 0.2`; the adapter has none.
  Untested whether 12.1M adapter params overfit 36,093 nodes.
- **Whether per-layer conditioning earns its keep is unmeasured.** The 2.6M-number output exists
  only because the graph enters at all 32 layers. Zeroing gates for layers 1-31 tests this cheaply
  and should be run before the design is defended.

---

## Housekeeping — `adapter/config.py`

**Date:** 2026-07-30

Every module-level constant in `prepare_data.py`, `graph.py` and `model.py` now lives in
`adapter/config.py`, grouped by stage. The scripts import what they need; no constant is defined in
two places any more (`SCRIPT_DIR` and `ROOT` had been duplicated across all three).

What did **not** move: values already exposed on the command line — input paths, `--batch-size`,
the `--seed 42` used to freeze the eval set — stay as argparse defaults in the script that owns
them, since they have a working home already and moving them would mean routing CLI defaults
through an import for no gain.

Verified by re-running all three scripts. `prepare_data.py` was run with `--out-dir` pointing at a
scratch directory and its two JSON outputs diffed byte-for-byte against the committed ones
(identical, so the frozen eval set has not moved). `graph.py` and `model.py` pass every assertion,
with parameter counts unchanged at 4,197,376 and 12,108,096.

Note the folder uses flat imports (`from config import ...`, `from graph import ...`), which assume
the scripts are run from inside `adapter/` or with it on `sys.path`. That was already true of
`model.py` importing `graph`; `config.py` follows the same convention rather than introducing a
package. If Stage 4 wants `from adapter.model import ...`, this needs an `__init__.py` and relative
imports throughout.

---

## Stage 4, step 1 — verifying prefix injection (`verify_prefix_injection.ipynb`)

**Date:** 2026-08-01

### Goal

Establish that a hand-built KV cache is treated by `transformers==4.39.3` exactly like real
tokens, before any training code is written. Stage 3b listed the `past_key_values` mechanics as
unverified; everything downstream rests on them, and the failure mode is **silent** — a wrong mask
or offset produces plausible logits and a decreasing loss, not an exception. A model trained
against a misread prefix would be unfalsifiable after the fact.

### Method, and why it is trustworthy

Every check compares **two runs of the same model**, so the weights are irrelevant and a 4-layer
random Llama exercises the same `forward`, `_update_causal_mask`, RoPE and `DynamicCache.update` as
Vicuna-7B. The notebook therefore runs on a laptop CPU in seconds. The local `DyTAG` env has
`transformers==4.39.3`, the same version the Komondor container pins, and the notebook asserts this.

The ground truth is text: split a token sequence into `prefix ++ body`, then compare a plain
forward over the whole sequence against a forward over `body` with the *real* key/values of
`prefix` injected. A zero prefix would prove nothing — a broken mask would still give plausible
numbers — so the injected cache has to be one whose correct effect is already known.

### Results

Injection works, but only under a specific recipe. Agreement with the plain forward pass is
**3.3e-07**, and greedy generation matches token-for-token.

| | requirement | what happens otherwise |
|---|---|---|
| cache type | `DynamicCache`, built via `.update()` | a legacy tuple raises `AttributeError` under `use_cache=False` |
| `attention_mask` | length `P + L` | a length-`L` mask raises; omitting it is correct only for unpadded batches |
| `cache_position` | pass `arange(P, P+L)` explicitly when `use_cache=False` | **max logit error 0.74, silently** |
| `position_ids` | free choice — but train and generate must agree | prefix read at an offset it was not trained for |
| cache lifetime | rebuild every step | a forward pass appends the body's keys: length 10 → 26 |
| `generate()` | override `prepare_inputs_for_generation` | different tokens; passing `cache_position` to `generate()` is ignored |

Gradients reach every adapter parameter through the frozen LLM, and **zero** LLM parameters carry a
gradient — `DynamicCache.update` uses `torch.cat`, which keeps the graph connected.

### The two findings that change Stage 4

**1. `cache_position`, not `position_ids`, is the real hazard.** `LlamaModel.forward` computes
`past_seen_tokens` from the cache *only inside* `if use_cache:`. With `use_cache=False` it stays 0,
so `cache_position` defaults to `arange(0, L)` while the key axis is `P + L` long — and
`cache_position` is what sizes the causal mask. Every row shifts by `P`, so each token attends
mostly to prefix positions instead of to the tokens before it. This matters because
`use_cache=True` is **incompatible with gradient checkpointing**, which `LlamaModel.forward`
silently turns off — i.e. the corruption appears exactly in the configuration training runs in.

**2. `generate()` cannot be used unmodified.** Every search method in `generation/utils.py` begins
with `model_kwargs["cache_position"] = torch.arange(cur_len)`, where `cur_len` is the length of
`input_ids` alone. It **overwrites** any `cache_position` passed in and ignores the pre-filled
cache, so generation hits the same corruption on every step and cannot be fixed by argument.
Overriding `prepare_inputs_for_generation` — the last thing to touch `cache_position` before the
forward pass — repairs every step including incremental decoding; the override must name its
catch-all `**kwargs`, because `_validate_model_kwargs` inspects the signature by name.

### Correction to Stage 3b

Stage 3b recorded the `position_ids` offset as the expected failure ("the classic place
`position_ids` end up offset by `prefix_len`"). **That was wrong.** Llama's RoPE is relative, so
shifting every real token by the same `P` leaves token-to-token attention unchanged — measured at
3.6e-07 against a run with no cache. The offset only changes how queries see the *prefix*, whose
keys are position-free (0.041 max logit difference between the two conventions). It is a design
choice to be applied consistently, not a correctness bug.

### Inputs / outputs

| | |
|---|---|
| In | `adapter/config.py`, `adapter/model.py` (`PrefixAdapter`) — no data, no checkpoint |
| Out | none — a verification artefact. `prefix_to_cache()` and the `prepare_inputs_for_generation` override are the code Stage 4 lifts from it |
| Run | open in Jupyter with the `DyTAG` env, from inside `adapter/` (flat imports); ~10 s on CPU |

### Limitations

- **Random weights, tiny dimensions.** Verifies the code path, not the real checkpoint. Whether
  Vicuna-7B loads, and whether bf16 or 8-bit numerics behave, is the first Komondor run's job.
- **Batch size 1, no padding.** Padded batches need a real `P + L` mask; that combination is
  untested here.
- **Gradient checkpointing itself is not exercised** — the notebook infers the `use_cache=False`
  requirement from it, and tests that setting directly, but never enables checkpointing.
- **`sdpa` attention only** (the CPU default). The GPU run may select a different implementation.

---

## Stage 4 — training the adapter (`text.py`, `train.py`)

**Date:** 2026-08-01

### Goal

Train the GraphSAGE encoder and prefix adapter against a frozen Vicuna-7B, so that the graph
neighbourhood conditions the generated abstract. 16.3M trainable parameters; the LLM never
receives a gradient.

### Task framing

The prompt contains the **1-hop reference titles and abstracts and nothing about the target** —
not even its title. The model must infer the topic from the neighbourhood alone, which is both
the hardest version of the task and the one that leaves the most room for the graph prefix to
matter. The Vicuna-v1.1 chat format is carried over verbatim from `scripts/generate_test_csv.py`,
so results stay comparable with the earlier Vicuna and GPT-5.4-mini baselines in `test_data/`.
Loss is cross-entropy on the target abstract's tokens only; the prompt is masked with `-100`.

### The training sets

| set | size | how |
|---|---|---|
| train | 4,000 | seeded sample of eligible train nodes whose sequence fits the context |
| valid | 495 | 45 at each out-degree 5–15, mirroring the eval set's construction |
| test | 495 | unchanged — the frozen `eval_nodes.json` |

Nodes whose prompt + target + 10 prefix tokens exceed 4,096 are **dropped**, so every training
sequence is complete. The alternative — truncating neighbour text — would have given
high-degree nodes systematically less text per reference, which is a worse confound than a mild
shift in the degree distribution. Candidates are shuffled and walked until enough survivors are
found, which matches sampling from the fitting set but tokenises a few thousand nodes rather
than all 48,000.

Validation is 495 rather than a round 500 because 500 does not divide across 11 degrees, and
equal bins are worth more than a round total — the same reasoning that fixed the eval set at 495.

### The training recipe

Each part of this fails **silently** if dropped; all of it comes from
`verify_prefix_injection.ipynb`:

- `PrefixCache` — a read-only cache that concatenates the prefix and stores nothing. A
  `DynamicCache` cannot be used: gradient checkpointing recomputes every layer, and a mutating
  cache appends the body's keys a second time (10 → 26 → 42 positions).
- `attention_mask` of length `10 + L`, covering the prefix.
- explicit `cache_position = arange(10, 10 + L)`, because `use_cache=False` leaves
  `past_seen_tokens` at 0 and the causal mask is then built as if the prefix were absent.
- `use_reentrant=False`, because with a frozen LLM no tensor input to layer 0 requires a
  gradient and reentrant checkpointing refuses to build a graph.

Positions run `10 .. 10+L-1` (the default, `position_ids = cache_position`) — the same
convention the patched `generate()` produces. Train and generate must agree, or the prefix is
read at an offset it was never trained for.

**Hyperparameters:** AdamW, lr 1e-4, 100-step linear warmup then constant, weight decay 0
(decaying `base`/`gate` would pull the prefix toward the ablation), grad clip 1.0, batch 4 with
gradient accumulation 4 → effective 16. Batch 4 also keeps the seed-collision rate from the
Stage 3a addendum near 0.2%.

**The gate is left to learn freely** from zero — no warmup, no forced opening. If it stays near
zero, that is a real result about the graph rather than an artefact of the schedule. Validation
therefore reports the loss twice, with the learned gate and with the gate forced to zero: the
"graph off" ablation on identical weights, measured every evaluation instead of in a separate run.

### Budget on one A100 40 GB

| | bf16, no checkpointing | bf16 + checkpointing |
|---|---|---|
| frozen weights | 13.5 GB | 13.5 GB |
| activations, batch 1 @ 2,500 tok | ~14 GB | ~0.7 GB |
| same @ 3,800 tok (p90) | ~22 GB | ~1.0 GB |
| **peak, batch 1** | **28–36 GB** | **~15 GB** |

Checkpointing costs one extra forward (~4 forward-equivalents, ~5.4e10 FLOPs/token), giving an
estimated **~1.1 s per 2,500-token sequence** at ~40% MFU — about **1.2 h per epoch** over 4,000
nodes, so two epochs fit one 6 h job. A full 36,093-node epoch would be ~11 h and needs the
resume path.

### Verification

The real model cannot run locally, so every check uses a 4-layer random Llama with the **real
Llama tokenizer** (same 32,000 vocab, BOS 1, EOS 2) and a synthetic graph, through the actual
functions in `train.py` and `text.py`.

| check | result |
|---|---|
| full `main()`: 2 epochs, validation, checkpoint, resume | runs; resume restores epoch and step |
| gradient checkpointing genuinely active | `llm.model.gradient_checkpointing == True` |
| padding is neutral (22- vs 62-token prompts, 40 tokens of padding) | batched loss vs token-weighted singles differ by **2.4e-04** |
| the prefix can learn (overfit one batch, 30 steps) | **10.26 → 7.95** |
| labels aligned, target abstract and title absent from the prompt | asserted |
| `fits()` boundary | rejects at 4,087 tokens, accepts 4,086 |

### Deviations from the plan

- **`PrefixCache` lives in `train.py`, not `model.py`.** It is training-only — generation uses a
  pre-filled `DynamicCache` — and keeping it out preserves the documented property that
  `model.py` does not import `transformers`.
- **Validation is 495, not 500**, for the equal-bins reason above.

### Inputs / outputs

| | |
|---|---|
| In | `joint_dataset/nodes.csv`, `edges.csv`, `node_embeddings.npy`, `adapter/splits_eligible.json` |
| Out | `adapter/training_sets.json` (frozen train/valid ids), `adapter/checkpoints/epoch*.pt` (encoder + adapter + optimizer, ~196 MB — never the LLM) |
| Run | `python adapter/text.py` once, then `python adapter/train.py --epochs 2 --batch-size 4` |

### Blockers before the first GPU run

- **The container has no PyG.** `Komondor_files/baseline-gpu.def` installs transformers, peft,
  accelerate and bitsandbytes but not `torch_geometric`, and `NeighborLoader` needs it plus
  `pyg-lib`/`torch-sparse`. Linux wheels exist for torch 2.2 + cu121, so this is a container
  rebuild, not the source build the macOS environment needed.
- **`text.py` cannot run locally.** Vicuna's tokenizer ships only as a sentencepiece model with
  no `tokenizer.json`, and `sentencepiece` is not in the local `DyTAG` env (it *is* in the
  container). Run `text.py` on Komondor, or install sentencepiece locally.

### Limitations

- **Nothing has run on a real GPU or the real checkpoint.** Memory and throughput are estimates;
  bf16 numerics, the actual attention implementation and OOM behaviour on p99-length sequences
  are unverified.
- **Tokenisation happens per batch**, repeated every epoch. Negligible against ~1.1 s of GPU
  time per sequence, but it is wasted work.
- **No early stopping and no LR decay** — constant LR after warmup, fixed epoch count.
- **The 4,000-node subset is a first run, not a final model.** Its purpose is to answer whether
  the loss moves and whether the gate lifts.

---

## Stage 4b — metric tracking with Weights & Biases (`train.py`)

### Goal

Komondor jobs are batch jobs: without live metrics the only feedback is a log file read after
the fact, and a run that has stopped learning burns its whole 6 h allocation before anyone
notices. W&B also makes runs comparable across the ablations planned for Stage 6, which is what
the thesis actually needs from them.

### What is logged

Opt-in via `--wandb` (default off), so CPU smoke tests and debug runs stay dependency-free.
`--wandb-project` (default `dytag-adapter`) and `--wandb-run` name the run.

| series | why it is there |
|---|---|
| `train/loss`, `train/lr`, `train/tokens_per_sec`, `train/epoch` | the basics; tok/s also confirms the A100 estimate of ~1.1 s per 2,500-token sequence |
| `train/grad_norm` | the pre-clip norm returned by `clip_grad_norm_`. Free, and says whether `GRAD_CLIP = 1.0` is binding or inert |
| `valid/graph`, `valid/gate0`, `valid/delta` | the graph-vs-no-graph ablation, already computed by `validate()` on identical weights. `delta` is the central claim of the thesis, watched during training rather than discovered at the end |
| `gate/mean_abs`, `gate/max_abs`, `gate/per_layer` (histogram) | the gate is the switch that lets the graph in. The histogram over the 32×2 entries shows *where* in the stack the graph gets used — a figure on its own |
| `prefix/graph_fraction` | mean ‖prefix − base‖ / ‖base‖ over the batch: the share of the prefix the graph supplies, rather than the node-independent part |

`graph_fraction` is a new helper in `train.py`. Because `gate` starts at zero, `prefix - base` is
exactly the graph's contribution, and the metric is 0 at initialisation by construction.

`step_loss` now returns `(loss, prefix)` rather than just the loss, so the prefix statistics come
from the forward pass that was already run instead of a second one. The value logged is a
snapshot from the last micro-batch of the accumulation window, not an average over it.

### Verification

Real W&B is not installed locally, so the code path was exercised against a stub module that
records every call, through the actual `main()`.

| check | result |
|---|---|
| `init` receives project, run name and config | 21 config keys, including all hyperparameters and `trainable_parameters` |
| all twelve series reach `log` | asserted by name |
| histogram construction | 4 built over 4 logging steps |
| `step` is monotonic across train and validation logs | asserted |
| `finish()` called exactly once | yes |
| `summary` carries the final numbers | `final/graph`, `final/gate0`, `final/delta` |
| `--wandb` absent | full run + resume complete with no wandb import |
| `graph_fraction` at `gate = 0` vs open gate | `0.000000` vs positive |

Observed on the tiny smoke model: with `gate` forced to 1.0, `graph_fraction` reaches ~130 — the
un-gated `delta` is far larger than `base` (std 0.02), so the gate only has to reach ~0.01 for the
graph to contribute on the same scale as the node-independent prefix. Expect the useful range of
`gate/mean_abs` to be small; that is not a sign the gate is stuck.

### Known limitations

- **No offline mode.** Training runs on internet-connected nodes, so `wandb` writes online; a
  node without egress would need `WANDB_MODE=offline` plus a later `wandb sync`.
- **Resume starts a new W&B run** rather than continuing the old one (`wandb.init(resume=...)`
  is not wired up). Step numbers stay monotonic, so the curves concatenate correctly, but they
  land in two runs.
- **Checkpoints are not uploaded as artifacts.** They stay on the filesystem.
- **`wandb` is in neither the local env nor the container** — see below.

---

## Stage 4c — the frozen subsets, measured (`training_sets.json`)

`text.py` ran locally on 2026-08-02 with the real `lmsys/vicuna-7b-v1.5` tokenizer. The only
thing that had blocked it was a missing `sentencepiece` in the local `DyTAG` env — one wheel,
no dependencies. `protobuf` was already present (as `google.protobuf`), so it was never a
second blocker. Komondor is no longer needed for this step.

`adapter/training_sets.json`: **4,000 train + 495 valid** node ids, no duplicates, no overlap
with each other or with the frozen 495-node eval set (asserted).

### Sequence lengths, in real Vicuna tokens

| set | mean | median | p99 | max |
|---|---|---|---|---|
| train (4,000) | 2,323 | 2,226 | 4,023 | 4,082 |
| valid (495) | 2,757 | 2,789 | 4,055 | 4,083 |

The A100 budget in Stage 4 assumed ~2,500 tokens per sequence; the measured train mean is
2,323, so the ~1.2 h/epoch estimate stands. Validation is ~19% longer than train, because it is
degree-stratified and length rises steeply with out-degree.

### Context overflow is concentrated at high out-degree

Training nodes: **5.9% overflowed** (4,000 kept from 4,253 examined). But the validation
collection, which draws a fixed 45 per degree, exposes where those losses sit:

| out-degree | examined for 45 | overflow | mean tokens |
|---|---|---|---|
| 5 | 45 | 0% | 1,590 |
| 10 | 46 | 2% | 2,850 |
| 13 | 53 | 15% | 3,390 |
| 14 | 62 | 27% | 3,576 |
| 15 | 108 | 58% | 3,650 |

At out-degree 15 more than half of all candidates do not fit 4,096 tokens. This matters for
interpretation: **the dropped nodes are not a random sample.** Within any given out-degree, the
nodes that survive are those with shorter abstracts among their references, so the training set
is biased toward terser neighbourhoods at the high-degree end — precisely the end where the
graph has the most to say. The 5.9% figure understates this because the eligible train split is
dominated by low degrees, where nothing is lost.

The frozen **eval set overflows at 13.1% (65/495)** — higher than train, since it is
degree-stratified. It is deliberately left unfiltered; Stage 5 has to decide whether to
truncate those 65 or report them separately, and either choice needs stating in the results.

### Open question

Whether to raise the effective budget for high-degree nodes — truncating each neighbour's
abstract rather than dropping the node — is unresolved. It would remove the sampling bias but
changes what the model reads, so it is a Stage 5/6 decision, not a silent fix here.

---

## Stage 4d — a separate container (`Komondor_files/adapter-gpu.def`)

### Why a new container rather than extending `baseline-gpu.def`

The first attempt added PyG and wandb to `baseline-gpu.def`. That was reverted. `baseline-gpu.sif`
is what produced the Vicuna baseline in `test_data/`, and `run_generate.sh` depends on it;
rebuilding it in place puts those results behind a container that no longer exists. A second
`.sif` costs disk and nothing else, and the two experiments stay independently reproducible.

`adapter-gpu.def` therefore stands alone rather than inheriting: same proven base image
(`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`), its own package list.

### Contents

| package | why |
|---|---|
| `transformers==4.39.3` | pinned exactly — the `past_key_values` injection depends on this version's cache internals (`DynamicCache`, `cache_position`, `_update_causal_mask`) |
| `torch_geometric==2.8.0` | `Data`, `NeighborLoader`, `SAGEConv`, `coalesce`, `remove_self_loops` |
| `torch_sparse==0.6.18+pt22cu121`, `torch_scatter==2.1.2+pt22cu121` | `NeighborLoader` refuses to sample without a backend |
| `sentencepiece`, `protobuf` | the Vicuna tokenizer is a sentencepiece model |
| `pandas` | `Corpus` reads `nodes.csv` / `edges.csv` |
| `wandb==0.28.0` | `--wandb`; same pin already proven on Komondor in `graphsage-gpu.def` |
| `accelerate`, `peft` | **not used by `train.py`** — carried so the LoRA stage does not force another rebuild |

`bitsandbytes` was dropped: the adapter trains in bf16, and 8-bit quantisation belongs to the
baseline generation container.

**PyG 2.8.0, not the 2.5.3 in `graphsage-gpu.def`.** 2.5.3 is known to build on Komondor, but
2.8.0 is the version every local verification ran against. The API surface used is five symbols
wide, so matching the verified environment was worth more than matching the older build.

**torch-sparse, not pyg-lib.** Both satisfy `NeighborLoader`; the local environment uses
torch-sparse, so this avoids introducing a sampler nothing has tested. pyg-lib is the faster
backend and is worth revisiting if CPU-side sampling bounds throughput.

**The `+pt22cu121` local versions are pinned in full.** A bare `==2.1.2` lets pip fall back to the
PyPI sdist and start a source compile inside the build; the full pin fails immediately instead.
Wheels for these versions exist on the PyG index for cp38 through cp312, so the pin resolves
whatever Python the base image ships.

A `%test` section asserts a sampler backend is present, numpy is still 1.x, and transformers is
exactly 4.39.3 — so a container that cannot run the pipeline fails at build time rather than in a
queued job.

### Building

Lustre cannot host a fakeroot build, so the image is built on node-local `/tmp` and copied to
`/project` afterwards.

```bash
module load singularity
cd /project/c_gnn42/csanad_thesis/baseline

singularity build --fakeroot /tmp/adapter-gpu.sif Komondor_files/adapter-gpu.def
cp /tmp/adapter-gpu.sif adapter-gpu.sif
rm /tmp/adapter-gpu.sif
```

Run this on a **login node** — `%post` installs from PyPI and compute nodes have no internet.
The `.def` may live on `/project`; only the build *output* has to be local.

No `SINGULARITY_TMPDIR` or `SINGULARITY_CACHEDIR` overrides are needed. The build sandbox already
defaults to `/tmp`, and the existing `baseline-gpu.sif` and `graphsage-gpu.sif` were built from
this same base image with the default `$HOME` layer cache. If a build ever does fail with "no
space left on device", that cache is the cause and `singularity cache clean` clears it.

Checking the copy, since `cp` off `/tmp` is where a truncated image would show up:

```bash
singularity exec adapter-gpu.sif python -c "import torch_geometric.typing as t; print(t.WITH_TORCH_SPARSE)"

srun --account=c_gnn42 --partition=ai --gres=gpu:1 --time=00:10:00 \
     singularity exec --nv adapter-gpu.sif python -c "import torch; print(torch.cuda.is_available())"
```

### Not verified

**This container has not been built.** macOS has no Singularity, so the only checks performed
were that every pinned wheel exists on the PyG index (confirmed for cp38–cp312) and that the
version constraints are mutually satisfiable on paper. `%test` runs automatically at the end of
`singularity build` and is the first real check; `--notest` skips it.

---

## Stage 4e — the training job (`Komondor_files/run_train_adapter.sh`)

Modelled on `run_generate.sh`: same account, partition, GPU request, `$BASE_DIR:/app` bind, and
the same offline HuggingFace environment (`HF_HOME=/app/.hf_cache`, `HF_HUB_OFFLINE=1`), since
compute nodes cannot download the Vicuna checkpoint.

```
--epochs 2 --batch-size 4 --grad-accum 4 --log-every 10 --eval-every 100 --wandb
```

4,000 train nodes at batch 4 is 1,000 micro-batches per epoch, so with `--grad-accum 4` an epoch
is **250 optimizer steps** — 25 log lines and 2 validations per epoch, and an effective batch of
16. Batch 4 is the conservative end of the 4–8 the memory estimate allows.

### Three things the script does beyond the obvious

**Walltime is 12 h, not the 6 h of `run_generate.sh`.** Two epochs are estimated at ~1.5 h each,
but that estimate has never been checked against a real A100 — it is derived from FLOP counts at
an assumed 40% MFU. 12 h absorbs a 4x miss. The cost is queue time, so lower it once one epoch
has actually been timed.

**It auto-resumes.** The newest `adapter/checkpoints/epoch*.pt` is found with `sort -V` and passed
as `--resume`, so a resubmit after a timeout continues rather than restarting. `train.py` restores
encoder, adapter, optimizer and scheduler and restarts at the next *epoch* — progress within an
interrupted epoch is lost, which is the price of checkpointing per epoch rather than per N steps.
With no checkpoint present the variable is empty and expands to nothing.

**The W&B key comes from `adapter/.env`.** `train.py` gained the same six-line `load_dotenv`
helper `GraphSAGE_classification/train_graphsage_full.py` already uses, reads `WANDB_KEY`, and
calls `wandb.login(key=...)`. The key therefore never appears in the sbatch script, the command
line, or the job log, and `.env` is covered by the repository's global `.gitignore`.

The login happens **immediately after argument parsing**, before the graph and the 13 GB
checkpoint are loaded — a missing or malformed key should cost seconds, not the two minutes it
takes to reach `wandb.init()`.

Setup is one line: `cp GraphSAGE_classification/.env adapter/.env`.

### Verified

The shell logic was exercised locally under `bash`: with no checkpoints the resume variable is
empty and adds no arguments; with `epoch1/2/10.pt` present `sort -V` selects **epoch10**, not
epoch2, and expands to two separate argv entries. Nothing else in the script can be tested off
Komondor.

The `.env` path was exercised through the full smoke run: a file containing a comment, a blank
line, and a single-quoted `WANDB_KEY` yields exactly that key at `wandb.login()`.

### Resolved: the partition does have egress

An earlier note here warned that `run_generate.sh` says compute nodes have no internet, and that
online W&B might therefore stall. `run_train_graphsage_full.sh` settles it — it runs on the same
`ai` partition and logs to W&B live. The offline-mode plumbing was removed as unnecessary; the
`HF_HUB_OFFLINE=1` in both job scripts is about not re-downloading a cached 13 GB checkpoint, not
about the node lacking a route.

---

## Stage 3b addendum — the adapter drawn, and what the gate actually does

**Date:** 2026-08-03

### Goal

Stage 3b records *why* the adapter is factorised the way it is. This section records *what it looks
like* end to end, and answers the question the design keeps raising when explained out loud: what
is `gate`, and why does a single scalar deserve its own parameter?

Nothing here is new code. It documents `model.py` as built.

### The architecture

```mermaid
flowchart TB
    subgraph ENC["GraphSAGEEncoder — 4,197,376 params"]
        direction TB
        X["batch.x — voyage embeddings<br/>N x 1024, N = seeds + 1-hop + 2-hop"]
        MSK["mask — learned parameter, 1024"]
        SUB["seed rows 0..B-1 replaced by mask<br/>so a node never sees its own abstract"]
        L1["SAGEConv 1024 to 1024, mean aggr<br/>ReLU, Dropout 0.2"]
        L2["SAGEConv 1024 to 1024, mean aggr<br/>no output activation"]
        G["g — first B rows of h<br/>B x 1024"]
        X --> SUB
        MSK --> SUB
        SUB --> L1 --> L2 --> G
    end

    subgraph ADP["PrefixAdapter — 12,108,096 params"]
        direction TB
        TR["trunk — 788,736<br/>Linear 1024 to 768, GELU, LayerNorm"]
        Z["z — B x 768"]
        CD["codes — 4,724,736<br/>Linear 768 to 6144, reshape"]
        C["c — B x 32 x 2 x 96<br/>one 96-d code per layer-role pair"]
        DEC["decoder — 3,973,120<br/>Linear 96 to 40960<br/>ONE instance, shared by all 64"]
        DL["delta — B x 32 x 2 x 32 x 10 x 128<br/>node-dependent"]
        BASE["base — 2,621,440<br/>32 x 2 x 32 x 10 x 128<br/>node-independent, init std 0.02"]
        GATE["gate — 64<br/>32 x 2 scalars, init ZERO"]
        MUL(["x"])
        ADD(["+"])
        PFX["prefix — B x 32 x 2 x 32 x 10 x 128"]
        TR --> Z --> CD --> C --> DEC --> DL --> MUL
        GATE --> MUL
        MUL --> ADD
        BASE --> ADD
        ADD --> PFX
    end

    subgraph LLM["frozen Vicuna-7B-v1.5 — 6.7 B params, no gradient"]
        direction TB
        PC["PrefixCache — read-only, concatenates and stores nothing"]
        TOK["prompt + target abstract, right-padded"]
        FW["32 decoder layers, bf16, gradient checkpointing"]
        CE["cross-entropy on the target tokens only<br/>prompt and padding masked to -100"]
        PC --> FW
        TOK --> FW
        FW --> CE
    end

    G --> TR
    PFX --> PC
```

Three shapes worth keeping straight: `g` is one vector per **seed node**; `c` is one small code per
**(layer, role)**; `delta` is a full prefix slab per (layer, role) per node. The whole factorisation
lives in the step from `c` to `delta`, where 64 different codes go through the *same* decoder.

The `transpose(3, 4)` at the end of `PrefixAdapter.forward` is not cosmetic: the decoder emits
`(tokens, heads, head_dim)`, matching how Llama reshapes a 4096-d projection, but the KV cache wants
`(heads, tokens, head_dim)`.

### The gate, plainly

`gate` is 64 scalars — one per (layer, role), i.e. 32 layers x {key, value}. It is a volume knob on
the graph:

```
prefix[l, r] = base[l, r]  +  gate[l, r] * delta[l, r]
               ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^
               identical for   depends on this node's
               every node      neighbourhood
```

- `gate = 0` -> `prefix = base` -> every node gets the same prefix. That is plain prefix tuning with
  the graph disconnected.
- `gate` large -> the graph dominates the prefix.
- One scalar per layer *and* role, so the model can decide the graph is worth a lot in layer 20's
  values and nothing in layer 3's keys. It is not a single global knob.

There is no sigmoid and no bound. It is a plain multiplier and can go negative — "gate" in the
zero-init residual sense (LayerScale, ReZero), not the LSTM sense.

### Why it is initialised to zero — three payoffs

**A known starting regime.** `delta` comes out of a randomly-initialised decoder. Propagating
torch's default `Linear` init through `trunk -> codes -> decoder`, delta's elements have std ~0.34
against base's 0.02 — roughly **17x larger**. Ungated, the prefix at step 0 would be almost entirely
noise from an untrained decoder, drowning the carefully scaled `base`. The gate makes step 0 exactly
prefix tuning instead.

**A free ablation.** Zeroing the gate at eval makes the prefix bit-identical to `base`, so "graph
off" is measurable on the *same* weights without training a second model. This is what `validate()`
does on every eval and what the `valid/gate0` curve is.

**A readable diagnostic.** `gate/per_layer` says *where* in the stack the graph is being used, which
no aggregate loss number can.

### The cost — the bootstrap problem

Because `prefix = base + gate * delta`:

```
d(prefix) / d(delta) = gate
```

So at `gate = 0` the entire graph pathway — encoder, trunk, codes, decoder, **13,683,968 of the
16,305,472 trainable parameters** — receives *exactly* zero gradient. It is frozen by arithmetic,
not by choice.

What still trains at step 0:

- `base`, since `d(prefix)/d(base) = 1`
- `gate` itself, since `d(loss)/d(gate[l,r]) = <d(loss)/d(prefix[l,r]), delta[l,r]>`, and `delta` is
  non-zero even from random weights

So the gate has to learn its way open **from a random delta**. It only lifts if a random projection
of the neighbourhood is slightly better than nothing. Once it does, a feedback loop starts: larger
gate -> real gradient reaches the encoder -> better delta -> larger gate.

That loop is visible in the first run's telemetry, and the three curves are one phenomenon:

| observation | step | reading |
|---|---|---|
| `gate/mean_abs` flat, then S-curve | ~0-50, then rising | the graph pathway is frozen, then unfreezes |
| `train/grad_norm` spike | ~85 | the unfreezing moment; `GRAD_CLIP = 1.0` binds only ~50-130 |
| `valid/gate0` reaches its minimum and turns up | 200 | `base` stops being optimised for its own sake |

### What the gate is *not*

- **Not per-node.** The same 64 scalars apply to every node in the batch. Node-dependence lives
  entirely in `delta`.
- **Not a mask over prefix tokens.** The 10 tokens inside one (layer, role) slab share a single
  scalar; the gate cannot switch individual prefix tokens on or off.
- **Not "no prefix" at zero.** `gate = 0` still injects `base` at 10 real attention positions, which
  absorb attention mass. The ablation is honest as "graph off", not "prefix absent" — as already
  noted in Stage 3b.

### Relationship to the logged metrics

`prefix/graph_fraction` is `mean ||gate * delta|| / ||base||`. At initialisation, with delta ~17x
base's element scale, that gives roughly `graph_fraction ~ 17 * mean|gate|` — so a small number on
the `gate/mean_abs` chart is not a weak effect. A `graph_fraction` of 3-4 corresponds to
`mean|gate|` of only ~0.2.

That 17x holds **at init only**; once training moves the decoder, delta's scale drifts. The honest
ratio at any later step is `prefix/graph_fraction` divided by `gate/mean_abs`, straight from the
W&B logs.

### Open questions

- **The bootstrap is a risk, not just a curiosity.** If a random delta happened to be useless, the
  gate would have no reason to open and the 13.7M-parameter graph pathway would stay frozen for the
  whole run. It worked here; it is not guaranteed to work under a different seed, a different
  `code_dim`, or with LoRA changing the loss landscape. Worth a seed sweep before the design is
  defended.
- **`valid/gate0` drifting upward after step 200 means `valid/delta` overstates the graph's
  contribution.** Once the gate opens, `base` is optimised as a centre for graph-conditioned offsets
  rather than as a good standalone prefix, so the ablation baseline degrades on its own. Over steps
  200-500 the first run's delta grew +0.044, of which +0.010 (~23%) is `base` getting worse rather
  than the graph getting better. A run with the gate frozen at zero throughout is the only
  stationary baseline.
- **Whether per-(layer, role) granularity earns its 64 parameters is untested.** A single global
  scalar is the obvious cheaper alternative; the `gate/per_layer` histogram should be checked for
  real spread before claiming the granularity matters.

---

## Stage 1b — the capped-prompt node sets (`prepare_data_capped.py`, `neighbour_cap.py`)

### Goal

Out-degree 5-15 was never a modelling choice. It was the context window: the whole reference
list went into the prompt, and 15 abstracts is as much as Vicuna's 4,096 tokens will hold. The
consequence is that the model has only ever been trained where the graph has least to say.

This stage separates the two things the ceiling was doing at once:

| | before | after |
|---|---|---|
| nodes the GNN may read | out-degree 5-15 | out-degree 5-100, **whole** neighbourhood |
| references shown in the prompt | all of them | the closest to the reference centroid, as many as fit |
| nodes lost to context overflow | ~13% of eval, 6% of train | none |

The GNN's view is now independent of what fits in the prompt. Out-degree becomes a measure of
available structure rather than a context-window constraint, which is the thing the prefix is
supposed to exploit.

### Which references, and how many

`neighbour_cap.py` reuses the score from `src/filters/centroid_cosine_distance.py`:
`(1 - cos(reference, centroid)) / 2` over the L2-normalised voyage embeddings, where the
centroid is the mean direction of that node's *own* reference set. The smallest distances are
kept — the references most representative of the bibliography. This is the same direction the
old filter used (`distance <= threshold`), with a per-node budget instead of a global
threshold, so the earlier `cosine-centroid-*` runs remain the precedent for the choice.

Two limits apply in order: at most **15** references, then the farthest of those is dropped one
at a time until the prompt fits

```
4,096  -  10 prefix tokens  -  max(256, the node's own abstract + EOS)
```

The reserve is the node's real abstract when that is longer than 256 tokens, so the training
sequence is always complete, and never less than 256, so generation has room to write —
matching `generate.py`'s `--max-new-tokens` default. **15 is a ceiling, not a target**, and it
has to be: measured on the previous freeze, ~45% of full 15-reference prompts did not fit
4,096 tokens. Trimming is what makes overflow structurally impossible rather than something
downstream has to handle. Kept references stay in `edges.csv` order, so a trimmed prompt reads
like an untrimmed one.

Alternatives considered and rejected: keeping the *farthest* references (maximises spread, but
throws away the core of the bibliography and keeps its outliers); stratifying across the
distance range (no precedent in this project's earlier filtering results); and dropping any
node that overflows, which was the first version of this stage and cost 45% of every
out-degree above 15 — biased, since the survivors are those whose central references happen to
have short abstracts.

### Sampling, and why the bins are not equal

Quotas are per out-degree, drawn separately from each split: 100 train, 10 valid, 10 eval
(eval from the test split). These are **ceilings, not guarantees** — the graph does not contain
enough high-degree nodes to fill them:

| out-degree | train | valid | test |
|---|---|---|---|
| 5 | 6,335 | 1,478 | 1,813 |
| 20 | 814 | 529 | 1,016 |
| 39 | 104 | 139 | 358 |
| 40 | 82 | 118 | 374 |
| 50 | 31 | 47 | 200 |
| 73 | 10 | 8 | 41 |
| 99 | 0 | 2 | 4 |

Train falls below 100 per degree from out-degree 40 upward and is empty at 99; valid falls
below 10 from out-degree 73. Taking everything available caps the sets at **4,597 train /
806 valid / 945 eval** — the quota arithmetic of 9,600 / 960 / 960 is not reachable. Because
nothing is now lost to overflow, these availability ceilings are exactly what the freeze
produced. Every degree above ~40 contributes every node it has, so any per-degree result read
off the high end rests on tens of nodes, not hundreds.

### What the trimming actually costs

Not nodes — references. Every eligible node is usable; what varies with out-degree is how much
of its bibliography is shown (measured 2026-08-06, train set):

| out-degree | nodes | references shown | prompt tokens |
|---|---|---|---|
| 5-9 | 500 | 7.0 | 2,016 |
| 10-15 | 600 | 12.1 | 3,315 |
| 16-29 | 1,400 | 13.9 | 3,724 |
| 30-39 | 1,000 | 14.1 | 3,704 |
| 40-59 | 879 | 14.3 | 3,670 |
| 60-79 | 162 | 14.3 | 3,656 |
| 80-100 | 56 | 14.4 | 3,495 |

Above out-degree 15 the prompt saturates: ~14 references and ~3,700 tokens regardless of
whether the paper cites 16 others or 100. Roughly **half of all prompts sit at the 15-reference
cap** (46% train, 49% valid, 47% eval); the rest were trimmed to 13 or 14 by long abstracts.
The minimum is 5, which is a node at out-degree 5 keeping everything, not a node trimmed to the
bone — no node lost so many references that the graph could not be shown.

This is the real shape of the experiment, and it should be said plainly: **beyond out-degree
~16 the prompt stops growing, so anything the model gains at higher degree has to come through
the GNN prefix**, which is the only channel still reading the other 85 references. That is
precisely the effect the adapter is meant to demonstrate, but it also means the text baseline
is nearly flat across the high-degree range by construction.

### Inputs / outputs

Reads `data/processed/joint/{nodes.csv,edges.csv,splits.json,node_embeddings.npy}`
(embeddings memory-mapped; only sampled rows are read) and writes to `data/frozen/`:

```
splits_eligible_capped.json   every eligible node per split (core, out-degree 5-100)
training_sets_capped.json     {"train": [...], "valid": [...]}
eval_nodes_capped.json        the eval nodes, from the test split
```

New files, not overwrites: `eval_nodes.json`, `splits_eligible.json` and `training_sets.json`
still back the two 495-node runs in `experiments/`, which stay reproducible.

Runs locally in the `DyTAG` env — CPU only, like `prepare_data.py` and `text.py` before it. No
Komondor job: the cluster is for the GPU stages.

```bash
cd src/adapter
python neighbour_cap.py          # check the capping score against the frozen filter column
python prepare_data_capped.py    # freeze the sets
```

`neighbour_cap.py` recomputes the distances for the ogbn subset and asserts they match the
frozen `centroid_cosine_distance` column — largest absolute difference **9.7e-17**, so the
score is the same quantity the earlier `cosine-centroid-*` runs filtered on. It then checks the
trimming itself against a synthetic 100-tokens-per-reference budget: 15 kept when the budget is
generous, 4 when it allows only 4, none when a single reference will not fit, always the
closest ones and always in edge order. Assertions in `prepare_data_capped.py` cover duplicates,
core membership, eligibility, test-split membership for eval, pairwise train/valid/eval
disjointness, and that every frozen sequence really fits `MAX_LENGTH`.

### What was frozen (2026-08-06)

111,538 nodes are eligible (48,479 train / 22,296 valid / 40,763 test). After the quotas:

| set | nodes | mean tokens | median | p99 | max | refs shown | degrees below quota |
|---|---|---|---|---|---|---|---|
| train | 4,597 | 3,466 | 3,729 | 4,075 | 4,086 | 13.0 | 61 of 96 |
| valid | 806 | 3,610 | 3,793 | 4,069 | 4,083 | 13.5 | 26 of 96 |
| eval | 945 | 3,655 | 3,823 | 4,078 | 4,086 | 13.6 | 3 of 96 |

**No node was lost.** The counts are exactly the availability ceilings from the table above, so
the only thing limiting these sets is how many high-degree papers the graph contains.

Comparable in size to the old 4,000 / 495, but spread over 96 out-degrees instead of 11, and
~1,100 tokens longer on average — the old train mean was 2,323, because most of its nodes sat
at low degree. Sequences now run close to the ceiling by construction, so the A100 time per
epoch will be worse than Stage 4c's ~1.2 h estimate; budget on ~3,500 tokens per sequence, not
~2,500.

One train degree (99) and two valid degrees (87, 97) contribute no nodes, because the split
holds none at all. The eval set covers all 96 degrees and falls short of 10 at only three of
them. Two eval nodes coincide with the old 495-node set — expected, since both are drawn from
the same test split.

### Not yet done

Only the node indices are frozen here. Nothing downstream reads them yet: `text.py`,
`graph.py`, `train.py` and `generate.py` still build uncapped prompts from the old sets, so
wiring `fit_refs` into the prompt path is the next step and no training has been run. Note that
`generate.py` computes its own budget (`MAX_LENGTH - prefix - max_new_tokens`) and must end up
with the same references the freeze measured, or the eval prompts will not be the ones these
sets were selected under.

### Open questions

- **Whether the top of the range is reportable at all.** Above out-degree 80 the train set holds
  56 nodes in total and one degree holds none. Any curve up there is anecdote, not measurement,
  unless the degrees are binned.
- **Whether a saturated prompt makes the high-degree comparison meaningful.** From out-degree 16
  upward the text prompt is effectively constant at ~14 references, so degree varies only what
  the GNN sees. That is the intended design, but it means a flat text baseline is guaranteed
  rather than measured.
- **Whether the centroid is the right anchor.** It is the reference set's own mean, so it is
  defined by the very neighbourhood being summarised; a node whose bibliography is genuinely
  two-topic gets one of its topics dropped by construction.

---

## Stage 1c — train quota raised to 200 per out-degree

### Goal and change

One constant: `CAPPED_TRAIN_PER_DEGREE` 100 → 200, to enlarge the training set. Valid and eval
were to stay exactly as frozen in Stage 1b, so any later result can be attributed to training
data rather than to a moved evaluation target. Nothing else changed — no code, only the config
value and its comment (the "quotas bind above out-degree ~39" note became ~34, measured).

The Stage 1b tables above describe the superseded 100-per-degree freeze and are kept as the
record of it; the numbers below are what `data/frozen/*_capped.json` now holds.

### Valid and eval are provably untouched

Worth stating why this was not automatic. All three sets are drawn from a single
`random.Random(42)` threaded through train → valid → eval, so a bigger train draw could have
shifted the stream and silently reseeded the other two. It does not: `collect` consumes
randomness only in `rng.shuffle(pool)`, which depends on the pool size, not on how many nodes
the quota keeps. Verified against a pre-run backup rather than argued:

```
valid identical: True (806 nodes)
eval  identical: True (945 nodes)
eligibility identical: True          (48,479 / 22,296 / 40,763)
old train is a subset of new: True   (0 nodes dropped)
train n valid = 0,  train n eval = 0
```

### What was frozen (2026-08-11)

| set | nodes | mean tokens | median | p99 | max | refs shown | at the 15-ref cap |
|---|---|---|---|---|---|---|---|
| train | **7,750** (was 4,597) | 3,428 | 3,714 | 4,075 | 4,086 | 12.8 | 42% |
| valid | 806 | 3,610 | 3,793 | 4,069 | 4,083 | 13.5 | 49% |
| eval | 945 | 3,655 | 3,823 | 4,078 | 4,086 | 13.6 | 47% |

+3,153 train nodes, a 69% increase. 7,750 is the exact pool ceiling: every degree contributes
either 200 or everything it has. The full quota is met at out-degree 5–33 only; from 34 the
train split runs out (171 at 34, 82 at 40, 31 at 50, 1 at 100, 0 at 99). 67 of 96 degrees are
below quota, against 61 before.

### The cost: train skews further from eval

The added nodes are not spread across the range — **all 3,153 sit at out-degree ≤ 39**, because
40 and above was already exhausted at the old quota and had nothing left to give. Doubling the
quota therefore buys depth only where the graph is already deep:

| | n | mean degree | median | share ≤15 | share ≥40 |
|---|---|---|---|---|---|
| old train | 4,597 | 29.1 | 27 | 24% | 24% |
| **new train** | **7,750** | **25.5** | **24** | **28%** | **14%** |
| the added nodes | 3,153 | 20.3 | 20 | 35% | **0%** |
| eval | 945 | 51.8 | 52 | 12% | 63% |

Train's mean out-degree falls 29.1 → 25.5 and its high-degree share is nearly halved
(24% → 14%), while eval is unchanged at mean 51.8 with 63% of its nodes at degree ≥40. The
train/eval degree mismatch that already existed is now wider, not narrower. Mean tokens and
references shown drop slightly (3,466 → 3,428; 13.0 → 12.8) for the same reason.

This is a real tension in the design and should be carried into any result: more training data
was bought at the price of a training distribution that looks less like the evaluation set. If
the goal is specifically to train the prefix where the graph has most to say, a quota is the
wrong instrument above degree ~34 — the nodes simply do not exist, and only reweighting,
binning, or an eval set matched to train's degree profile can close the gap.

### How to reproduce

```bash
cd src/adapter && python prepare_data_capped.py    # ~exit 0, rewrites the three *_capped.json
```

CPU only, in the `DyTAG` env. The HF warning `Token indices sequence length is longer than the
specified maximum (4491 > 4096)` is expected and harmless — measuring untrimmed prompts in order
to trim them is exactly the script's job.

### Still not done

Unchanged from Stage 1b: nothing downstream reads these sets. `text.py`, `graph.py`, `train.py`
and `generate.py` still build uncapped prompts from the old node lists, and `generate.py`'s
budget mismatch (fixed 256-token reserve vs the freeze's `max(256, len(target)+1)`) is still
outstanding. No training has been run at this quota.

---

## Stage 1d — one capped-prompt path (`text.capped_example`)

### Goal

Preparation for wiring the filter into training. Three programs need to build the *same*
prompt for a node — the freeze (which decides whether a node is usable and how many
references it gets), `train.py`, and `generate.py` — but until now only the freeze had the
logic, as a private helper inside `prepare_data_capped.py`. The other two still build
uncapped prompts from the full reference list.

Wiring them up by copying that helper into each would leave three copies free to drift. The
failure mode is the reason this is a separate stage rather than a detail of the next one:
**a mismatched budget is silent.** It does not raise; it produces a fluent abstract written
from a different set of sources than the node was selected under, and nothing downstream can
tell. One shared function makes the bug impossible instead of something to test for.

### Change

The helper moved from `prepare_data_capped.py` to `text.py` as `capped_example`, next to the
prompt machinery it belongs with (`build_prompt`, `encode`, `Corpus`). The freeze imports it
instead of defining it; its two now-unused imports were dropped. `text.py` gained the imports
the moved code needs. No import cycle — `text` → `neighbour_cap` → `config`.

Behaviour is untouched by construction: the function body is character-for-character what it
was. No logic, constants or budget arithmetic changed.

### Verification

A pure refactor has to prove it is one, so the freeze was re-run and its output compared
against a checksum taken beforehand:

```
data/frozen/eval_nodes_capped.json:      OK
data/frozen/splits_eligible_capped.json: OK
data/frozen/training_sets_capped.json:   OK
```

All three files byte-identical, and every reported statistic (7,750 / 806 / 945 nodes, token
means, references shown, eval overlap) identical to the pre-refactor run.

### Scope

Nothing downstream behaves differently yet. `train.py` and `generate.py` still read the old
node lists and still build uncapped prompts; pointing them at `capped_example` and the
`*_capped.json` sets is the next stage. This one only creates the shared foundation and shows
the move cost nothing.

---

## Stage 1e — wiring training and generation to the capped sets

### Goal

Stage 1d built the shared prompt path; this stage makes `train.py` and `generate.py` use it.
Both previously read the legacy node lists and built prompts from the *whole* reference list,
so the capped sets existed but nothing consumed them.

### The switch

Both scripts take `--sets {capped,legacy}`, defaulting to `capped`:

| | capped (default) | legacy |
|---|---|---|
| `train.py` | `training_sets_capped.json` — 7,750 / 806 | `training_sets.json` — 4,000 / 495 |
| `generate.py` | `eval_nodes_capped.json` — 945 | `eval_nodes.json` — 495 |
| prompts | `text.capped_example` | `Corpus.example`, untrimmed |
| overflow | impossible; a violation aborts the run | 63/495 written as `ERROR:` rows |

The flag deliberately controls the node list *and* the prompt builder together. They are not
independent: the legacy sets were selected on the assumption that a node's whole reference
list fits, the capped sets on the assumption that it is trimmed. Either list paired with the
other builder trains or evaluates on prompts nobody measured.

### Three guards, because every failure here is silent

None of these fail loudly on their own, which is why each is checked explicitly.

1. **Checkpoint provenance.** `train.py` records `sets` in every checkpoint; `--resume` and
   `generate.py` both refuse a checkpoint trained on the other sets. Checkpoints predating
   the flag have no `sets` key and are read as `legacy`. This is not hypothetical:
   `run_train_adapter.sh` resumes from whatever checkpoint is newest, and the `epoch2.pt`
   behind the 2026-08-04 run is still on Komondor. Verified — both a `legacy` checkpoint and
   a pre-flag one are refused against capped eval nodes.

2. **Generation reserve.** `--max-new-tokens` above `GENERATION_RESERVE` is refused in capped
   mode. The sets were frozen holding 256 tokens back; asking for more would need references
   the freeze already dropped. Verified — 512 exits at startup, before the model loads.

3. **Overflow is now an assertion.** In capped mode a prompt over budget means the prompt path
   has drifted from the freeze, so the run aborts rather than writing another
   plausible-looking `ERROR` row into a CSV whose earlier rows are then equally suspect.

### Embeddings

The centroid needs the voyage embeddings at prompt-build time. `train.py` reuses `data.x` from
`load_graph()` via `.numpy()` — a view of the matrix the GNN already holds, so no second load
and no extra 1.6 GB. `generate.py` memory-maps the file instead, because **baseline mode must
build byte-identical prompts** to adapter mode and never loads a graph at all.

### Verification (2026-08-11, CPU, local)

100 sampled nodes across train/valid/eval, checked through both code paths:

```
train   40 nodes: prompts identical across both paths, all fit
valid   20 nodes: prompts identical across both paths, all fit
eval    40 nodes: prompts identical across both paths, all fit
largest prompt seen: 3,824 tokens (budget 3,830)
```

Prompts, targets and kept references are identical whether the embeddings arrive as a view or
a memory-map; every training sequence fits `MAX_LENGTH`; kept references are a subsequence of
`edges.csv` order. The largest prompt lands 6 tokens under budget — tight, as designed, since
trimming stops as soon as a prompt fits.

Legacy mode re-checked end to end: 495 nodes, 63 overflows (12.7%), exactly the figure the
existing runs were produced under.

### Open questions

- **Batch memory is untested at this sequence length.** The ~1 GB activation figure in
  `train.py`'s header was measured at a ~2,300-token mean; the capped train set averages
  ~3,428 with a 4,086 maximum. Batch 4 may not fit — Stage 2 of the plan lowers it to 2 with
  `--grad-accum 8` to hold the effective batch at 16, but that is a precaution, not a
  measurement.
- **Prompt-build cost now sits in the training loop.** Every batch re-runs the trim, which
  tokenises the prompt once per candidate reference count. It was free when it happened once
  at freeze time; whether it is material against a 7B forward pass has not been measured.

---

## Stage 1f — the pre-flight check, early stopping, and the job scripts

### The pre-flight check (`verify_capped_prompts.py`)

Promoted from a scratch script to a pipeline stage, and wired into `run_train_adapter.sh`
ahead of the GPU work — the same placement, and for the same reason, as
`verify_generation.py`. It re-proves that the prompts `train.py` will build are the ones the
frozen sets were selected under, exercising `text.capped_example` itself rather than a
reimplementation.

On a sample per set it asserts that train's embedding view and generate's memory-map yield
identical prompts, targets and reference lists; that every node keeps at least one reference;
that kept references are a *subsequence* of `edges.csv` order (a subset check would pass on
reordered references — `edges.csv` order is not ascending); that prompt + target + prefix fits
`MAX_LENGTH`; and that the prompt alone fits generate.py's budget.

```
train   100 nodes ok
valid   100 nodes ok
eval    100 nodes ok
largest prompt 3,826 tokens, 4 under budget
```

20 s on a laptop CPU, so gating a multi-hour GPU job on it is free. `--sample 0` checks every
frozen node.

### Early stopping

`--epochs` becomes an upper bound of 10 rather than a target of 2, with `--patience 2`.
Validation runs at the end of every epoch — separately from the periodic `--eval-every`
checks, which exist for the curve — and the run stops after 2 consecutive epochs without a
better graph-on validation loss. `--patience 0` disables it.

Two consequences that are easy to get wrong, both handled:

- **The last checkpoint is not the best one.** When early stopping fires, the final epoch is by
  definition worse than the best. `train.py` copies the best epoch to `checkpoints/best.pt`,
  and `run_generate_eval.sh` now prefers it, falling back to the newest `epoch*.pt` for runs
  that predate it. Selecting by highest epoch number would have scored a model the run had
  already decided to abandon.
- **Patience must survive a resubmit.** 10 epochs does not fit one job. Best loss and the
  patience counter are stored in every checkpoint and restored on `--resume`; otherwise the
  counter would reset on each restart and the run could never stop early. A resumed run with
  nothing left to do now says so and exits, instead of entering an empty epoch loop.

Verified against scripted validation curves: a plateau stops at the right epoch, a noisy dip
inside the patience window correctly resets the counter, and `--patience 0` runs to the end.

### Job scripts

| | before | after | why |
|---|---|---|---|
| `--epochs` | 2 | 10 (+ `--patience 2`) | early stopping decides the real length |
| `--batch-size` / `--grad-accum` | 4 / 4 | 2 / 8 | effective batch still 16; sequences went ~2,300 → ~3,428 tokens and the batch-4 memory figure was measured at the old length |
| `--eval-every` | 100 | 250 | at 100, validation cost more tokens than training |
| train walltime | 12 h | 48 h | ~4.3 h/epoch estimated, up to 10 epochs |
| eval walltime | 8 h | 12 h | 945 eval nodes, not 495 |

`run_train_adapter.sh` also documents the prerequisite that bites first: a stale `epoch*.pt`
from the uncapped experiment makes `train.py` refuse to resume, so the job dies at startup
until it is moved to `checkpoints/legacy-uncapped/`.

### Open questions

- **48 h may exceed the partition limit.** It was chosen deliberately to fit 10 epochs in one
  job, but Komondor's maximum walltime has not been checked from here; if the scheduler
  rejects it, drop to 24 h and rely on the resume, which is now safe across restarts.
- **Batch 2 is a precaution, not a measurement.** A short smoke run would settle both it and
  the ~4.3 h/epoch estimate the walltime rests on, and would cost minutes.
- **Validation still runs ~3 times per epoch.** `--eval-every 250` was chosen before
  end-of-epoch validation existed, which adds roughly one more pass per epoch. If validation
  turns out to be a material share of the wall-clock, raising it to 500 costs nothing now that
  every epoch ends with a validation anyway.

---

## Stage 6a — encoder depth as a run parameter (`--gnn-layers`)

**Date:** 2026-08-18

### Goal

Run the 3-layer and 4-layer GraphSAGE encoders against the existing 2-layer result, without
either run depending on which version of the source happened to be on Komondor when the
scheduler started it.

### Why the depth could not just be edited between submissions

The obvious plan — upload a 3-layer `model.py`, `sbatch`, edit it to 4 layers, `sbatch` again —
does not work. `sbatch` copies the *batch script* into the job spool at submit time, so that
file is safe to edit afterwards; the Python under `/app` is not, because it is read when the
job actually starts, after however long it waits in the queue. A job submitted first can
therefore run the architecture intended for the second. Both runs would also write `epoch*.pt`
and `best.pt` into the same `checkpoints/`, and the job script's auto-resume takes the newest
file it finds there.

So the depth travels as an argument. `run_train_adapter.sh` now carries `--gnn-layers` and a
`CKPT_DIR` written literally, edited together per experiment — safe, because editing that file
cannot reach a job already queued.

### Decisions

**One number sets both the encoder depth and the sampling depth.** `NUM_NEIGHBORS = [-1, -1]`
is gone; `config.num_neighbors(gnn_layers)` returns `[-1] * gnn_layers` and `make_loader` takes
the layer count. This is the failure the change exists to prevent: a third `SAGEConv` layer
reading from a 2-hop sample aggregates hop-2 nodes whose own neighbourhoods were never sampled.
It does not raise — it returns a plausible vector and a plausible loss curve, the same class of
silent failure as the `cache_position` bug in Stage 4.

**The depth is recorded in every checkpoint and asserted**, exactly as `sets` already is.
`train.py` refuses a `--resume` whose recorded depth differs, and `generate.py` reads the depth
*out of* the checkpoint rather than taking it on the command line — there is no way to pair a
3-layer checkpoint with a 4-layer encoder. Checkpoints predating this default to 2, so the
existing runs still load.

**Widths stay at 1024 throughout** and dropout stays between every pair of layers, so depth is
the only thing that varies against the 2-layer baseline.

**The 2-layer run keeps `checkpoints/`**; deeper runs get `checkpoints/gnn3/`, `gnn4/`. The
existing results and their W&B history stay where they are.

### The cost of deeper hops, measured

Sampled locally on 500 of the 7,750 capped train seeds, same graph construction as `graph.py`:

| hops | mean nodes | median | p90 | p99 | max |
|---|---|---|---|---|---|
| 2 | 199 | 168 | 380 | 696 | 1,087 |
| 3 | 876 | 708 | 1,770 | 3,590 | 4,629 |
| 4 | 2,645 | 2,186 | 5,597 | 10,451 | 13,836 |

At `--batch-size 2` a micro-batch samples 397 nodes / 734 edges at 2 hops, 1,712 / 4,842 at 3,
and 5,204 / 17,992 at 4. The worst 4-hop batch is ~21 MB of fp32 features against 13.5 GB of
frozen Vicuna, and sampling 500 seeds took 0.1 s on a laptop CPU. **Neither depth is a memory
or throughput problem** — the run stays LLM-bound, as it was at 2 hops.

The 2-hop mean of 199 is far above the 50.6 recorded in Stage 2. That figure was measured on
the legacy out-degree 5–15 nodes; the capped sets go to out-degree 100.

| encoder | params |
|---|---|
| 2 layers | 4,197,376 |
| 3 layers | 6,295,552 |
| 4 layers | 8,393,728 |

The 12,108,096-parameter adapter is unchanged, so trainable totals are 16.3M / 18.4M / 20.5M.

### Settings consolidated into `config.py`

Done at the same time, because the depth sweep made the sprawl obvious: the job script was
restating `--epochs 10 --patience 2 --batch-size 2 --grad-accum 8 --log-every 10
--eval-every 250 --wandb-project dytag-adapter`, and `"lmsys/vicuna-7b-v1.5"` was written out
as a literal in five separate files.

All of it now lives in `config.py`: `EPOCHS`, `PATIENCE`, `BATCH_SIZE`, `GRAD_ACCUM`,
`LOG_EVERY`, `EVAL_EVERY`, `SEED`, `WANDB_PROJECT`, `DEFAULT_SETS`, `MAX_NEW_TOKENS`, and
`BASE_MODEL` (imported by `train.py`, `generate.py`, `text.py`, `prepare_data_capped.py` and
`verify_capped_prompts.py`). Their argparse flags are gone — a constant with a flag shadowing
it is two places to change, which is the problem being fixed.

This reverses the housekeeping decision recorded earlier in this document ("values already
exposed on the command line stay as argparse defaults in the scripts that own them"). That
held while the pipeline was one script; with two sbatch scripts restating the schedule it
stopped holding.

**Five flags survive on `train.py`**, and the test each one passes is *has to differ between
two submissions of the same experiment*:

| flag | why it cannot be a constant |
|---|---|
| `--gnn-layers` | the sweep variable — see below |
| `--sets` | the other arm (capped vs legacy) |
| `--out-dir` | one directory per depth |
| `--resume` | computed by the job script from the filesystem |
| `--wandb` / `--wandb-run` | opt-in, and the name carries `$SLURM_JOB_ID` |

`--gnn-layers` is the one that matters, and it is the reason `GNN_LAYERS` in `config.py` is a
*default* rather than the only value. Editing that constant between two submissions has the
race this whole section exists to avoid: `sbatch` snapshots the job script, but `config.py` is
read when the job starts, so a still-queued 3-layer job would come up as 4 layers. The
checkpoint would record `gnn_layers: 4` and be internally honest — but `CKPT_DIR` and the W&B
run name would both say `gnn3`, and the result would be mislabelled rather than wrong.

Since `vars(args)` no longer carries the hyperparameters, `train.py` names them explicitly in
the W&B config. A run whose config omits its own batch size is not comparable to anything,
which is the entire point of logging to W&B.

### Inputs / outputs

| | |
|---|---|
| In | unchanged — the same frozen capped node sets |
| Out | `checkpoints/` (2 layers), `checkpoints/gnn3/`, `checkpoints/gnn4/`; each checkpoint records its own `gnn_layers` |
| Run | edit `--gnn-layers` and `CKPT_DIR` in `run_train_adapter.sh`, `sbatch`, repeat; then `CKPT_DIR` and `ARM` in `run_generate_eval.sh`. Everything else is `config.py` |

### Verification

`python model.py --gnn-layers {2,3,4}` — every existing assertion passes at all three depths,
including the one that matters most, that randomising the seed rows leaves the output
bit-identical (the target's abstract cannot leak through a deeper encoder either). Parameter
counts came out as tabulated above.

`python graph.py --gnn-layers 3` — the seed-row and edge-direction assertions still hold.

Checkpoint round-trip, tested directly at all three depths: the depth survives `torch.save` →
`load_adapter`, weights come back bit-identical, a checkpoint with no `gnn_layers` key loads as
2 layers, and a deliberate 3-vs-4 mismatch aborts on the new assertion rather than on a
`state_dict` key error.

The job scripts' path logic was exercised under `bash` for each depth: `sort -V` picks the
newest checkpoint from the right directory, and the container-side path is `/app/checkpoints`
or `/app/checkpoints/gnn3` as appropriate.

After the `config.py` consolidation every check above was re-run unchanged, and all ten
modules in `src/adapter/` import cleanly — the cheapest way to catch a constant that was
referenced but never moved.

### Open questions and limitations

- **Nothing has been trained at depth 3 or 4.** Everything above is shape, memory and plumbing;
  whether a deeper encoder actually lowers validation loss or lifts `valid/delta` is the
  experiment.
- **Depth interacts with the seed-collision effect** from the Stage 3a addendum. Collision rates
  there were measured at 2 hops; a 4-hop neighbourhood is ~13x larger, so a seed is far more
  likely to appear inside another seed's neighbourhood and be replaced by the mask. This has not
  been re-measured and could quietly grow the input loss that section describes as sub-1%.
- **Over-smoothing is untested.** Four rounds of mean aggregation over a citation graph is where
  node representations classically start to converge; the encoder output norms in the smoke test
  (0.71–0.73 at both depth 3 and 4, against 0.79–0.81 at depth 2) are the only signal so far and
  come from random weights.
- **The eval job still re-runs the frozen-Vicuna baseline for every arm**, ~3 GPU-hours for a
  file that is byte-identical across depths (greedy decoding, identical prompts, no prefix).
  Left as it is because the two conditions running on one node and one GPU is a deliberate part
  of the comparison, but for the depth sweep specifically it is avoidable waste.

---

## Stage 6b — the prompt rewrite (2026-08-19)

### Goal

Raise `llm_accuracy`, which is the only metric the graph currently moves (baseline 2.81 →
adapter 3.29 over the 945 capped eval nodes) and the one sitting furthest from its ceiling.
The trigger was the qualitative comparison in `experiments/sentiment_analysis/report.md`: on 57
sampled nodes, gpt-5.4-mini judged the adapter against the baseline with the gold abstract as
reference, and the failure modes it names turned out to be specified by the prompt itself.

### Change

`INSTRUCTION` in `text.py`, previously carried over verbatim from
`scripts/generate_test_csv.py`:

> You are a scientific writing assistant. Based on the source publications listed below, write a
> concise abstract for a **new application paper** that **builds on and combines these works**.
> Output ONLY the abstract text, without any heading or explanation.

Three defects, each matching a measured failure:

1. **"a new … paper"** specifies invention, but the target is the existing paper's own abstract.
   The report's summary — "a fluent abstract of a plausible-but-nonexistent paper on the right
   topic" — describes the task as written.
2. **"application paper"** fixes the genre. Surveys fail as a class (98612, 126535, 134751,
   105246) because the prompt does not permit a survey abstract to be written.
3. **"builds on and combines these works"** asks for a synthesis of the sources — the
   neighbourhood centroid. Measured consequence: accuracy rises monotonically with how close a
   paper sits to its own bibliography's centroid (2.56 in the most atypical quintile → 3.82 in
   the most typical, Spearman 0.31 over 945 nodes).

Also **"concise"**, against a gold mean of 174 words and generations averaging 123 (baseline) /
142 (adapter), with 238 of 945 adapter outputs ending mid-sentence.

The replacement frames the task as reconstruction, leaves the genre open, asks for the paper's
departure from its sources rather than their combination, and names a length target. The
surrounding frame moved with it: `Source publications:` → `Cited publications:`, and the
trailing cue `Generated abstract:` → `Abstract:`, which stops the prompt from signalling
"produce something new" at the point of generation.

### Cost and scope

+74 tokens (~250 → ~510 characters) against a mean capped prompt of ~3,428 tokens and a mean
reference of ~200. `fit_refs` re-trims to whatever budget the prompt leaves, so no prompt can
overflow; the effect is that a node already sitting near the budget may show one reference
fewer. `build_prompt` is called only from within `text.py` (`Corpus.example` and
`capped_example`), so training, generation and the freeze all move together, as Stage 1e
requires.

**Runs before and after this change are not prompt-comparable**, and that includes the archived
Vicuna and GPT baselines. The adapter checkpoints were trained under the old wording, so
generating with the new one is a train/test mismatch — both arms need regenerating, and the
adapter needs retraining, before the numbers mean anything.

### Verification

Not yet run. `verify_capped_prompts.py` is the gate: it re-derives every frozen node's prompt
through `capped_example`, asserts at least one reference survives, and prints the largest prompt
and the headroom left. It has to pass before any GPU time is spent.

### Open questions and limitations

- **The length target and the reserve interact.** 180 words is ~245 tokens; the reserve was
  raised from 256 to 320 in the same batch of changes to give it slack, at the cost of a
  slightly smaller prompt budget.
- **"what it concludes" invites fabricated results.** The report records invented figures
  ("66%", "54.2%") under the old prompt. The clause is kept because gold abstracts end with
  claims and because the judge's accuracy rubric explicitly forgives "creative but plausible
  predictions", but it is the clause most worth ablating.
- **The true reference count is still hidden.** Under the cap a paper citing 96 works and one
  citing 6 produce visually identical prompts, which removes the strongest available survey
  signal. Adding it needs a `build_prompt` signature change and is deliberately left out of this
  change so its effect can be attributed separately.
- **Untested.** Every claim here is about what the prompt specifies, not about what it produces.
