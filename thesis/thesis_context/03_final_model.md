# The final model (run 11) — architecture and training

Sources: `docs/run11_model.md` (the primary description), `docs/run9_embedding_decoder.md`
§8–9, `docs/adapter.md` (stages 3b, 4, step 1), `docs/qwen_migration.md`, `src/adapter/config.py`,
`src/adapter/model.py`, `hpc/komondor/jobs/run11_all.sh`, W&B project `dytag-adapter`, Slurm
job **14541449** (Komondor, 2026-10-09/10). Parameter counts for the GNN are computed from the
layer shapes; the adapter and LoRA counts are the ones printed by the code.

This file describes the architecture conceptually. It avoids code; module names are given only
so a reader can find them.

---

## 1. One-paragraph summary

The model writes a paper's abstract **without seeing any text of the paper or of its
references**. Its only input is the paper's position in the citation graph: the Voyage
embeddings of the papers it cites. A one-layer graph attention network (GATv2) with a linear
head turns these into **one 1024-d vector**, an estimate of the paper's own embedding. A
**prefix adapter** turns that vector into **32 key/value tokens at each of the 40 layers** of a
frozen **Qwen3-14B**, which is adapted with **LoRA** (rank 16, query and value projections) and
writes the abstract after a fixed instruction that is identical for every paper. The design
splits the problem into two separately trainable and separately measurable halves: *where in
embedding space is this paper?* (the GNN) and *what abstract does this embedding describe?* (the
reader).

```
  references' Voyage embeddings        ┌──────────────────┐      one 1024-d unit vector
  (target row replaced by [MASK]) ───► │ GATv2 (1 layer)  │ ───► "estimate of the paper's
                                       │ + linear head    │       own embedding"
                                       │ + L2 normalise   │
                                       └──────────────────┘                │
                                                                           ▼
                                                          ┌──────────────────────────────┐
                                                          │ prefix adapter               │
                                                          │ trunk → 80 codes → 2 decoders│
                                                          │ prefix = base + gate · delta │
                                                          └──────────────────────────────┘
                                                                           │ 40 layers × (K,V)
                                                                           │ × 8 KV heads × 32 tokens × 128
                                                                           ▼
  fixed ChatML prompt ("…Write the abstract.") ───────────────► Qwen3-14B (frozen, bf16) + LoRA r16 (q,v)
                                                                           │
                                                                           ▼  greedy, no repeated 6-gram
                                                                       abstract
```

## 2. Components

| component | trainable params | role |
|---|---|---|
| Voyage `voyage-4-large` embeddings | 0 (fixed input) | one unit 1024-d vector per paper (`02_data.md` §3) |
| GNN: 1-layer GATv2 + `[MASK]` + linear head | ≈ 3.15M | references → estimate of the target's embedding |
| Prefix adapter | 32,065,616 | one vector → key/value prefixes for all 40 layers |
| LoRA r16 on q_proj, v_proj (40 layers) | 10,485,760 | lets Qwen read the prefix and write in abstract register |
| Qwen3-14B | 0 (frozen, bf16, 29.6 GB) | writes the text |

Totals: reader (adapter + LoRA) **42.6M** trainable; joint phase (+ GNN) **45.7M**; against
~14.8B frozen LLM parameters.

### 2.1 The GNN estimator

**Input.** The target and its references, each with its Voyage embedding; **all** references
are used (no sampling cap; eval papers have 5–100). Edges point from cited to citing paper, so
the target aggregates only works it cites.

**Masking.** The target's own embedding is the answer, so its row is replaced by one learned
`[MASK]` vector (shared by all targets, initialised to unit norm) before message passing. A
shared vector cannot carry paper-specific information; it acts as a learned "this is the
unknown node" flag. Changing the target's own features provably leaves the output unchanged
(tested).

**One GATv2 layer** (Brody et al. 2022): 4 attention heads × 256 channels, concatenated to
1024. For target *i* with references *j*, GATv2 computes an attention score

  e_ij = aᵀ · LeakyReLU(W_l x_j + W_r x_i),  α_ij = softmax_j(e_ij),
  h_i = Σ_j α_ij · W_l x_j  (per head, then concatenated),

where x_i is the `[MASK]` vector for the target. Each reference therefore gets a learned
weight; a plain mean gives every reference weight 1/|N(i)|. One layer means only direct
references are read, not references of references (the grid showed 2 layers were worse, §4.2).

**Head and normalisation.** A 1024 × 1024 linear layer, then L2 normalisation. This puts the
output in the same space, and at the same norm, as a real Voyage embedding — the space the
reader was trained on (`EmbeddingEstimator`).

**The structure-agnostic baseline it must beat:** the plain reference mean
`normalise(mean_j x_j)` — no parameters, no attention, no head. Cosine to the true embedding
0.816 (validation) / 0.823 (eval).

### 2.2 The prefix adapter (vector → soft prompt)

**What a prefix is.** In prefix tuning (Li & Liang 2021) extra key/value vectors are placed in
the attention cache of every layer, as if some tokens had been processed before the prompt.
Every real token can attend to them, but they are never words and are never fed through the
embedding layer. Here they are **generated per paper** from the input vector.

**Size.** Qwen3-14B uses grouped-query attention: 40 query heads share 8 key/value heads of
dimension 128. One paper's prefix is therefore

  40 layers × 2 (key, value) × 8 KV heads × 32 tokens × 128 = **2,621,440 numbers**.

A direct linear map from 1024 inputs would need ~2.7 billion parameters; the cost is in the
output, so the adapter is factorised:

1. **Trunk:** Linear 1024 → 1024, GELU, LayerNorm → z (1024). Shared by everything after it.
2. **Codes:** one Linear 1024 → 40 × 2 × 192 = 15,360, reshaped into **80 codes of 192
   dimensions**, one per (layer, key/value). Each layer reads the graph through its own 192-d
   view; 80 × 192 = 15,360 code dimensions collectively cover all 1024 input directions.
3. **Two decoders**, one for keys and one for values, each a Linear 192 → 32 × 8 × 128 =
   32,768, **shared across all 40 layers of its role**. Keys and values have separate decoders
   because they do different jobs: keys (which Qwen3 RMS-normalises per head and rotates with
   RoPE) decide *where* attention goes, values carry *what* is read.
4. **Base + gate:**

   prefix[l, r] = base[l, r] + gate[l, r] · delta[l, r]

   - `base` (2,621,440 parameters, init std 0.02 = Qwen's `initializer_range`): a learned
     prefix that is **the same for every paper** — classic prefix tuning, carrying the task
     format.
   - `delta`: the paper-specific part from steps 1–3.
   - `gate`: **80 scalars**, one per (layer, key/value), **initialised to 0**, unbounded (can be
     negative; a zero-init residual scale in the ReZero/LayerScale sense, not an LSTM gate).

| block | parameters |
|---|---|
| trunk (Linear + LayerNorm) | 1,051,648 |
| codes | 15,744,000 |
| two decoders | 12,648,448 |
| base + gate | 2,621,520 |
| **adapter total** | **32,065,616** |

**Why the gate starts at 0, and its cost.**
- *Known start:* at initialisation an untrained `delta` is ~17× larger than `base` element-wise
  (measured on the original design), so an ungated prefix would be noise. With gate = 0
  training starts as plain prefix tuning.
- *Free ablation:* setting gate = 0 at evaluation makes the prefix exactly `base` — "graph
  off" on the same weights, no second model. Logged every validation as `valid/gate0`, and
  `valid/delta = loss(gate 0) − loss(graph)`.
- *Readable:* the per-layer gate values show where in the stack the graph is used.
- *Cost — the bootstrap problem:* d(prefix)/d(delta) = gate, so at gate = 0 the entire
  pathway behind it (trunk, codes, decoders, and the GNN) receives exactly zero gradient. The
  gate itself still gets gradient ⟨∂L/∂prefix, delta⟩, so it must learn its way open from a
  random delta. It did, in run 11 (§4.1). An attempt to start the gate open (calibrated to 20%
  of the prefix norm, runs 7–9) is part of the history and did not help (`04`).
- *Caveat:* gate = 0 is "graph off", not "prefix absent": the 32 `base` positions still absorb
  attention.

**Prefix layout and positions.** The cache tensor per paper is (40, 2, 8, 32, 128). The prefix
occupies positions 0–31 and the prompt starts at position 32, in training and generation alike.
Keys are injected already in RoPE space (position-free). The tooling that makes this exact —
a read-only training cache, an explicit `cache_position`, a patched `generate()` — is described
in `04_development_history.md` §2 and `08_implementation_reproducibility.md`; a wrong offset
produces fluent text and a falling loss, not an error.

### 2.3 The LLM, LoRA and the prompt

- **Qwen3-14B** (Yang et al. 2025): 40 layers, hidden size 5120, 40 query heads / 8 KV heads,
  head dim 128, vocabulary 151,936; frozen, bf16, gradient checkpointing.
- **LoRA** rank 16, alpha 32 (scale 2), dropout 0.05, on **q_proj and v_proj** of all 40 layers
  (80 modules). Because of GQA the two differ in shape: q_proj 5120 → 5120, v_proj 5120 → 1024;
  per layer 16·(5120+5120) + 16·(5120+1024) = 262,144 parameters, ×40 = 10,485,760. LoRA
  weights held in fp32 (AdamW on bf16 parameters silently drops small updates).
  - Why q and v: the prefix supplies its own keys and values, so `k_proj` never touches it;
    the query projection decides how much prefix is attended to, the value projection adapts
    what the real tokens carry.
  - Why LoRA at all: a frozen Qwen could not read the prefix (oracle frozen 0.3% vs oracle +
    LoRA 12.0% TF-IDF top-1, `04` steps 3–4).
  - **LoRA has no per-paper information.** Paper identity can only arrive through the prefix;
    the zero-input arm (0.1% top-1) confirms the content comes from the vector.
- **The prompt is identical for every paper** (`text.GRAPH_ONLY_SYSTEM`, ChatML, Qwen
  "thinking" disabled), 241 tokens. System message (verbatim):

  > You are a scientific writing assistant. The context you have been given encodes the
  > reference list of one paper, whose own abstract is missing. That paper is not among its
  > references: it is a separate, more recent work that cites all of them. Reconstruct its
  > abstract as continuous prose covering, in order: the problem it addresses; what the work
  > it cites has already established about that problem; what it does about what remains open,
  > described by the mechanism it uses the kind of model, data or analysis rather than by a
  > name; and what it evaluates. Where the context leaves the paper's specific contribution
  > undetermined, state what it does determine about its problem and approach, and stop there
  > — a correct general statement is worth more than a specific guess. Write in the voice of
  > the paper's own authors: never mention the cited works as a collection, how many there
  > are, or the authors, and do not restate any single one of them. Write plain prose of
  > 150-180 words, with no markup, mathematical notation or headings. Output ONLY the
  > abstract text.

  User turn: `Write the abstract.` (Verbatim from `src/adapter/text.py`; the missing comma in
  "the mechanism it uses the kind of model" is in the original prompt.)
- **Loss:** next-token cross-entropy on the target abstract tokens only (prompt masked). The
  logits are computed only at supervised positions (the "sliced" loss), which is numerically
  identical to the standard loss (checked to 4.8e-07) and saves ~2.4 GB — 92.9% of positions
  in the old text prompts carried no label.

### 2.4 Decoding

- Greedy (deterministic), at most 320 new tokens, no repetition penalty, batch size 1.
- **`no_repeat_ngram_size = 6`**: no 6-token sequence may appear twice. Chosen automatically
  by the job on 200 eval papers with the oracle input: sentence-loop rate **14.0% → 0%**,
  TF-IDF top-1 52.0% → 51.5% (within the pre-set 2-point tolerance), mean length 123 → 107
  words. Every run-11 arm uses it. (Run 10a, without it, looped in 17.6% of oracle outputs;
  looped outputs scored coherence 3.8 vs 7.7.)

## 3. Inputs that can be fed to the same reader (the ablation ladder)

The reader takes any unit 1024-d vector. The evaluation feeds it:

| input | what it is | role |
|---|---|---|
| **zero** | zero vector | floor: does anything paper-specific come from LoRA alone? |
| **refs-mean** | normalised mean of reference embeddings | structure-agnostic, no learning |
| **GNN** | GATv2 + head estimate (phase 2) | the learned graph estimator |
| **joint** | GNN + reader after phase 3 | the "end-to-end fine-tuned" pipeline |
| **oracle** | target's own embedding | ceiling of a single-vector reader; leaks the answer |
| (text baseline) | ~13 reference abstracts as text, no prefix, frozen Qwen | the text alternative |

## 4. Training: three phases in one Slurm job

One job on Komondor (one NVIDIA A100 40 GB; 8 CPUs; 72 h walltime) ran: CPU verification
scripts → phase 2 (GNN grid) → phase 1 (reader) → loop-fix test → reader evaluations → phase 3
(joint) → joint evaluation → oracle evaluation. ~18 h of training plus evaluation generations.
Every phase resumes from its newest checkpoint on resubmission.

Node sets (`02_data.md`): 90,461 training papers, 806 validation, 945 evaluation (read once,
at the end). An "epoch" is a 15,000-paper slice of one seeded permutation of the training set,
so consecutive epochs see different papers before any repeats.

| phase | trains | data | objective | time |
|---|---|---|---|---|
| 1. reader | prefix adapter + LoRA (Qwen frozen) | 6 × 15,000 papers | LM loss; input = own embedding, or with p = 0.5 the reference mean | 11.5 h |
| 2. GNN grid | GNN + head (no LLM) | 83,259 papers with ≥ 1 reference, 30 epochs | 1 − cos(estimate, own embedding) [+ 0.05 · InfoNCE] | ~31 min for all 8 |
| 3. joint | GNN + head + adapter + LoRA | 3 × 15,000 papers with references | LM loss + 1 · (1 − cos) ; GNN lr 1e-5, reader lr 1e-4 | 5.7 h |

### 4.1 Phase 1 — the reader

- **Learns** to turn a vector into the abstract it describes.
- **Training input mix (`--refs-mix 0.5`):** for each paper, with probability 0.5 the
  reference mean, otherwise the paper's own embedding. The own-embedding half teaches that the
  vector carries the content; the reference half exposes the reader to the imperfect estimates
  it will receive at test time (a paper without sampled references keeps its own embedding).
  The GNN is not used.
- **Settings:** micro-batch 1 × 16 gradient accumulation (effective 16) → 937 optimizer steps
  per epoch, 5,622 total; AdamW lr 1e-4, 100 warm-up steps then constant; weight decay 0
  (decaying `base`/`gate` would pull the prefix toward the ablation); gradient clip 1.0; gate
  initialised to 0; raw (unwhitened) features; cleaned targets.
- **Validation:** LM loss every 250 steps (`valid/graph`, `valid/gate0`, `valid/delta`), and at
  every epoch end a **generation-based** identifiability check: the reader writes 96 tokens for
  128 fixed validation papers from each input, each text is ranked by TF-IDF against those
  papers' gold abstracts → MRR and top-1 (chance 1/128 = 0.8%). Generation, not LM loss, is
  used because LM loss misled earlier runs (`04`).
- **Selection:** the epoch with the best validation MRR on the **reference-mean** input (the
  realistic input). **Stop rules in code:** a "canary" (16 fixed validation papers, 48 greedy
  tokens; stop if fewer than 9 distinct texts) and "oracle top-1 ≥ 10× chance from epoch 2".
  Neither fired.

| epoch end | valid/delta (nats) | mean |gate| | oracle MRR / top-1 | refs-mean MRR / top-1 |
|---|---|---|---|---|
| 1 | 0.024 | 0.019 | 0.162 / 7.8% | 0.126 / 4.7% |
| 2 | 0.052 | 0.045 | 0.350 / 21.1% | 0.328 / 21.1% |
| 3 | 0.071 | 0.066 | 0.503 / 37.5% | 0.428 / 30.5% |
| 4 | 0.078 | 0.080 | 0.604 / 47.7% | 0.479 / 33.6% |
| **5 (selected)** | 0.082 | 0.092 | 0.625 / 48.4% | **0.530** / 41.4% |
| 6 | 0.085 | 0.102 | 0.704 / 59.4% | 0.528 / 39.8% |

- The graph channel opened from the start: `valid/delta` 0 → 0.085 nats; gate 0 → 0.10, still
  rising; graph share of the prefix norm ‖gate·delta‖/‖base‖ 0 → ~12. Training LM loss 3.1 →
  ~2.15 in the first 500 steps, then slowly to ~2.13.
- The oracle input was still improving at the end (MRR 0.62 → 0.70) while the reference-mean
  input had flattened; the learning rate never decayed.

### 4.2 Phase 2 — the GNN grid (no LLM)

Eight variants: {GATv2, GraphSAGE} × {1, 2 layers} × {cosine, cosine + 0.05 · InfoNCE}.
30 epochs, batch 256, lr 1e-3. Epoch selected on validation mean cosine; winner = best
validation cosine (`select_encoder.py` → `winner.pt`). VICReg terms off; output-similarity gate
off.

- **Why cosine:** in run 10a the judge score followed the estimate's cosine to the true
  embedding, ~+0.6 judge points per +0.1 cosine (quintiles 2.33 → 3.21).
- **InfoNCE** (van den Oord et al. 2018): each of 256 anchors must pick its own true embedding
  among 256 in-batch + 4,096 randomly sampled embeddings, cosine / τ (τ = 0.05), softmax
  cross-entropy; false negatives masked. Because 1/τ = 20 scales its gradient, its weight is
  0.05 = τ so both terms pull about equally.

| variant | val cosine | val top-1 (/806) | eval cosine | eval top-1 (/945) |
|---|---|---|---|---|
| plain reference mean (no training) | 0.816 | 63.5% | 0.823 | — |
| **GATv2 1-layer, cosine (winner)** | **0.828** | 70.5% | **0.833** | 70.8% |
| GATv2 1-layer, cosine + NCE | 0.820 | 72.3% | 0.826 | 75.2% |
| SAGE 1-layer, cosine | 0.820 | 67.2% | 0.825 | 68.4% |
| GATv2 2-layer, cosine | 0.822 | 64.4% | 0.822 | 63.2% |
| SAGE 2-layer, cosine | 0.819 | 66.0% | 0.821 | 65.2% |
| SAGE 1-layer, cosine + NCE | 0.818 | 72.0% | 0.822 | 72.8% |
| GATv2 2-layer, cosine + NCE | 0.814 | 71.0% | 0.814 | 69.4% |
| SAGE 2-layer, cosine + NCE | 0.810 | 70.6% | 0.812 | 71.0% |

(top-1 = the target's own embedding is the closest among all validation/eval papers.)

- All eight are within 0.02 cosine of each other and of the plain mean. Two layers are worse
  than one. InfoNCE trades cosine for top-1 (more distinctive estimates: average pairwise
  cosine between different papers' estimates 0.45 vs 0.53; the mean's is 0.65).
- Winner's validation curve: cosine 0.807 (epoch 1) → 0.818 (3) → 0.825 (15) → 0.828 (29);
  selected `best/epoch` 28. Training cosine (0.79) is below validation cosine (0.83);
  uninvestigated.
- **The InfoNCE GNN was never put through the LLM**; whether its better retrieval gives better
  abstracts is untested.

### 4.3 Phase 3 — the joint fine-tune

- Starts from the phase-1 reader and the phase-2 winner; the reader is now fed the **GNN
  estimate**.
- **Loss:** LM loss + 1 · (1 − cos(GNN estimate, own embedding)). The cosine term keeps the
  estimate in the space the reader understands so the LM loss cannot drag it elsewhere (the
  encoder drifted that way in runs 7–8).
- **Learning rates:** GNN 1e-5, reader 1e-4 (GNN 10× slower because it arrives trained).
  45.7M trainable, 2,811 optimizer steps. Selection: best validation MRR on the GNN input.

| epoch end | GNN-input MRR / top-1 | refs-mean MRR / top-1 | valid/delta |
|---|---|---|---|
| 1 | 0.592 / 44.5% | 0.525 / 39.8% | 0.062 |
| 2 | 0.616 / 49.2% | 0.475 / 33.6% | 0.060 |
| **3 (selected)** | **0.643** / 53.1% | 0.484 / 34.4% | 0.061 |

- **The GNN did not move:** `train/cos_loss` stayed at 0.20 and the pairwise similarity of GNN
  outputs at 0.54 for all three epochs. At lr 1e-5 the GNN was effectively frozen, so this
  phase adapted the reader to the GNN output. **It is not true end-to-end training.**
- The reader specialised: GNN-input MRR 0.59 → 0.64 while refs-mean MRR fell 0.53 → 0.48
  (128 papers; SE ≈ 0.04).
- `valid/delta` here is measured on the GNN input (phase 1: oracle input) — not comparable.

## 5. Monitoring (W&B project `dytag-adapter`)

Runs `run11-reader-14541449`, `run11-joint-14541449`, `run11-gnn-<conv>-l<layers>-<cos|cosnce>-14541449`.

| metric | meaning |
|---|---|
| `train/loss`, `train/lr`, `train/grad_norm`, `train/tokens_per_sec` | basics |
| `valid/graph`, `valid/gate0`, `valid/delta` | LM loss with the graph, with gate forced to 0, and their difference (how much the vector helps predict the abstract) |
| `gate/mean_abs`, `gate/max_abs`, `gate/per_layer` | how far the graph channel is open, per layer |
| `prefix/graph_fraction` | ‖gate·delta‖/‖base‖ |
| `prefix/graph_contrast`, `encoder/output_cosine` | mean pairwise cosine of different papers' deltas / GNN outputs (collapse check; added in run 8) |
| `canary/distinct` | distinct greedy outputs for 16 fixed papers (collapse itself) |
| `val_id/oracle_*`, `val_id/refs_*`, GNN-input `val_id` | generation-based TF-IDF MRR/top-1 on 128 validation papers |
| `pretrain/mean_cosine`, `pretrain/top1/top5`, `pretrain/output_cosine`, `pretrain/loss` (InfoNCE), `pretrain/batch_accuracy`, `pretrain/refs_mean_*`, `best/valid_*`, `best/eval_*` | GNN grid metrics; `pretrain/variance`/`covariance` (VICReg) had weight 0 and are not meaningful in run 11 |

## 6. Design decisions and their reasons (for the method chapter)

| decision | reason |
|---|---|
| split into reader + estimator | each half trainable and measurable alone; end-to-end training from scratch collapsed in every attempt (`04`) |
| train the reader on 90k papers seen ~once, not 7,750 seen 10× | variety of vector → text pairs; the 7,750 cap came from the text arm's context budget, which does not apply to a vector |
| refs-mix 0.5 instead of synthetic noise | trains on the real estimation error; synthetic isotropic noise (run 9) was tied to whitened space and run 9 failed |
| raw features, gate starts at 0 | the only configurations that opened the channel (step 4, run 10a) |
| GNN trained to imitate the target's embedding | the reader already reads embeddings; the estimate's cosine predicts the judge score |
| keep the GNN head | its output is normalised and in Voyage space, as the reader expects (run 8 failed by discarding the head) |
| select on generated text, not LM loss | LM loss is dominated by generic scientific English and preferred templates in earlier runs |
| stop rules in code | run 8 ran 1,250 steps past a stop rule written only in prose |
| one prefix vector, no per-reference tokens | keeps the joint-stage gain attributable; per-reference tokens failed in step 6 |
| greedy + 6-gram ban | deterministic reruns; removes sentence loops at ≤ 0.5 points of identifiability |
| fixed prompt for all papers | everything paper-specific must come through the prefix |

## 7. Compute

- Hardware: one NVIDIA A100 40 GB on Komondor (Hungarian national supercomputer),
  Singularity container `adapter-qwen-gpu.sif`, project account `c_gnn42`.
- Reader 11.5 h; GNN grid ~31 min; joint 5.7 h; each 945-paper evaluation arm ~7 GPU-hours
  (estimate from the job plan; the loop test ~3 h). Reader throughput in run 9 was measured at
  1,386 tokens/s.
- Peak memory: Qwen3-14B bf16 weights 29.6 GB of the 40 GB.
