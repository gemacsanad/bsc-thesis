# Step 6 failure analysis — why per-neighbour tokens made it worse

Job 14447555 (train) / 14450409 (eval). Analysed 2026-09-28 over
`experiments/2026-09-25-14450409_ntok64-gatv2-lora64/`.

Step 6 changed three things at once against step 5: GATv2 replaced GraphSAGE, each 1-hop
neighbour got its own input-embedding token (k ≤ 64), and LoRA went r=16 → r=64. The result is
the **worst graph run of the project on every conditioning metric**, while simultaneously
producing the **lowest training loss ever recorded**, including the oracle and the text runs.

That combination is the finding. This document establishes what broke, proves it is not a bug,
locates where the information is lost, and proposes a fix.

---

## 1. Results

| run | distinct | singletons | largest cluster | rows 3+ | judge acc | coherence | top-1 ID |
|---|---|---|---|---|---|---|---|
| text baseline (era 1) | 945 | 945 | 1 | 752 | **3.592** | 7.90 | **43.49%** |
| step 4 oracle + LoRA | 944 | 943 | 2 | 456 | 2.712 | 7.20 | 16.72% |
| step 2 graph, frozen LLM | 809 | 742 | 11 | 137 | 1.737 | 7.42 | 2.86% |
| step 5 graph + LoRA r=16 | 453 | 324 | 61 | 48 | 1.490 | 7.33 | 1.16% |
| **step 6 ntok64 + GATv2 + r=64** | **342** | **230** | **165** | **20** | **1.314** | **7.94** | **0.32%** |

Chance top-1 = 0.106% (n=945). "top-1 ID" is TF-IDF retrieval of the node's own gold abstract
out of all 945 using the generated text as the query — reimplemented for this analysis
(`scratchpad/probe.py`), so all five rows are computed identically.

Other metrics are flat and uninformative: ROUGE-1 0.2866, ROUGE-L 0.1605, BERTScore-F1 0.8325 —
all within noise of step 5 and step 2. **The surface-overlap metrics cannot see this failure at
all.** Only the judge and the identifiability probe can.

Paired against step 5 on the same 945 nodes: step 6 is **worse on 235, better on 96**, unchanged
on 614. 71.0% of rows score the minimum accuracy of 1, up from 57.8%.

### The shape of the failure

Coherence (7.94) and language (8.01) are the **highest of any graph run** and statistically
level with the text baseline (7.90 / 8.06). The model writes better English than it ever has,
about the wrong paper.

Topic agreement — coarse keyword bucketing of gold vs generated, 8 buckets:

| run | topic agreement |
|---|---|
| text baseline | 0.625 |
| oracle + LoRA | 0.597 |
| step 5 | 0.568 |
| **step 6** | **0.324** |

Always guessing the majority bucket (vision, 391/945) scores 0.414. **Step 6 is below the
majority-class baseline.** Its prior is actively mis-shaped: it emits 528 vision abstracts for
391 vision papers, 185 RL abstracts for 57 RL papers (3.2× over-generated), and 0 speech
abstracts for 8 speech papers.

Step 5 still transmitted topic at 0.568. Step 6 destroyed that.

### What collapse looks like at the row level

One identical string is emitted for **165 nodes (17.5%)**:

> We propose a novel approach to the problem of learning a deep neural network for the task of
> image classification. […] We show that this approach can be used to train a deep neural
> network with a large number of layers, and that it can be used to train a deep neural network
> with a large number of parameters. We also show that this approach can be used to train a deep
> neural network with a large number of training images.

Note the text is itself degenerate — three near-identical clauses. 243 of 945 generations have
>10% 8-gram self-repetition.

Those 165 papers are about selective classification, optimal transport, GANs, weakly supervised
instance segmentation, backpropagation speedups, autonomous driving, and VoxCeleb speaker
recognition. Their degrees span k = 5 to 100.

The three best rows (accuracy 4, out of 945) are topic hits and nothing more — a video-QA paper
gets a generic VQA abstract, a UniViLM pre-training paper gets a generic video-captioning
abstract. **No row in the run reproduces a specific contribution.**

---

## 2. It is not a bug

Checked directly, all clean:

- `train.step_loss` sequence arithmetic is consistent: `inputs_embeds` is k+L, `cache_position`
  is `arange(P, P+k+L)`, attention is P+k+L, labels are padded by k. Shapes agree.
- `generate.generate_with_prefix` calls `prefill_neighbours` on the real path
  (`prefix_generation.py:170`), and `patch_for_prefix` is given `max_extra=ntoken.max_tokens`.
  Training and generation see the same sequence — `verify_neighbour_tokens.py` holds that to
  9.5e-07.
- Early stopping is correct. `best.pt` is the lowest-validation epoch (≈2.181, around step 960),
  **not** the overfit tail that ran to step 1930. The evaluated checkpoint is the right one.
- The eval scored the correct checkpoint via the `oracle` flag guard.

**Not memorisation either.** Generated text vs the nearest of the 7,750 training abstracts
(TF-IDF cosine): mean 0.272, max 0.495, **zero** rows above 0.6. Real gold abstracts score 0.236
against the same pool. The model did not memorise the training set — it learned a generic
register. Generated text vs its own gold averages 0.056.

So the low train loss is real and honestly earned, by writing plausible scientific English that
is uncorrelated with the target.

---

## 3. Where the information is lost

This is the measurement that matters, and it overturns the step-5 conclusion.

**In Voyage space, with no model at all, a plain mean of a node's reference embeddings retrieves
that node's own embedding out of 945 at 64.34% top-1** (median rank 1, MRR 0.757).

| pooling of the 1-hop references | top-1 | median rank | MRR |
|---|---|---|---|
| mean of all references | **64.34%** | 1 | 0.757 |
| max-pool | 47.41% | 2 | 0.601 |
| single best-matching reference | 34.50% | 3 | 0.501 |

`scratchpad/refsignal.py`, memory-mapped, CPU-only. Mean references per eval node: 51.8.

`docs/soft_prompt_only.md` currently states, from step 5:

> **The channel works. The single pooled 1024-d vector is what does not carry the paper.**

**That is false, and it is the claim that misdirected step 6.** A single mean-pooled 1024-d
vector carries the paper extremely well. The step-5 inference compared oracle (11.96%) against
GraphSAGE-pooled (0.95%) *as measured through a trained model*, which cannot distinguish "the
vector does not carry it" from "training destroyed the ability to read it". The direct
measurement says it is the second.

So step 6 added information capacity to a pipeline that was never short of information. That is
why it did not help — and the extra LoRA capacity it shipped alongside actively hurt.

### The two gaps, sized

- Signal available in the pooled references: **64.3%** (embedding-space retrieval)
- Best the KV-prefix channel has ever delivered, given a *perfect* target vector: **16.72%**
  (step 4 oracle, text-space retrieval)
- What step 6 delivers: **0.32%**
- What putting the same references in the prompt as text delivers: **43.49%**

Two separate problems. The 16.7 → 0.32 gap is a **training-dynamics failure** and is the whole
of step 6's regression. The 43.5 → 16.7 gap is a **channel-capacity ceiling** — even a perfect
vector through 32 KV tokens is worth about a third of the reference text. The first is fixable
and is where all the value is right now.

---

## 4. Why it collapsed

### 4.1 Adapter capacity is anti-correlated with graph use

Same encoder family, same prefix, same prompt, three levels of LLM adaptation:

| LLM adaptation | top-1 ID | distinct |
|---|---|---|
| frozen | 2.86% | 809 |
| LoRA r=16 | 1.16% | 453 |
| LoRA r=64 | 0.32% | 342 |

Monotonic, and steep. **The best graph identifiability the project has ever achieved came with
no LoRA at all.** Every increment of decoder capacity has been spent learning not to need the
graph. This is now measured three times and should be treated as a law of this setup rather than
as three coincidences.

### 4.2 Both gates start closed, and LoRA starts open

At step 0: `adapter.gate = 0` (the pooled prefix is node-independent) and `ntoken.gate = 0` (all
64 neighbour tokens are literally the same vector). The model's first view of the task is the
graph-only prompt, a generic prefix, and a block of 64 identical unexplained vectors — **zero
graph information, maximum distraction** — while LoRA r=64 is already training.

The fastest available loss descent from there is: learn the generic register, and learn to
suppress the 64 junk tokens. Adding `k_proj` to `LORA_TARGETS` in step 6 handed the model the
exact mechanism to do the suppressing, since a neighbour token's key is computed by `k_proj`.

By the time the gates could usefully open, the template is already a strong local optimum and
the gradient pushing them open is small. **Staged training was in the step-6 plan precisely as
the fix for this, and was never implemented.**

### 4.3 The gates barely opened, and the channels cancelled

Final values, read from the W&B curves (approximate, from the plots):

| metric | final |
|---|---|
| `gate/neighbour` | 0.0083 |
| `gate/mean_abs` (prefix) | 0.0076 |
| `prefix/graph_fraction` | ~0.70 |
| `valid/graph` | ~2.228 |
| `valid/gate0` (both channels off) | ~2.229 |
| `valid/ngate0` (neighbours off, prefix on) | ~2.252 |

`valid/ndelta` and `valid/delta` are **loss differences**, not norm ratios (`train.py:264`). So:

- Turning off **both** channels costs **~0.001 nats**. The entire graph is worth nothing.
- Turning off **only the neighbours** costs **~0.023 nats** — i.e. the pooled prefix *without*
  the neighbour tokens is **worse than no graph at all**.

The second is the more damning number. The two channels co-adapted into mutual noise
cancellation: each is only tolerable in the presence of the other, and together they are worth
zero. `prefix/graph_fraction` reaching 0.70 is not evidence the graph is being used — it means a
large node-dependent perturbation is being injected and the LLM has learned to be invariant
to it.

### 4.4 The objective does not reward identity

Cross-entropy over ~230 abstract tokens is dominated by generic scientific English. Getting the
paper right buys a few nats across a handful of content words; writing fluent boilerplate buys
more on every token. Train loss 1.83 — the lowest of any run including the oracle — alongside
topic agreement 0.324 — the worst of any run — is exactly that trade being made.

Nothing in the loss ever asks the prefix to be about *this* paper. The gate architecture then
makes "ignore it" the default and the starting point.

### 4.5 Secondary: the reference selection policy is backwards

`select_neighbour_rows` keeps the 64 references **closest** to the centroid — that is, the most
typical and most redundant ones — and drops the most distinctive. Measured on the 345 eval nodes
where the cap actually bites (`scratchpad/selpolicy.py`):

| selection of 64 | top-1 |
|---|---|
| all references (no cap) | 70.43% |
| **64 nearest centroid (current)** | **66.38%** |
| 64 random | 69.57% |
| 64 farthest from centroid | **71.88%** |

The current policy is the worst of the three, and farthest-first beats even using every
reference. Worth ~5.5 points of available signal for a one-line change. It is a secondary
effect — it does not explain a collapse from 64% to 0.32% — but it is free.

### 4.6 Secondary: ablation arms never ran

`RUN_GATE0=1` and `RUN_NGATE0=1` were set, but only `.../2026-09-25-14450409_ntok64-gatv2-lora64/`
exists — no `-gate0` or `-ngate0` directories. The generation-side ablations are missing, so the
attribution above rests entirely on the validation-loss arms.

Separately, `run_generate_eval.sh:167` predicts that this run's gate=0 arm will emit "one
identical abstract x945". That is false for step 6: at gate=0 both gates are zero, so every
neighbour token is the same vector — but **k still varies 5–64 per node**, so the sequence
length varies and the output does not collapse to one string. Expect up to ~60 distinct
outputs grouped by degree, which also means this arm now leaks degree into the floor.

The same wording in `soft_prompt_only.md` (lines 603 and 669) is *not* affected — those are
statements about step 5, which had no neighbour tokens, and for step 5 they are accurate.

---

## 5. Plan

Ordered by expected value. The first item is cheap and settles the diagnosis.

### P1 — Pre-train the encoder + adapter on a retrieval objective, with no LLM in the loop

The decisive change. We know the neighbourhood determines the target at 64% top-1. Train the
encoder and the prefix adapter to *preserve* that, before the LLM ever gets a vote:

InfoNCE over in-batch negatives, pulling the encoder's seed output toward the target node's own
Voyage embedding. No LLM, no LoRA, batch size can be large — **minutes on one GPU, not hours.**

Success criterion, measurable without the LLM: the trained encoder's seed output retrieves the
target embedding at **≥50% top-1** on the 945 eval nodes. If it does, the encoder is provably
carrying identity and the problem is isolated to LM training. If it cannot reach 50%, the GNN
architecture itself is destroying the signal a plain mean preserves, which is a different and
equally important finding.

This is not the ridge probe that was dropped, and the earlier objection to it does not apply.
That was a linear probe used as evidence about a ceiling, where only a high score was
conclusive. This is an **initialisation**, and the 64.34% measurement has already answered the
question the probe would have asked. Voyage is being used here as a *training target for
warm-start*, not as a claim about the best representation — the LM stage is still free to move
away from it.

### P2 — Stage the training, and keep the retrieval loss on

1. Encoder + adapters only, LoRA disabled and gates open, for the first epochs.
2. Then enable LoRA, keeping a weighted auxiliary InfoNCE term on the prefix for the whole run.

The auxiliary term is what stops the drift back to generic once LoRA is live. Without it, P1's
warm start will be undone by exactly the dynamics in §4.2.

### P3 — Open the gates at initialisation

`gate = 0` makes ignoring the graph both the default and the starting point, which is the
pathology in §4.2. Initialise the gates to a small non-zero value (or drop the gate and rely on
the LayerNorm). The free gate=0 ablation is preserved by zeroing at **evaluation** time — it
does not require training from zero, and it is not worth the training pathology it causes.

### P4 — Drop back to LoRA r=16, and run a no-LoRA arm

§4.1 says capacity is the enemy here. r=64 with `k_proj` should not return until the graph
channel demonstrably carries something. The frozen-LLM arm (step 2, 2.86% top-1) is still the
best graph result on record and deserves a rerun on the step-6 encoder as a control.

### P5 — Fix the selection policy

`select_neighbour_rows`: farthest-from-centroid, or random. One line, ~5.5 points of signal on
the 345 affected nodes (§4.5).

### P6 — Correct the record

One falsified claim, and one stale prediction. They are different kinds of thing:

- **`soft_prompt_only.md:681`** — "The channel works. The single pooled 1024-d vector is what
  does not carry the paper." This is **falsified** by §3, and it is the claim that directed the
  whole of step 6. It needs correcting in place, because it is stated as a general conclusion
  rather than as a step-5 observation.
- **`run_generate_eval.sh:167`** — predicts gate=0 will emit one identical abstract ×945. False
  for step 6 (§4.6); the arm did not run, so nothing was mis-reported, but the comment will
  mislead the next person to read it. The walltime note at line 68 that advises dropping
  `RUN_GATE0` first "because it produces one distinct string" rests on the same error.

The identical wording about gate=0 at `soft_prompt_only.md:603` and `:669` is **correct and
should be left alone** — those describe step 5, which had no neighbour tokens, and for step 5
one identical abstract ×945 is what actually happened.

### Ablation discipline

Step 6 changed three things at once and the result is uninterpretable as a consequence — the
regression cannot be attributed to GATv2, to the neighbour tokens, or to r=64. Given that all
three landed together and the outcome was the project's worst run, the next run should change
**one** thing: P1 + P2 with everything else reverted to step 5's configuration.

---

## 6. Open questions

- **Is GATv2 or GraphSAGE responsible for any of the topic-signal loss (0.568 → 0.324)?**
  Unattributable from this run. P1 answers it directly and cheaply, since the retrieval score
  can be computed for both convolutions with no LLM.
- **Does the 16.7% oracle ceiling move with more KV tokens?** The step-4 oracle is the honest
  upper bound for the channel as built, and it is ~⅓ of what reference text achieves. If P1
  succeeds, this ceiling becomes the binding constraint and is the next thing to attack.
- **Is the degree confound growing?** corr(degree, accuracy) went 0.073 (step 5) → 0.120
  (step 6). Small, but it is the known confound and it moved the wrong way.
- **No era-4 945-node text baseline** still, so the absolute comparison remains against a prompt
  known to be ~1.46 judge points weak on the 48-node stratified sample.

---

## Reproducing the analysis

Four CPU-only scripts in `src/analysis/`, each runnable as `python3 src/analysis/<name>.py`
from the repo root with the stock `python3` (sklearn 1.6.1, numpy 2.1.3) — no GPU, no conda env:

| script | what it measures |
|---|---|
| `probe.py` | TF-IDF top-1 identifiability of generated text against the 945 golds, for all runs |
| `refsignal.py` | reference→target retrieval in Voyage space, memory-mapped (§3) |
| `selpolicy.py` | centroid selection policy comparison (§4.5) |
| `memo.py` | generated text vs the 7,750 training abstracts — the memorisation test (§2) |

`refsignal.py` reads ~105 MB of the 1.5 GB embedding file via `mmap_mode='r'`. Note that
`edges.csv` is stored `(citing, cited)` — column 0 cites column 1 — which is the opposite of the
`edge_index` convention in `graph.py:57-62`.
