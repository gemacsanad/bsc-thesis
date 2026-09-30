# Step 8 — representation, preprocessing and feature engineering

Written 2026-09-29, after step 7 (`experiments/2026-09-29-1446*_rank-gatv2-lora16`) returned the
most collapsed generations in the project. Companion to
[`step6_failure_analysis.md`](step6_failure_analysis.md) and [`infonce_plan.md`](infonce_plan.md).

This file exists because the step-7 diagnosis turned out to be about the **input representation**,
not about the encoder, the loss or the adapter — and because the pipeline has never had a
feature-engineering or text-cleaning pass. Everything below is measured on the frozen splits with
CPU-only scripts.

---

## 1. Where step 7 left us

| run | distinct | largest | open8 | top-1 ID |
|---|---|---|---|---|
| text baseline | 945 | 1 | 490 | 43.49% |
| oracle kv192+lora | 944 | 2 | 531 | 16.72% |
| step 2 frozen | 809 | 11 | 137 | 2.86% |
| step 6 ntok64 lora64 | 341 | 165 | 34 | 0.32% |
| **step 7 best (ep4)** | **34** | **533** | **2** | **0.11%** |
| step 7 last (ep6) | 195 | 199 | 6 | 0.32% |

Chance top-1 is 0.106%. The ep4 checkpoint is **at chance**, and 890 of 945 abstracts open with
the same eight words.

The `gate=0` control settles what the graph contributed: it emits **one string for all 945 nodes**,
scores **the same 0.11% top-1 as the adapter arm**, and the adapter's text is word-identical to it
for a **median of 93 words**. With `prefix/graph_fraction` running at 1.8–7.0 — the graph supplying
several times the base prefix's norm — the channel was loud and carried nothing.

---

## 2. The named failure mode

The pooled reference vector the adapter consumes:

```
mean pairwise cosine between different nodes : 0.6602
cosine to the global mean direction          : 0.8127
node-specific energy                         : 0.3321  of 1.0
```

Two thirds of the signal is one shared direction. `PrefixAdapter.trunk` is
`Linear → GELU → LayerNorm`, and LayerNorm normalises **per sample across features** — it cannot
remove a component shared *across nodes*. A near-constant input therefore produces a near-constant
`delta`, and greedy decoding emits one string.

This is **anisotropy** (Mu & Viswanath, ICLR 2018; Ethayarajh, 2019) compounded by **dimensional
collapse** (Jing et al., ICLR 2022) — both established, both with published remedies. It is not a
quirk of this project, and naming it is what makes the fix defensible rather than ad hoc.

**Why nothing caught it.** `info_nce` does `F.normalize(z)` then divides by `τ = 0.05`, a ×20
amplification of cosine differences. That objective is *invariant* to anisotropy: it scored 67%
top-1 on a representation the adapter reads as a constant. `prefix/graph_fraction` measures the
graph channel's **amplitude**, not its **node-dependence**, so it read 7.0 while every node got the
same prefix. No logged number was asking whether the prefix varied.

---

## 3. Feature engineering

### 3.1 The features are already anisotropic before the GNN touches them

Measured over the 25,539 nodes reachable from the eval set (`node_embeddings.npy`, 384,756 × 1024):

```
L2-normalised already                : yes (norm 1.0000, sd 0.0000)
participation ratio (effective rank) : 92.6 / 1024
entropy effective rank               : 226.2 / 1024
top-10 principal components explain  : 26.0%
top-100                              : 69.9%
||mean of all unit features||        : 0.5975
```

For isotropic unit vectors the mean of 25k of them would have norm ≈ 0.006. It is **0.5975**. The
1024-dimensional voyage space is effectively ~93–226 dimensional, and the collapse is inherited
from the embedding model, not created by our encoder. Mean-pooling then concentrates it further —
node-specific energy falls 0.470 → 0.283 from the lowest to the highest degree band, and generation
diversity falls 61% → 24% across the same bands.

### 3.2 Whitening — measured

Retrieval of the right paper out of 945, from the pooled reference vector (ZCA with a 1e-3
eigenvalue floor, so these are regularised numbers, not the optimistic ones):

| variant | top-1 |
|---|---|
| A. `pool(raw)` — **the pipeline today** | **64.34%** |
| B. `ZCA(pool(raw))` — whiten the pooled vector | 84.34% |
| **C. `pool(ZCA(features))` — whiten the inputs, then pool** | **78.52%** |
| D. both | 83.81% |

It is ordinary feature preprocessing — no new model component, applied identically to the GNN, the
neighbour tokens and the oracle, so every arm is treated the same and the comparisons stay honest.
That placement answers the objection to bolting a fixed 1024×1024 matrix onto the adapter: the
anisotropy belongs to the features, so it should be corrected where the features are made.

**Correction (2026-09-30), from the honest fit.** The table above was fitted on nodes *including*
the eval set. `whiten_features.py` fits on the 383,005 nodes that are not eval or validation seeds,
and the retrieval gain is smaller than the leaky fit promised:

| | raw | whitened (train-only fit) |
|---|---|---|
| eval top-1 out of 945 | 64.34% | **72.59%** (+8.3, not +14.2) |
| pooled pairwise cosine | 0.6602 | **0.0547** |
| participation ratio | 122 / 1024 | **948 / 1024** |
| ‖mean direction‖ | 0.5458 | **0.0078** (isotropic ≈ 0.006) |

**Retrieval was never the diagnosis, and it should not be the gate.** Step 7 had 64.34% of the
information retrievable and delivered 0.11% — what it lacked was a prefix that *varied between
nodes*. That is the row that moved by a factor of twelve. The **oracle**, which generates 944
distinct abstracts at 16.72% top-1, is fed vectors at pairwise cosine **0.378**; whitening takes the
graph input to **0.055**, seven times past it. The gate in `config.py` is therefore pooled cosine
< 0.10 *and* no loss of retrieval, both measured in the same run so neither can go stale.

`eps` barely matters: swept from 1e-4 to 1e-1 over 945 held-out **training** nodes — never the eval
set — dev retrieval moves only between 69.3% and 70.6%, which is inside the noise at n=945. 1e-3 is
kept as the standard choice rather than chasing a 1.3-point difference.

### 3.3 Fixing it in the objective instead of by matrix

The principled version of the same correction is to make Stage A produce a decorrelated
representation *because its loss demands it*. **VICReg** (Bardes, Ponce & LeCun, 2021) adds exactly
the two terms missing from `info_nce`:

- a **variance** term — a hinge on each dimension's standard deviation across the batch, which is
  what prevents collapse outright;
- a **covariance** term — off-diagonal covariance driven to zero, decorrelating the dimensions.

Stage A runs at batch 256, so both are estimable at no meaningful cost. This is preferable to a
post-hoc transform because it removes the drift problem (a fixed transform goes stale as the
encoder trains) and because it converts a patch into a finding.

### 3.4 Unused metadata

`nodes.csv` carries `year` and `core`; neither is a model input today. `year` is a strong, clean
candidate: a 2012 paper and a 2019 paper differ enormously in topic, the seed's own year is
metadata rather than abstract content, and the seed's *features* are already masked so nothing
leaks. `edges.csv` has no timestamps, so richer temporal modelling is not available without
rebuilding the graph.

Out-degree is **not** recommended as a feature: `s2-coverage-confound` records that ~44% of core
nodes lack a real bibliography, so degree is partly a crawl artifact and already confounds judge
accuracy (oracle 2.00 → 2.88 across degree bands with no graph input at all).

---

## 4. Text preprocessing

### 4.1 Audit of the 7,750 training abstracts

| artefact | docs | share |
|---|---|---|
| inline math `$...$` | 1,047 | 13.5% |
| LaTeX commands `\textit{`, `\emph{` | 542 | 7.0% |
| super/subscripts `^2`, `_1` | 481 | 6.2% |
| whitespace / newline runs | 166 | 2.1% |
| citation brackets `[12]` | 101 | 1.3% |
| escaped chars `\\`, `\%`, `\&` | 90 | 1.2% |
| URLs | 24 | 0.3% |
| HTML tags / entities / non-Latin | 0 | 0.0% |

One empty abstract across 384,756 nodes; five exact duplicate texts. The corpus is clean apart from
**LaTeX and maths, which touch roughly a fifth of documents.**

Markup correlates with worse scores on the eval set (93 of 945 golds, 9.8%):

```
bertscore_f1   with markup 0.8218   without 0.8328   delta -0.0110
llm_accuracy   with markup 1.3011   without 1.3850   delta -0.0839
```

Worth doing, with one caveat that has to be declared: **the voyage embeddings were computed on the
raw text.** Cleaning targets without re-embedding introduces a mismatch between the feature and the
target. Re-embedding 384k documents through a commercial API is not on the table, so the mismatch
is accepted and stated. It is mild — the features are an input representation, not a label.

### 4.2 The gold abstracts are *not* templated, and ours are

| | distinct 8-word openings | share |
|---|---|---|
| **gold training abstracts** | **7,645 / 7,750** | **98.6%** |
| text baseline | 490 / 945 | 51.9% |
| oracle | 531 / 945 | 56.2% |
| step 7 last | 6 / 945 | 0.6% |
| step 7 best | 2 / 945 | 0.2% |

28.2% of golds open with a stock phrase ("in this paper", "we propose"), and the top-5 three-word
openings cover 12.1% — but by word 8 they are **98.6% unique**. The collapse is not justified by the
data.

This **corrects an earlier claim of mine** that the ranking window targets formulaic text. It does
not: 32 tokens is ~24 words, and the golds are almost entirely distinct by word 8. The window is
sound. Sampling a random offset instead remains worth doing, but as *objective alignment* — the
opening states the problem, the middle states the method, and methodology recovery is the stated
goal — not as a bug fix.

### 4.3 The length deficit

```
gold      median 170 words   mean 174
generated median  99 words   mean 105     ratio 0.60
generations shorter than their gold: 90.3%
```

The model writes **60% of the expected length**, on 90% of rows. This is not truncation —
`MAX_NEW_TOKENS = 320` is ~240 words and the p95 gold is 270 words, so only ~9% of golds exceed what
the model is allowed to produce. It stops early on its own.

It costs recall on every row, and it is most of the apparent quality drop between the two step-7
checkpoints: BertScore **precision** barely moved (−0.0025) while **recall** fell, and restricting to
the 164 nodes of near-equal length shrinks the ROUGE-1 gap from −0.0172 to −0.0057 and erases the
judge-accuracy gap entirely (−0.0444 → **+0.0061**).

A perfect 105-word abstract caps at ROUGE-1 **0.7685** against the full gold, so length is not the
binding constraint today (we are at 0.2737).

**Correction (2026-09-30).** I wrote above that the prompt does not state a target length. It does —
`GRAPH_ONLY_SYSTEM` says *"Write plain prose of 150-180 words"*. And length does not track
information:

| run | median words | in 150-180 |
|---|---|---|
| text baseline, full reference abstracts | 136 | 22.1% |
| oracle kv192 | 117 | 16.1% |
| step 7 last | 99 | 3.8% |
| step 7 `gate=0`, no information at all | 135 | 0.0% |

The arm with the most information is not the longest and the arm with none is not the shortest. This
is a systemic instruction-following shortfall shared by every arm, not a conditioning symptom, so it
is **dropped from run 8**. The one thing it does still bear on: the prompt also says *"no markup,
mathematical notation or headings"* while 13.5% of the targets contain inline maths — the
instruction and the supervision contradict each other on a fifth of the training data, which
strengthens §4.1 rather than §4.3.

---

## 5. The plan for run 8

Ordered by expected value, with the cheap gate first. The executable version of this list — files,
values, gates and the attribution argument — is [`run8_plan.md`](run8_plan.md).

1. **Whiten the node features in preprocessing**, fitted on training nodes with shrinkage. Measured
   64.34% → 78.52% retrieval. No new model component; applies to every arm equally.
2. **Add VICReg variance + covariance terms to Stage A**, and gate Stage A on encoder-output
   pairwise cosine rather than top-1 alone. Top-1 passed at 67% on an unusable representation.
3. **Clean LaTeX and maths from the targets**; normalise whitespace; drop the empty abstract and
   the handful under 30 words from the training set and the negative pool.
4. **Fix the `rank_acc` selection bug** — `NegativePool` is indexed over `sets["train"]`, so no
   validation node is ever in it, `rank_acc` is never computed, and `selection_score` silently falls
   back to LM validation loss. Index over `train + valid`.
5. **Warm λ in over the first epoch** rather than running at 1.0 from step 0. The −0.39 coherence
   regression landed exactly when the rank term started winning.
6. **Sample the ranking window from a random offset** — objective alignment toward methodology, per
   §4.2.
7. **Add a generation canary at validation**: 16 nodes, greedy, count distinct strings. One number,
   logged every 250 steps, that would have caught this at epoch 1 instead of after two evaluation
   jobs.
8. **Add a contrast metric beside `prefix/graph_fraction`** — mean pairwise cosine between `delta_i`
   across a probe batch. Amplitude is not contrast, and no logged number was measuring contrast.
9. ~~State a target length in the prompt.~~ **Withdrawn** — the prompt already asks for 150-180
   words and length does not track information across arms. See the correction in §4.3.

**Not changing:** `gate-init 0.2` (graph_fraction hit 7 — the gate was never the problem), LoRA
r=16, pooled-only, `RANK_HARD_FRACTION 0.5`, τ, T=32, GATv2.

---

## 6. When to add the 1-hop neighbour tokens

**Run 9, on a precondition.** The case for them is stronger than step 6 suggested, and it comes from
the same measurement: individual voyage embeddings sit at pairwise cosine **0.378**, the pooled mean
at **0.660**. Per-neighbour tokens bypass the aggregation collapse entirely — they are the channel
that does not average. And a pooled mean structurally cannot represent "this paper builds on the
method in reference 7", which is exactly what methodology recovery needs.

Step 6 is not evidence against them: it confounded them with r=64 LoRA and a shut gate, and the
pooled channel it was asked to supplement was feeding a near-constant.

The precondition is that run 8 shows the pooled channel actually conditioning — **distinct > 600 on
the eval set, and top-1 clearly above the `gate=0` floor with early divergence from it**. Steps 5, 6
and 7 each changed three or four things and none could attribute a result; run 8 must be
attributable. Adding the tokens is one flag (`--neighbour-tokens 128`), so run 9 costs a submission,
not a rewrite.

---

## 7. What this is expected to buy

The oracle converts fully-informative input into 16.72% top-1 — roughly a 6× loss through the
channel. Whitened graph input carries ~78% retrievable information at better contrast than the
oracle's. The same channel treatment puts it near 13%.

That extrapolation assumes a linearity that cannot be defended, so the honest range is **5–15%
top-1 and 700–900 distinct abstracts**, against today's 0.32% and 195.

Three gates, in order, each cheap enough to stop on:

1. **Before any GPU time** — train-fitted whitening scores ≥75% on eval.
2. **After Stage A** — encoder-output pairwise cosine stays below ~0.4.
3. **Epoch 1 of Stage B** — canary shows >8 distinct of 16, and `rank_hit` leaves chance well before
   step 2,100. It took 2,100 steps last time to extract signal from a near-constant input; with
   contrast it should break early. Still flat at chance by step 800 means the diagnosis is wrong.

---

## 8. Open questions

- **Does whitening survive the GNN?** §3.2 measures whitened features through a *mean*. A trained
  GATv2 is not a mean, and nothing guarantees its output stays decorrelated — which is precisely why
  §3.3 (VICReg) matters more than the transform.
- **Freeze the encoder after Stage A?** Unsettled since `infonce_plan.md` §6. Freezing removes drift
  outright at the cost of the LM never shaping the representation.
- **Why does the model stop at 105 words?** Not truncation, and not conditioning — every arm
  undershoots the instructed 150-180 (§4.3). Unexplained, and now known to be out of scope.
- **Is `year` worth adding?** Clean and free, but it is new information rather than a fix, and run 8
  is already carrying enough.
- **The feature/target mismatch** created by cleaning text that was embedded raw (§4.1) is accepted
  and unquantified.

---

## References

- Mu & Viswanath, *All-but-the-Top: Simple and Effective Postprocessing for Word Representations*, ICLR 2018.
- Jing, Vincent, LeCun & Tian, *Understanding Dimensional Collapse in Contrastive Self-supervised Learning*, ICLR 2022. https://arxiv.org/abs/2110.09348
- Su et al., *Whitening Sentence Representations for Better Semantics and Faster Retrieval*, 2021.
- Bardes, Ponce & LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, 2021. https://arxiv.org/abs/2105.04906
- Ethayarajh, *How Contextual are Contextualized Word Representations?*, EMNLP 2019.
