# Run 8 — the change spec

Written 2026-09-30. The analysis behind every item is in
[`step8_representation_plan.md`](step8_representation_plan.md); this file is the decision list:
what changes, in which file, why, and what has to be true for the change to count.

**The diagnosis in one sentence.** The pooled reference vector the adapter consumes has
two-thirds of its energy in a single direction shared across all nodes (pairwise cosine 0.6602),
`PrefixAdapter.trunk`'s LayerNorm cannot remove a between-node component, so the adapter received
a near-constant and greedy decoding emitted a near-constant — while `prefix/graph_fraction` ran at
7.0 because it measures amplitude, not contrast.

**The goal is top-1 identifiability.** Everything else (judge accuracy, BertScore, ROUGE) moves with
length and register and has been misleading us; top-1 and distinct-count are the two that have never
lied. Today: **0.32% top-1, 195/945 distinct.** Chance is 0.106%.

---

## 1. Tier 0 — free, and they happen before any GPU time is requested

| # | change | file | verification |
|---|---|---|---|
| 1 | **ZCA-whiten the node features**, fitted on the 383,005 non-eval/valid nodes, eigenvalue floor `1e-3·λmax`, L2-renormalised afterwards. Written to `node_embeddings_whitened.npy`; `load_graph` already takes an `embeddings` path, so nothing else changes. | new `src/adapter/whiten_features.py`, `config.py` | **measured**: pooled pairwise cosine 0.6602 → **0.0547**, eval retrieval 64.34% → **72.59%**, participation ratio 122 → 948 of 1024. Gate is cosine < 0.10 and no loss of retrieval — *not* a retrieval threshold; see below |
| 2 | **Fix the `rank_acc` selection bug.** `NegativePool` is indexed over `sets["train"]`, so no validation node is ever in it, `rank_acc` is never computed, and `selection_score` silently falls back to LM validation loss — the criterion step 6 proved is anti-correlated with conditioning. Index over `train + valid`. | `train.py:525` | `valid/rank_acc` appears in W&B from the first validation, and is not `None` |
| 3 | **Clean the targets**: unwrap `\textit{…}`/`\emph{…}`, strip `$` delimiters keeping their contents, decode `\%` `\&` `\\`, collapse whitespace runs, drop the one empty abstract and anything under 30 words from the training set *and* the negative pool. | `text.py`, `rank.py` | the §4.1 artefact counts go to ~0 on a re-audit; train-set size falls by <10 nodes |

Item 3 is not cosmetic: the system prompt already instructs **"no markup, mathematical notation or
headings"**, and 13.5% of the targets contain inline maths. The instruction and the supervision have
been contradicting each other on a fifth of the training data.

---

## 2. Tier 1 — the run's actual hypothesis

| # | change | file | value |
|---|---|---|---|
| 4 | **VICReg variance + covariance terms in Stage A**, on the head output *before* `F.normalize`. `info_nce` is invariant to anisotropy — it scored 67% top-1 on a representation the adapter reads as a constant — so the objective, not just the input, has to ask for a spread representation. | `pretrain_encoder.py` | `λ_var = 1.0` (hinge at σ ≥ 1), `λ_cov = 0.04`, i.e. VICReg's 25:1 ratio |
| 5 | **Warm λ_rank in linearly over epoch 1** (0 → 1.0 across ~484 optimizer steps) instead of 1.0 from step 0. `L_LM` and `L_rank` genuinely conflict, and the −0.39 coherence regression landed exactly when the rank term started winning. | `train.py`, `config.py` | `LAMBDA_RANK = 1.0`, `RANK_WARMUP_STEPS = 484` |
| 6 | **Sample the ranking window from a random offset** rather than always the first 32 tokens. This is objective alignment toward methodology — the opening states the problem, the middle states the method — **not** a bug fix: the golds are 98.6% distinct by word 8, so the current window was never targeting formulaic text. | `rank.py` | start ~ U[0, len − 32] |

Items 1 and 4 are the same correction applied twice, by transform and by objective. Both are in
because a fixed transform goes stale as the encoder trains (§8 of the analysis), and because if only
one of them ships and the run works we learn less than if the representation is fixed properly.

---

## 3. Tier 2 — instrumentation, and it is not optional

Three runs in a row produced a collapsed model that no logged number flagged until an evaluation job
had already burned seven GPU hours.

| # | metric | what it would have caught |
|---|---|---|
| 7 | **Generation canary at validation** — 16 fixed valid nodes, greedy, 64 new tokens, count distinct strings. Logged every `EVAL_EVERY = 250` steps. | the entire step-7 failure, at epoch 1 |
| 8 | **`prefix/graph_contrast`** — mean pairwise cosine between `delta_i` over a probe batch, logged beside `graph_fraction`. | a loud channel carrying nothing |
| 9 | **`encoder/output_cosine`** — the same quantity on the encoder output, so Stage A's gate is contrast and not top-1 alone. | Stage A passing at 67% on an unusable representation |

---

## 4. Not changing, and why

- **`--gate-init 0.2`** — `graph_fraction` reached 7.0. The gate was never the constraint.
- **LoRA r=16** — decoder capacity is anti-correlated with graph use while conditioning is weak
  (2.86% → 1.16% → 0.32% across frozen / r16 / r64). r=64 returns after the channel works.
- **Pooled prefix only, no neighbour tokens** — run 9, on the precondition in §6 of the analysis.
- **`RANK_HARD_FRACTION 0.5`, τ, T=32, GATv2, the prompt** — untouched.
- **Target length.** I had this on the list as "state a target length in the prompt". The prompt
  *already* says 150–180 words, and length does not track information: the text baseline with full
  reference abstracts writes 136 words, the oracle 117, and the `gate=0` arm with no information at
  all writes 135. It is a systemic instruction-following shortfall, not a conditioning symptom, and a
  105-word abstract still caps at ROUGE-1 0.77 while we sit at 0.27. **Dropped from run 8.**

---

## 5. Keeping run 8 attributable

Steps 5, 6 and 7 each changed three or four things and none of them could assign credit. Run 8
changes six, so the split has to be stated:

- **2, 3, 7, 8, 9** are bug fixes and instrumentation. They cannot plausibly produce a false positive.
- **1 and 4** are the hypothesis, and they are the *same* hypothesis measured two ways.
- **5 and 6** are objective shaping, and they are the two that could muddy the result. If run 8 lands
  between the gates rather than clearly above them, 5 and 6 are what gets ablated in 8b.
- The **`gate=0` arm stays on**, as the within-run floor. Adapter-minus-gate0 is the graph's
  contribution and nothing else.

---

## 6. Order of operations

```
1.  whiten features locally            -> gate A: pooled cosine < 0.10, retrieval not worse
2.  re-audit cleaned text locally      -> artefact counts ~0
3.  Stage A on GPU (cheap, no 14B)     -> gate B: output pairwise cosine < 0.4 AND top-1 > 64.34%
4.  Stage B, 10 epochs, patience 2     -> gate C at epoch 1 (below)
5.  evaluate best.pt + gate0, 2 arms   -> ~14 h of the 24 h walltime
```

Evaluation goes back to **`best.pt`** once item 2 lands — the last-epoch override in
`run_generate_eval.sh` exists only because selection was silently reading LM validation loss.

**Gate A is not a retrieval threshold, and that is a correction.** The first draft of this plan set
it at "eval retrieval ≥ 75%", transcribed from a measurement whose transform had been fitted on
nodes including the eval set. The honest train-only fit reaches 72.59%, and the gate as written
would have blocked a transform that does exactly what it is for. Retrieval was never the diagnosis:
step 7 had 64.34% of the information retrievable and delivered 0.11%. The quantity that failed is
**contrast between nodes**, and that is what the gate now reads.

**Gate C, at epoch 1 of Stage B:** canary shows **> 8 distinct of 16**, and `rank_hit` has left
chance (0.111) well before step 2,100. It took 2,100 steps last time to extract signal from a
near-constant input; with contrast it should break early. **Still flat at chance by step 800 means
the diagnosis is wrong** — stop the job rather than spend the remaining epochs.

---

## 7. What success looks like

| | today | run 8 target |
|---|---|---|
| top-1 identifiability | 0.32% | **5–15%** |
| distinct abstracts | 195 / 945 | **700–900** |
| 8-word distinct openings | 6 | > 300 |
| median words identical to `gate=0` | 93 | < 15 |

The oracle converts fully-informative input into 16.72% top-1 through this same channel, so ~13% is
the extrapolation and 5–15% is the honest range around it. Anything at or below the step-2 frozen
run's **2.86%** means the representation was not the binding constraint after all, and the next
question is the channel — neighbour tokens — rather than the features.
