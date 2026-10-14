# Results

Sources: `docs/chosen_evaluations.md`, `docs/run11_model.md`, `docs/run9_embedding_decoder.md`
§9, `docs/detailed_evaluation.md`, the experiment CSVs (all means and paired differences below
were **recomputed on 2026-10-13** from `experiments/*/evaluated_result.csv` and
`evaluated_result_detailed.csv`), and the Voyage embeddings for §6.

> **Status: not final.** The evaluation is being extended (`05_evaluation_methods.md` §8:
> oracle/zero/joint arms in the detailed evaluator, judge validation, ROUGE/BERTScore
> validity). New results will be added; numbers here will not change, but the table will grow.

All arms: 945 eval papers, Qwen3-14B, greedy decoding; run-11 arms with the 6-gram ban.
Paired difference = per-paper difference, mean, SE = sd/√945. Old judge = `gpt-5.4-mini`,
accuracy 1–10 (`05` §2.1).

---

## 1. The main ablation table (old judge rubric)

| # | arm | input to the LLM | judge accuracy | ≥ 3 | ≥ 5 | coherence | language | median words | TF-IDF top-1 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **text baseline** (`2026-09-10-14364743_qwen14b-baseline`) | ~13 reference titles + abstracts as text; frozen Qwen; no prefix | **3.59** | 79.6% | 16.9% | 7.90 | 8.06 | 136 | **42.9%** | 0.557 |
| 2 | **run 11 joint** (`…_run11-joint`) | GNN → prefix; GNN + adapter + LoRA fine-tuned jointly | 2.94 | 59.7% | 6.8% | 7.86 | 7.85 | 95 | 18.8% | 0.301 |
| 3 | **reader ← GNN** (`…_run11-reader-gnn`) | GNN estimate → prefix (separately trained) | **2.98** | 61.2% | 7.4% | 7.83 | 7.80 | 101 | 18.5% | 0.305 |
| 4 | **reader ← refs-mean** (`…_run11-reader-refs-mean`) | plain mean of reference embeddings → prefix | 2.86 | 55.3% | 6.3% | 7.89 | 7.88 | 101 | 16.5% | 0.271 |
| 5 | **reader ← oracle** (`…_run11-reader-oracle`) — *ceiling* | the paper's own embedding → prefix | not judged yet | | | | | | 27.8% | 0.408 |
| 6 | **reader ← zero** (`…_run11-reader-zero`) — *floor* | zero vector → prefix | not judged | | | | | | 0.1% | 0.008 |

Gold abstracts: median 170 words. Chance TF-IDF top-1: 0.11%.

Reference rows from other runs (same 945 papers, same judge):

| arm | judge accuracy | ≥ 3 | coherence | TF-IDF top-1 |
|---|---|---|---|---|
| run 10a oracle (earlier reader, same recipe, no loop fix) | 3.41 | 70.5% | 6.99 | 32.6% |
| run 10a refs-mean | 2.82 | 52.1% | 7.19 | 15.2% |
| best end-to-end graph run (step 1, frozen LLM) | 1.95 | 19.4% | 7.53 | 4.2% |
| gate=0 controls (one identical string for every paper) | 1.29–1.35 | | | chance |
| retrieval floor: nearest *training* abstract to the reference mean, no LLM | not judged | | | 27.5% raw / 32.0% whitened |

## 2. Paired comparisons (old judge accuracy)

| comparison | question | difference | SE | papers better / worse |
|---|---|---|---|---|
| reader←GNN − reader←mean | does the learned GNN beat the structure-agnostic mean? | **+0.113** | 0.037 | 281 / 220 |
| joint − reader←GNN | does the joint fine-tune add anything? | −0.032 | 0.038 | 264 / 284 |
| joint − reader←mean | full pipeline vs no-learning graph baseline | +0.081 | 0.038 | 292 / 242 |
| text − joint | cost of one vector instead of ~13 reference abstracts as text | +0.649 | 0.047 | 497 / 171 |
| text − reader←GNN | same, separately trained pipeline | +0.617 | 0.048 | 480 / 192 |
| 10a oracle − 10a refs-mean | how much is lost to the estimate (earlier reader) | +0.589 | 0.051 | 445 / 196 |
| joint − 10a refs-mean | run 11 vs run 10a | +0.125 | 0.039 | 313 / 243 |

- The joint arm reaches **82%** of the text baseline's accuracy (2.94 / 3.59) from a 32-token
  prefix carrying one 1024-d vector.
- **Joint − 10a refs-mean (+0.125) comes almost entirely from the loop fix:** on the 129 papers
  where run 10a looped the gain is +0.61; on the other 816 it is +0.05 (SE 0.04), not
  significant (`run9_embedding_decoder.md` §9).
- The zero arm (0.1% top-1) shows the paper-specific content comes from the vector, not from
  LoRA having memorised a generic abstract.

Historical paired gaps for context: Vicuna-7B prefix over text-only +0.473 (SE 0.050); Qwen3-14B
prefix over text-only −0.072 (SE 0.051).

## 3. Detailed evaluator (three arms scored so far)

**[recomputed]** from `evaluated_result_detailed.csv` (945 rows each; facet labels missing for a
few rows where the judge returned no valid label — counts in brackets).

### 3.1 Means

| metric | scale / chance | text baseline | reader ← GNN | reader ← refs-mean |
|---|---|---|---|---|
| field match | 0–1; majority 0.338, shuffled ≈ 0.22 | 0.789 (4 NA) | 0.784 (6 NA) | 0.773 (6 NA) |
| contribution match | 0–1; majority 0.669, shuffled ≈ 0.58–0.62 | 0.700 | 0.686 | 0.691 |
| modality Jaccard | 0–1 | 0.718 | 0.686 | 0.703 |
| method family match | 0–1; majority 0.542, shuffled ≈ 0.37–0.39 | 0.670 (35 NA) | 0.624 (44 NA) | 0.639 (51 NA) |
| subfield | 0–3 | **1.60** | 1.26 | 1.10 |
| problem | 0–3 | **1.69** | 1.39 | 1.25 |
| task | 0–3 | **1.75** | 1.44 | 1.30 |
| claims total | count | 8.07 | 5.59 | 5.42 |
| claims supported | count | 1.75 | 0.80 | 0.64 |
| claims contradicted | count | 3.88 | 2.25 | 2.26 |
| claim precision | 0–1 | 0.218 | 0.140 | 0.117 |
| identification correct (gold among 4 nearest-neighbour distractors) | chance 20% | **41.2%** | 25.4% | 24.6% |
| identification MRR | chance 0.457 | 0.629 | 0.511 | 0.505 |
| TF-IDF top-1 | chance 0.11% | 42.9% | 18.5% | 16.5% |
| on-target margin (gold sim − max reference sim) | relative | −0.169 | −0.126 | −0.130 |
| coherence (detailed call; not comparable to §1) | 1–10 | 6.25 | 7.25 | 7.36 |
| language (detailed call) | 1–10 | 7.60 | 7.77 | 7.89 |
| ROUGE-L / BERTScore F1 | | 0.158 / 0.846 | 0.179 / 0.845 | 0.175 / 0.844 |

### 3.2 Paired differences

**reader ← GNN − reader ← refs-mean** (what the learned GNN adds over the plain mean):

| metric | difference | SE | z |
|---|---|---|---|
| subfield | +0.162 | 0.032 | +5.1 |
| problem | +0.146 | 0.027 | +5.5 |
| task | +0.141 | 0.029 | +4.9 |
| claims supported | +0.159 | 0.031 | +5.1 |
| claim precision | +0.023 | 0.005 | +4.3 |
| TF-IDF rank (lower better) | −19.5 | 4.9 | −4.0 |
| TF-IDF top-1 | +2.0 pts | 1.3 | +1.6 |
| field / contribution / method family / modality | +0.011 / −0.005 / −0.012 / −0.016 | ~0.01 | all \|z\| ≤ 1.4 |
| identification correct | +0.9 pts | 1.6 | +0.5 |
| coherence / language (detailed) | −0.11 / −0.12 | 0.05 / 0.04 | −2.1 / −2.7 |

**text baseline − reader ← GNN** (what the soft-prompt pipeline still lacks vs reading text):

| metric | difference | SE | z |
|---|---|---|---|
| subfield / problem / task | +0.33 / +0.30 / +0.30 | ~0.03 | +9.0 / +9.6 / +9.2 |
| identification correct | +15.8 pts | 1.8 | +8.6 |
| TF-IDF top-1 | +24.3 pts | 1.8 | +13.8 |
| claims total / supported / contradicted | +2.48 / +0.96 / +1.63 | | +38 / +20 / +21 |
| claim precision | +0.077 | 0.007 | +11.5 |
| method family | +4.7 pts | 1.5 | +3.1 |
| modality | +0.032 | 0.014 | +2.3 |
| field | +0.4 pts | 1.4 | +0.3 |
| contribution | +1.4 pts | 1.0 | +1.3 |
| on-target margin | −0.043 | 0.004 | −12.1 (text is *closer to a cited paper* than the GNN arm) |
| coherence (detailed) | −0.99 | 0.06 | −15.7 |

### 3.3 What the detailed metrics say (factual reading)

- **Field is recovered equally well by all three arms** (0.77–0.79, against 0.34 majority and
  ~0.22 shuffled). The soft prompt gets the broad field as reliably as reading the references.
- **Contribution type is at the majority-class level for every arm** (0.69–0.70 vs 0.67) — no
  arm recovers it beyond "it is a new method".
- **Method family is weakly above majority** (0.62–0.67 vs 0.54) and the text baseline is
  ahead by 4.7 points.
- **Subfield / problem / task are where the GNN beats the mean** (+0.14–0.16 on 0–3, z ≈ 5) and
  where the text baseline still leads (+0.30–0.33).
- **Paper-specific identity is weak for the soft-prompt arms:** among the 4 most similar
  papers they pick the right one 25% of the time (chance 20%); the text baseline 41%.
- **Claims:** the soft-prompt arms make fewer concrete claims (5.4–5.6 vs 8.1) and fewer of
  them are supported (precision 0.12–0.14 vs 0.22). The text baseline makes more claims *and*
  more contradicted ones.
- **The text baseline is closer to its references than to its own paper more often** (margin
  −0.169 vs −0.126), i.e. it partly describes cited works. The soft-prompt arms never see
  reference text, so they cannot copy it.

This is the measured basis for the thesis claim **"the domain/field and, partly, the
problem/task are recoverable from the references; the specific contribution and methodology
are not"**: accuracy falls from field (≈ 0.78) → subfield/problem/task (≈ 1.3–1.7 of 3) →
identification among near neighbours (25–41%) → claim precision (0.12–0.22). The same ordering
appeared in the rubric study (domain 7.8 → problem 4.7 → task 3.9 → scope 2.6, `05` §3) and in
an early comparison: methodology wrong 70% of the time for Vicuna and 95% for `gpt-5.4-mini`,
while domain went 58% → 91% and task 22% → 53% (`project_overview.md` §6).

## 4. The GNN stage (phase 2)

| variant | val cosine | val top-1 /806 | eval cosine | eval top-1 /945 |
|---|---|---|---|---|
| plain reference mean | 0.816 | 63.5% | 0.823 | — |
| **GATv2 1-layer cos (winner)** | **0.828** | 70.5% | **0.833** | 70.8% |
| GATv2 1-layer cos+NCE | 0.820 | 72.3% | 0.826 | 75.2% |
| SAGE 1-layer cos | 0.820 | 67.2% | 0.825 | 68.4% |
| GATv2 2-layer cos | 0.822 | 64.4% | 0.822 | 63.2% |
| SAGE 2-layer cos | 0.819 | 66.0% | 0.821 | 65.2% |
| SAGE 1-layer cos+NCE | 0.818 | 72.0% | 0.822 | 72.8% |
| GATv2 2-layer cos+NCE | 0.814 | 71.0% | 0.814 | 69.4% |
| SAGE 2-layer cos+NCE | 0.810 | 70.6% | 0.812 | 71.0% |

- All variants within 0.02 cosine of each other and of the mean; the winner gains +0.010 eval
  cosine. By run 10a's slope (~0.6 judge per 0.1 cosine) that predicts ≈ +0.06 judge; measured
  +0.113.
- 2 layers (references of references) are worse than 1. Attention (GATv2) beats mean
  aggregation (SAGE) slightly. InfoNCE raises retrieval and lowers cosine.

## 5. Decoding: the loop fix (200 eval papers, oracle input)

| setting | sentence-loop rate | TF-IDF top-1 (/200) | mean words |
|---|---|---|---|
| no n-gram ban | 14.0% | 52.0% | 123 |
| `no_repeat_ngram_size = 6` | **0%** | 51.5% | 107 |

## 6. How much information the soft prompt carries

### 6.1 Input budget

| channel | what it carries | size |
|---|---|---|
| text baseline prompt | ~13.5 references (median 27% of a paper's references), titles + abstracts | eval sequences mean 3,655 Vicuna tokens incl. the reserved target (`adapter.md` stage 1c); largest Qwen prompt 3,627 tokens (`qwen_migration.md` step 10) |
| all references as text | every reference's title + abstract | median 8,605 words (~11k tokens), max 21,072 words |
| soft prompt | one 1024-d vector → 32 KV positions × 40 layers | covers **100%** of references, 32 positions |

### 6.2 How close the references can get to the paper (Voyage space, 945 eval papers, **[recomputed]**)

| estimate of the target's embedding | mean cosine to the true embedding |
|---|---|
| plain reference mean | 0.823 |
| best single reference (chosen with the answer) | 0.814 |
| mean of the 3 references closest to the answer (chosen with the answer) | 0.853 |
| best non-negative weighting of the references, fitted **to the answer** (NNLS) | 0.894 |
| GNN estimate (run 11 winner) | 0.833 |
| **control:** the same fit against an **unrelated** paper's embedding | mean 0.496 → NNLS 0.583 |

Reading: fitting weights to the answer gains +0.071 over the mean for the real target and
+0.087 for an unrelated one — fitting gains no more for the true paper than for a random one, so
the plain mean is already near what the references can say. The GNN's +0.010 is in line with
that.

### 6.3 The ceiling ladder (TF-IDF top-1, same reader)

zero 0.1% → refs-mean 16.5% → GNN 18.5% / joint 18.8% → oracle 27.8% → text baseline 42.9%.

- **Estimate gap** (GNN → oracle): 9.3 points; the larger part of what separates the pipeline
  from its own ceiling.
- **Reader gap** (oracle → text): 15.1 points. Run 10a's reader, same recipe, reached 32.6%
  oracle top-1, so the reader varies by ~5 points between retrainings.

### 6.4 Where the estimate fails **[recomputed]**

Judge accuracy by quintile of cos(reference mean, own embedding) — how well the references
describe the paper:

| quintile (mean cos) | text | joint | reader←GNN | reader←mean | 10a oracle |
|---|---|---|---|---|---|
| Q1 (0.75) | 2.94 | 2.52 | 2.57 | 2.42 | 3.34 |
| Q2 (0.81) | 3.29 | 2.87 | 2.88 | 2.76 | 3.39 |
| Q3 (0.83) | 3.75 | 2.81 | 3.03 | 2.78 | 3.30 |
| Q4 (0.85) | 3.89 | 3.19 | 3.13 | 3.14 | 3.44 |
| Q5 (0.88) | 4.08 | 3.32 | 3.27 | 3.21 | 3.57 |

- Every reference-based arm (text included) rises with how well the references describe the
  paper; the oracle arm (which does not use references) is flat — so this is not some papers
  simply being easier.
- The graph arms' gap to the oracle is largest where references do not describe the paper
  (Q1: 2.52 vs 3.34) and smallest where they do (Q5: 3.32 vs 3.57).

By out-degree (old judge):

| out-degree | n | text | joint | reader←GNN | reader←mean | 10a oracle |
|---|---|---|---|---|---|---|
| 5–25 | 210 | 3.39 | 2.77 | 2.78 | 2.68 | 3.07 |
| 26–50 | 250 | 3.68 | 3.03 | 3.05 | 2.89 | 3.42 |
| 51–75 | 250 | 3.63 | 2.97 | 3.04 | 2.91 | 3.46 |
| 76–100 | 235 | 3.63 | 2.97 | 3.00 | 2.94 | 3.64 |

The oracle arm also rises with degree although it sees no graph, so degree is confounded with
the paper itself (e.g. gold length, S2 coverage); degree effects must not be read as "more
neighbours help" without stratification. cos(reference mean, own) is nearly uncorrelated with
degree (r = 0.09).

## 7. Qualitative summary (from blind-judge reading, Vicuna era)

> Without the graph, the model typically writes a fluent abstract of *one of the cited papers*
> — a real, recognisable, wrong paper. With the graph, it writes about the right research
> problem, the right method family, the right vocabulary. But it still cannot recover the
> paper's actual contribution — its named method, its numbers, its conclusions — and fills that
> gap with plausible invention.

(`project_overview.md` §6, 57 pairs, Vicuna-7B text + prefix arms.) A comparable reading of
run-11 outputs has not been written up; examples can be drawn from the run-11
`generations.csv` files.

## 8. Pending results (placeholders)

- Run-11 oracle arm: old judge and detailed evaluator.
- Zero and joint arms: detailed evaluator (zero needed for the validity rule).
- Pairwise preference (metric D): GNN vs mean, joint vs GNN, joint vs text.
- Evaluation-validity experiments (`05` §8).
- Any true end-to-end run or other-dataset runs (`07`).
