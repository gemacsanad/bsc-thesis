# Evaluation methods

Sources: `src/evaluation/evaluate_results.py`, `evaluate_results_scores_only.py`,
`evaluate_detailed.py`, `docs/detailed_evaluation.md`, `docs/chosen_evaluations.md`,
`src/adapter/identify.py`, `loop_check.py`, `experiments/rubric_sensitivity/README.md`,
`experiments/sentiment_analysis/README.md`, `experiments/prompt_sensitivity_gpt/README.md`,
`docs/project_overview.md`, and statistics recomputed from the experiment CSVs on 2026-10-13
(**[recomputed]**).

**Status: not final.** Further evaluation experiments are planned (§8); results from them will
be added to `06_results.md`.

---

## 1. What is held fixed in every evaluation

- **The same 945 eval papers** (`data/frozen/eval_nodes_capped.json`), test split, out-degree
  5–100.
- **Greedy decoding**, batch 1, bf16, same GPU type; graph arms with `no_repeat_ngram_size=6`
  in run 11. Reruns reproduce.
- **Gold abstracts are the raw text** (not the cleaned training targets).
- Every arm writes `generations.csv` with `node_id, mag_id, original_abstract,
  generated_abstract`; scoring scripts are run unchanged on each.
- **Paired comparison per paper**: difference per node, mean, SE = sd/√n. All arms are
  evaluated on the same papers, so paired SEs are much smaller than unpaired ones.

## 2. The instruments

### 2.1 LLM-as-a-judge, original rubric ("old rubric", R1)

- Script: `evaluate_results_scores_only.py` (the same rubric as `evaluate_results.py` minus the
  free-text reason and failure categories). Judge **`gpt-5.4-mini`, temperature 0**, JSON
  output, retries with exponential backoff; failed calls are written as `ERROR` and excluded.
- The judge sees the **original (gold) abstract** and the **generated abstract** and returns
  three integers 1–10. System prompt (verbatim):

  > You are a rigorous scientific peer reviewer. Given an original paper abstract and a
  > generated abstract for the same paper, evaluate the generated abstract on three metrics.
  > […] Rubrics:
  > - accuracy (1-10): How closely does the generated abstract align with the core scientific
  >   objective, problem domain, and baseline methodologies of the original? Penalise explicit
  >   contradictions; do not penalise creative but plausible predictions.
  > - coherence (1-10): Is the argument logically coherent? Does it progress smoothly from a
  >   background problem to a proposed solution, or does it feel like a disjointed list of
  >   ideas?
  > - language (1-10): Does the text adhere to scientific publishing norms? Score on precise
  >   domain terminology, appropriate passive/active voice, and absence of generic
  >   conversational fluff.

- `llm_accuracy` is the headline because it is the only one asking *is this about the right
  paper*. Also reported: share of papers scoring ≥ 3 and ≥ 5.
- `evaluate_results.py` (used for the early/archived runs) additionally asks for a reason and
  then labels 11 failure categories (`cat_wrong_domain`, `cat_multi_paper_contamination`,
  `cat_wrong_methodology`, `cat_wrong_task`, `cat_too_generic`, `cat_truncated_incomplete`,
  `cat_correct_domain`, `cat_correct_task`, `cat_methodology_partially_correct`,
  `cat_scientific_language`, `cat_coherent_structure`).

### 2.2 ROUGE and BERTScore

- ROUGE-1/2/L F1 (`rouge-score`, Porter stemming) of generated vs gold.
- BERTScore P/R/F1 (`bert-score`, `lang="en"` → default `roberta-large`), **without baseline
  rescaling**.
- Kept in every scored CSV; free (local).

### 2.3 TF-IDF identification (no judge)

- `src/adapter/identify.py`: a `TfidfVectorizer(sublinear_tf=True, stop_words="english")` is
  fitted **on the gold abstracts only**; each generation is ranked against all golds of the set
  by cosine; rank 1 = its own gold is the closest. **Ties count against the generation** (an
  empty or constant text ranks last, so a collapsed model cannot look perfect).
- Reported as top-1 and MRR. On 945 papers chance top-1 is 1/945 = 0.11%; on the 128-paper
  validation probe 0.8%.
- It is lexical: it understates paraphrase. It is used to compare arms with each other, where
  that bias is shared.
- The same instrument is used in training (validation `val_id`, model selection), in
  `loop_check.py`, and as `tfidf_rank` in the detailed evaluator.

### 2.4 The detailed evaluator (`evaluate_detailed.py`)

Built 2026-10-10 because one 1–10 score hides *where* the graph helps. Four judge calls per
paper, split by what the judge is allowed to see. `llm_accuracy` is dropped; coherence and
language keep their rubric text word for word.

**A. Blind facets** — the judge labels **one abstract at a time** from fixed lists, never seeing
the other. Gold labels are cached once (`data/frozen/gold_facets.json`, 945 entries) and shared
by every arm.

| column | label set | score |
|---|---|---|
| `field_match` | 14 broad CS areas | 1 if equal |
| `contribution_match` | new method, dataset/benchmark, empirical study, survey, theory, system/tool, application, position paper | 1 if equal |
| `modality_jaccard` | text, images, video, audio, graphs, tabular, time series, code, 3D, sensor, none (multi-label) | Jaccard 0–1 |
| `method_family_match` | 12 coarse families (transformer/LLM, other deep net, RL, Bayesian, classical ML, optimisation, …) | 1 if equal |

**B. Comparison** — the judge sees gold and generation:
`subfield`, `problem`, `task` on an anchored **0–3** scale (0 different, 1 related but
different, 2 same with differences in detail, 3 same; methods, datasets and numbers excluded),
and claim counts: `claims_total` (concrete checkable claims in the generation),
`claims_supported` (stated/implied by the gold), `claims_contradicted`, `claim_precision` =
supported/total (counts clipped so supported + contradicted ≤ total). Coherence and language are
also scored in this call.

**C. Identification** — the judge sees the generation and **five real abstracts** (the gold +
4 distractors) and ranks which was described. Distractors = the gold's 4 nearest papers by
Voyage embedding out of all 384,756, excluding the paper itself, its references, near-duplicates
(cosine ≥ 0.95) and abstracts under 50 words; order seeded by node id, so every arm gets the
same question. `id_rank` 1–5, `id_correct` (chance 20%), MRR (chance 0.457).

**D. Pairwise preference** (optional, `--against`): which of two arms' abstracts describes the
same paper more accurately, asked in **both orders**; 1 / 0.5 / 0 (a judge always picking
position 1 scores exactly 0.5). Not yet run.

**No judge:** `tfidf_rank`; `ontarget_gold_sim`, `ontarget_max_ref_sim` (highest TF-IDF
similarity to any of the paper's reference abstracts) and `ontarget_margin` = gold sim − max ref
sim (negative = closer to a cited paper than to its own; read relative to other arms — the max
over ~50 references makes it negative for every arm).

**Reading rules fixed before scoring** (`detailed_evaluation.md`): report every metric;
**validity rule** — a metric is usable only if zero < refs-mean ≤ oracle on it (needs the zero
and oracle arms scored; not yet done); paired per-paper differences with SE and Holm correction;
pilot on ~150 papers first.

**Cost:** ~4.8× one old-rubric run per arm; gold labels once ~1.2×; pairwise ~2× per pair.

**Important:** coherence from the detailed evaluator is **not comparable** to old-rubric
coherence — it is asked in a call where the judge also compares content with the gold. E.g. the
text baseline: 7.90 (old rubric) vs 6.25 (detailed) **[recomputed]**.

### 2.5 Chance baselines for the blind facets **[recomputed]**

| facet | majority class (share of golds) | number of classes in golds | expected agreement if labels were shuffled (text / GNN / mean arm) |
|---|---|---|---|
| field | computer vision, 0.338 | 14 | 0.226 / 0.223 / 0.228 |
| contribution | new method or model, 0.669 | 8 | 0.621 / 0.579 / 0.585 |
| method family | other deep neural network, 0.542 | 10 | 0.389 / 0.373 / 0.383 |

So `contribution_match` ≈ 0.69 is barely above always answering "new method", while
`field_match` ≈ 0.78 is well above both baselines.

### 2.6 Training-time and decoding instruments

- `valid/delta` (LM loss gain from the vector), `val_id` (generation-based TF-IDF MRR on 128
  validation papers), `canary/distinct` (16 papers, 48 greedy tokens), stop rules
  (`03_final_model.md`).
- `loop_check.py`: share of outputs repeating a sentence of ≥ 5 words, top-1, mean length.

## 3. Study: rubric sensitivity (does the ruler decide the effect?)

`experiments/rubric_sensitivity/`. The **same** two sets of generations (Vicuna era-1 baseline
and GraphSAGE adapter, 08-15), 150 papers chosen by out-degree only, judge unchanged, four
accuracy rubrics; coherence/language text identical. 1,800 judge calls. A **negative control**:
two adapter runs differing only in repetition penalty (1.15 vs 1.0).

| rubric | what accuracy scores | baseline | adapter | graph gap | effect size | control gap (p) |
|---|---|---|---|---|---|---|
| R1 (current) | objective, domain **and methodology** | 2.83 | 3.21 | +0.387 | 0.29 | 0.140 (0.18) |
| R2 | problem and task only; methods/models/datasets/numbers excluded | 3.17 | 4.01 | **+0.833** | 0.45 | 0.360 (**0.01**) |
| R3 | R2 split into domain/problem/task/scope + overall | 2.98 | 3.45 | +0.473 | 0.32 | 0.227 (0.05) |
| R4 | R2 + cap at 3 if it describes a single cited work | 2.50 | 3.24 | +0.740 | 0.50 | 0.087 (0.64) |

R3 decomposition: domain 7.83 → 8.19 (+4.7%), problem 4.65 → 5.05 (+8.6%), task 3.85 → 4.34
(+12.6%), scope 2.64 → 3.16 (+19.7%) (baseline → adapter).

Conclusions recorded: **the rubric decides roughly half the measured effect**; R2 is more
sensitive but less specific (its negative control became significant); R4 was rejected — its
gain came from lowering the baseline and its penalty did not land on the papers that actually
copied a cited work (correlation −0.09); scores fall as the question gets more specific (domain
→ scope), and the graph's advantage grows with specificity. Caveats: no noise floor (same text
scored twice) was measured; 150 papers; in R3 only domain and scope are individually
significant. A cheaper judge (`gpt-5.6-luna`) was tried and rejected: it separated the arms
~1.5× less well and cannot run at temperature 0 (`judge_comparison/`).

## 4. Study: blind-judge pairwise comparisons (`experiments/sentiment_analysis/`)

Three rounds; an LLM judge compares two generations of the same paper against the gold, blind
to scores and sampling. Round 1 (baseline vs adapter, 57 pairs) gave the qualitative summary:
*without the graph the model writes a fluent abstract of one of the cited papers; with it, a
fluent abstract of a plausible but non-existent paper on the right topic.* Round 2 showed a
prompt rewrite's flat mean was +0.81 on the floor tier and −2.00 on the top tier. Round 3 found
a preference judge rewarding better-written, more hedged text that the accuracy rubric does not
reward — **do not tune on a preference judge**. Round 3 also holds `ref_overlap*.csv` (overlap
with gold vs most similar cited reference; basis of the copying analysis: 10.4% near-verbatim
copies).

## 5. Study: prompt sensitivity (`experiments/prompt_sensitivity_gpt`, `_qwen`)

Four prompt eras on the same 48 papers with byte-identical references. `gpt-5.6-luna`: 4.583 /
4.938 / 5.417 / 5.812; Vicuna-7B on the same papers 4.292 / 3.771 / 3.667. Qwen3-14B: 3.875 /
4.438 / 5.292 / 5.333. The 48 papers were stratified on how their scores moved between eras,
so means are not estimates of the 945; paired comparisons are. The 945-paper text baseline
uses **era 1**; the graph-only prompt is derived from **era 4** (+1.46 for Qwen on these 48) —
the wording confound works *against* the graph arms.

## 6. Evidence already in hand: ROUGE and BERTScore cannot separate the arms **[recomputed]**

| arm | judge accuracy | BERTScore F1 (sd) | ROUGE-1 | ROUGE-2 | ROUGE-L | distinct outputs |
|---|---|---|---|---|---|---|
| text baseline (Qwen) | 3.59 | 0.846 (0.011) | 0.336 | 0.054 | 0.158 | 945 |
| run 10a oracle | 3.41 | 0.844 (0.017) | 0.328 | 0.066 | 0.186 | 945 |
| run 11 reader ← GNN | 2.97 | 0.845 (0.013) | 0.324 | 0.065 | 0.179 | 939 |
| run 11 reader ← refs-mean | 2.86 | 0.844 (0.013) | 0.318 | 0.062 | 0.175 | 938 |
| step-7 gate=0 control: **one identical abstract for all 945 papers** | 1.35 | **0.828** (0.010) | 0.269 | 0.031 | 0.146 | 1 |
| run-8 gate=0 control: one identical abstract | 1.29 | 0.819 (0.010) | 0.259 | 0.033 | 0.167 | 1 |
| `_archive/random-sample`: Vicuna given **random** references (200-paper set) | 1.03 | 0.820 (0.015) | 0.240 | 0.022 | 0.128 | 200 |

- **BERTScore:** a single fixed abstract written for every paper scores 0.828, only 0.018
  below the text baseline; the text baseline and the GNN arm differ by 0.001 while the judge
  differs by 0.62. The entire range of meaningful systems is 0.844–0.846. 138 of 142 judged
  random-reference outputs scored accuracy 1 (95.8% labelled "wrong domain"), yet BERTScore F1
  0.820.
- **ROUGE:** the text baseline has *lower* ROUGE-2 and ROUGE-L than the GNN arm (0.054 vs
  0.065; 0.158 vs 0.179) despite being judged clearly better and identifying its paper twice as
  often. Generated abstracts are shorter than the golds (95–136 vs 170 median words), so ROUGE
  is partly measuring length.
- **Per-paper, within an arm** the metrics are not unrelated to the judge: Spearman ρ between
  judge accuracy and BERTScore F1 is +0.42 to +0.49 in the main arms, and +0.24 to +0.35 for
  ROUGE-L. The failure is **between-system discrimination** (the quantity the thesis needs),
  not a total absence of signal. State it that way.

## 7. Evidence already in hand: the judge tracks real differences

- **Controls behave correctly:** constant-output gate=0 arms 1.29–1.35; random references 1.03;
  zero-vector reader 0.1% TF-IDF top-1 (not judged).
- **Agreement with an objective instrument:** across arms the judge ordering matches TF-IDF
  identification (text 42.9% > oracle > GNN 18.5% > mean 16.5% > zero 0.1%) and the run-10a
  cosine–score relation (+0.6 per +0.1 cosine).
- **Rubric matters** (§3) — the absolute numbers depend on the rubric, the ordering of
  baseline vs adapter did not change across four rubrics.
- **Same-vendor bias:** judge and the archived `gpt-5.4-mini` generator are the same model;
  self-preference (Panickssery et al. 2024) is a known risk. The final arms are generated by
  Qwen, so this affects only archived comparisons.

## 8. Planned evaluation experiments (not yet run)

The author plans to add experiments that (a) demonstrate that ROUGE and BERTScore are not
meaningful for comparing systems on this task, and (b) show how close the LLM judge is to
correct. Designs that fit the existing data (proposals; adapt as needed):

### 8.1 Overlap/embedding metrics are not meaningful here

1. **Control ladder on the 945 papers**, scored with ROUGE, BERTScore (raw and
   `rescale_with_baseline=True`), the judge and TF-IDF identification:
   - gold vs itself (upper bound),
   - real generation vs its own gold (each arm),
   - **shuffled pairing**: each generation scored against a *different* paper's gold,
   - **same-field shuffle**: against a different gold from the same field/category (hardest),
   - a constant string (existing gate=0 outputs),
   - the nearest *reference* abstract returned as the "generation" (copying a cited work).
   A metric is meaningful for system comparison only if it separates these with a margin
   larger than its spread; report effect sizes (difference / pooled sd) per metric.
2. **Discrimination of known-different systems**: paired effect size of each metric for text vs
   GNN, GNN vs mean, oracle vs mean, against the judge and TF-IDF.
3. **Length confound:** regress ROUGE on generated length; compare after truncating golds to
   the generated length.
4. **BERTScore rescaling** and a different backbone (e.g. a scientific-text model) to show the
   compression is intrinsic, not a configuration accident.

### 8.2 How correct is the judge?

1. **Human agreement:** a stratified sample (e.g. 100–150 papers across arms and score levels)
   rated by humans on the same rubric (or on the detailed evaluator's 0–3 problem/task scale and
   the blind facets); Spearman/Kendall correlation and Krippendorff's α between judge and
   humans, and between humans (Chiang & Lee 2023; SummEval protocol).
2. **Test–retest noise:** re-score one arm (or a 150-paper subset) with the same judge, same
   settings; the per-paper disagreement gives the noise floor every gap must exceed.
3. **Judge swap:** re-score with a different model family; check that arm ordering and paired
   gaps hold.
4. **Bias probes:** position bias (pairwise both orders; already built into metric D), length
   bias (correlation of score with length within an arm), and gold-similarity bias
   (paraphrased gold vs copied cited abstract).
5. **Validity rule:** score the zero and oracle arms with the detailed evaluator and keep only
   metrics where zero < mean ≤ oracle.
6. **Objective anchors:** agreement between judge identification (metric C) and TF-IDF / Voyage
   retrieval on the same papers.

## 9. Evaluation coverage of the final arms (as of 2026-10-13)

| arm | old judge | detailed evaluator | TF-IDF |
|---|---|---|---|
| text baseline | ✅ | ✅ | ✅ |
| run 11 joint | ✅ | ❌ | ✅ |
| run 11 reader ← GNN | ✅ | ✅ | ✅ |
| run 11 reader ← refs-mean | ✅ | ✅ | ✅ |
| run 11 reader ← oracle | ❌ | ❌ | ✅ |
| run 11 reader ← zero | ❌ | ❌ | ✅ |
| run 11 loop tests (200 papers) | — | — | ✅ |
| retrieval floor (nearest training abstract) | ❌ | ❌ | ✅ (32.0% whitened / 27.5% raw) |
