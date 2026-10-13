# Detailed evaluation: the accuracy score split into separate metrics

Written 2026-10-10. Script: `src/evaluation/evaluate_detailed.py`. Not yet run on any arm.

## Why

Every accuracy number so far is one 1–10 judge score covering objective, domain and
methodology at once. Run 11 showed that a single score hides what the graph contributes:

- reader ← GNN beats reader ← mean by +0.11 (SE 0.04) overall, but the GNN's measured
  strength is narrower. At the embedding level it identifies the right paper 9 points more
  often, and its outputs are less alike from paper to paper.
- The rubric study (`experiments/rubric_sensitivity`, R3) found that the graph's advantage grew
  as the question became more specific: +4.7% on domain, +19.7% on scope.

This evaluator scores each dimension separately, so the thesis can show *where* each component
helps. The intent is to report every metric, including the ones where the model loses.

## How to run

The same interface as `evaluate_results_scores_only.py`: one generations file in, one scored
file out. It resumes from the output file if it exists.

```
python src/evaluation/evaluate_detailed.py --input <arm>/generations.csv --output <arm>/detailed.csv
python src/evaluation/evaluate_detailed.py --input <a>/generations.csv --output <a>/pair_vs_b.csv --against <b>/generations.csv
```

- `--against` is optional. It adds only the pairwise metric (D).
- `--gold-facets` (default `data/frozen/gold_facets.json`) caches the labels of the gold
  abstracts. The first arm pays for them; every later arm reuses the same labels, so all arms
  are compared against identical gold labels.
- The judge is `gpt-5.4-mini` at temperature 0, as before.

## Kept from the current evaluator

Unchanged:
- `rouge1_f`, `rouge2_f`, `rougeL_f`
- `bertscore_p`, `bertscore_r`, `bertscore_f1`
- `llm_coherence`, `llm_language`, with word-for-word the same rubric text
- `num_of_neighbors`

`llm_accuracy` is dropped. It is replaced by the metrics below.

## One prompt or several: split by what the judge sees

There are four judge calls per paper, not one, and not one per metric. The split follows one
rule: **metrics that need the judge to see different things go in different calls.**

- **Blind labels must not see the gold.** If the judge sees both abstracts, it tends to label
  the generation to match the gold. Labelling each abstract on its own makes "same field?" a
  comparison of two independent labels.
- **Identification must not see which abstract is the gold.** That is the whole test.
- **Metrics that do need both abstracts share one call** (subfield, problem, task, claims,
  coherence, language). Splitting them would cost 6× for little gain. The price is some halo
  effect: one judgement can colour the others. This is why the categorical facets that can be
  scored blind are taken out of this call.

## The metrics

### A. Blind facets — does it say the right *kind* of thing?

The judge labels one abstract at a time from fixed lists. The gold is labelled once (cached),
the generation per arm. The score is whether the two labels agree.

| column | labels | score |
|---|---|---|
| `field_match` | 14 broad CS areas (ML, CV, NLP, security, theory, …) | 1 if same, else 0 |
| `contribution_match` | new method, dataset or benchmark, empirical study, survey, theory, system or tool, application, position paper | 1 if same, else 0 |
| `modality_jaccard` | text, images, video, audio, graphs, tabular, time series, code, 3D, sensor, none (several allowed) | overlap of the two sets, 0–1 |
| `method_family_match` | 12 coarse families (transformer or LLM, other deep net, RL, Bayesian, classical ML, optimisation, …) | 1 if same, else 0 |

The raw labels are kept (`gold_field`, `gen_field`, …), so confusion tables can be drawn.

**Why labels instead of 1–10:** a match against a fixed list has a right answer. It is more
reproducible than a scale, and its gold side is identical for every arm.

**What to expect:**
- Field will be near the ceiling for every arm, so it is a sanity check, not a separator.
- Contribution type is the dimension where R3 found the largest graph effect (scope).
- Method family was found unrecoverable from references in the earlier rubric study. It is
  included to show that, not to win on.

### B. Comparison — is it about the same problem, and are its claims true?

The judge sees the gold and the generation.

| column | question | scale |
|---|---|---|
| `subfield` | Same research niche, narrower than field ("graph contrastive learning", not "ML")? | 0–3 |
| `problem` | Same underlying problem, gap or motivation? | 0–3 |
| `task` | Same concrete task (detection vs segmentation, …)? | 0–3 |
| `claims_total` | How many concrete, checkable claims the generation makes (a named method, dataset, task, quantity or result) | count |
| `claims_supported` | How many of them the gold states or implies | count |
| `claims_contradicted` | How many of them the gold contradicts | count |
| `claim_precision` | supported / total | 0–1 |

- **The 0–3 scale is anchored:** 0 different, 1 related but different, 2 same with
  differences in detail, 3 same. A short anchored scale is more stable at temperature 0 than
  1–10, which compressed the old scores near the bottom. Methods, datasets and numbers are
  excluded from these three scores, as in the R2 rubric.
- **Claims must be read together.** A vague abstract makes no claims and so contradicts
  nothing. Contradictions alone would reward vagueness. `claims_total` (specificity) is the
  counterweight, and `claim_precision` is the rate. Counts are clipped so that supported +
  contradicted never exceeds total.

### C. Identification — is it about *this* paper?

| column | how | scale |
|---|---|---|
| `id_rank`, `id_correct` | The judge sees the generation and five real abstracts: the gold and 4 distractors. It ranks which one was described. | rank 1–5; correct = rank 1 (chance 20%) |

- **Distractors are the gold's 4 nearest papers by Voyage embedding**, out of all 384k papers.
  So they come from the same niche, and telling them apart needs paper-specific content.
- **Excluded from the distractors:**
  - the paper itself;
  - its own references, which belong to the target-fidelity metric below;
  - near-duplicates with cosine ≥ 0.95, since other versions of the same paper reach 0.996;
  - abstracts under 50 words.
- **The distractors and their order depend only on the paper**, using the node_id as seed. So
  every arm gets the same question. `id_options` records the distractors.
- **This is the closest judge-based version of the GNN's proven strength** (retrieval). It has
  a right answer, so it involves no scale.

### D. Pairwise preference (only with `--against`)

| column | how | scale |
|---|---|---|
| `pair_score` | The judge sees the gold and the two arms' abstracts, and picks the one describing the same paper more accurately. Writing quality and length are to be ignored. It is judged twice, with the order swapped. | 1 = this arm wins both orders, 0.5 = split or tie, 0 = loses both |

- **Asking in both orders cancels position bias.** A judge that always picks "1" scores
  exactly 0.5, which the dry run confirmed.
- **Relative judgements catch smaller differences than absolute scores.** The GNN-vs-mean gap
  (+0.11 on a 1–10 scale) is the kind of difference this is for.

### No judge

| column | how |
|---|---|
| `tfidf_rank` | TF-IDF rank of the paper's own gold among all golds in the file (as in `src/adapter/identify.py`). Top-1 = share with rank 1. |
| `ontarget_gold_sim`, `ontarget_max_ref_sim`, `ontarget_margin` | TF-IDF similarity to the gold, the highest similarity to any of the paper's reference abstracts, and the difference. |

- **What the margin measures:** a negative margin means the generation is closer to one of its
  references than to its own paper. That is the "describes a cited work" failure, measured
  without a judge.
- **Why not R4:** R4 named this failure in the judge prompt and was rejected because it
  penalised the baseline across the board instead of on the copied papers.
- **Disclosure:** the soft-prompt arms never see reference text, so they cannot copy it. This
  metric measures a real failure, but a structural one.
- **Read it relative to other arms, not against zero.** It is mostly negative for every arm,
  because the maximum is taken over ~50 references. In the 20-row dry run on reader ← GNN the
  mean was −0.11.
- **Why TF-IDF and not Voyage:** embedding the generations with Voyage would cost money. TF-IDF
  is the project's existing identifiability instrument.

## Rules for reading the results (fixed before scoring)

1. **Report every metric.** Choose what goes in the thesis by the validity rule below, not by
   which arm wins.
2. **Validity rule: a metric is usable only if reader ← zero < reader ← mean ≤ reader ←
   oracle on it.** This needs the zero arm scored, at least on a pilot. A metric that fails is
   dropped, whichever arm it favours.
3. **Paired differences per paper, with SE**, and Holm correction across the metrics.
4. **Pilot on ~150 papers first** to check that labels are valid and scales are used, then the
   full 945.

## Cost compared with `evaluate_results_scores_only.py`

These are token estimates on the real joint and text-baseline generations, at characters ÷ 4,
priced at gpt-5.4-mini with output 4–8× the input price. Ratios are relative to one run of the
current evaluator on the same arm.

| | calls per paper | cost |
|---|---|---|
| current evaluator | 1 | 1× |
| detailed, per arm (A + B + C) | 4 | **~4.8×** |
| gold labels, once in total | +1 on the first arm only | +~1.2× once |
| pairwise, per pair of arms (D) | +2 | +~2× per pair |

- **Where the cost goes:** identification is the biggest part, since it reads six abstracts.
  The comparison call is next.
- **Total for 5 arms** (text baseline, joint, reader ← GNN, reader ← mean, reader ← oracle):
  5 × 4.8 + 1.2 ≈ **25 current-evaluator runs**.
- **Pairwise for 3 pairs** (GNN vs mean, joint vs GNN, joint vs text baseline): ≈ 6 more.
- **Adding reader ← zero** for the validity rule costs about 4.8 more.
- **ROUGE and BERTScore** are local and free, as before.

## Limitations

- **The judge still sees the gold in B and D**, so it can reward surface similarity. A and C
  limit this: A labels blind, and in C the gold is unmarked.
- **The label lists are a design choice.** "Field" with 14 classes and "method family" with 12
  are coarse on purpose, so that two judges would agree. They are not a standard taxonomy.
- **No noise floor is measured yet.** Scoring one arm twice would give it, at ~4.8× cost.
- **Not run on the API yet.** It was checked with a mocked judge on 20 papers: columns,
  clipping, distractor selection, resume and pairwise order-cancelling.

## Analysis notebook

`experiments/<arm>/analysis_detailed.ipynb` reads `evaluated_result_detailed.csv` in its own
folder. The first copy is in `2026-10-09-14541449_run11-reader-gnn`. It follows the layout of
`analysis.ipynb` (per-metric bar charts by neighbour-count bins of 5, weighted by
`num_of_neighbors`, and score distributions).

It has these sections:

- a summary table of every metric;
- ROUGE-L, BERTScore, coherence and language;
- **A:** facet match rates, and gold-vs-generated confusion tables for field, contribution
  and method family;
- **B:** subfield/problem/task on the 0–3 scale, and the claim counts and precision;
- **C:** identification rank, top-1 and MRR against chance (20% / 3.0 / 0.457);
- TF-IDF top-1 and the on-target margin;
- **D:** the pair score (skipped if the CSV has no `pair_score`);
- examples;
- **a paired comparison with another arm.** Set `OTHER` in the last cell (default: the
  refs-mean arm). It prints the per-paper difference, SE and z for every metric, and plots the
  relative differences with 95% CIs.

To use it for another arm, copy it and change `ARM` in the first cell, and `OTHER` /
`OTHER_NAME` in the last.
