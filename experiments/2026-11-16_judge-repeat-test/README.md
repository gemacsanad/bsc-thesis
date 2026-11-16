# Judge repeat test (2026-11-16)

**Question.** How much do the LLM judge's scores vary when the same output is scored again?
Thesis Ch. 6 §6.4, Tab. `tab:judge-repeat`.

**Design.** 100 of the 945 run-11 GNN-estimate generations
(`experiments/2026-10-09-14541449_run11-reader-gnn/generations.csv`, seed 0), each scored 3 times
with `gpt-5.4-mini`, temperature 0:
- content: the unchanged `evaluate_detailed.compare` call (subfield, problem, task, 0–3);
- quality: the new `QUALITY_SYSTEM` call in `run.py` (coherence, language, anchored 0–3; sees
  only the generated abstract). This is the scale the thesis describes; the configurations and
  the metric check still have to be re-scored with it (`thesis/plans/06-evaluation.md`).

**Run** (from the repository root): `python experiments/2026-11-16_judge-repeat-test/run.py`
→ `scores.csv`, `summary.csv`. 300 scorings, 0 failures.

**Result.**

| aspect | repeat 1 | repeat 2 | repeat 3 | mean per-paper SD | identical in all 3 |
|---|---|---|---|---|---|
| subfield | 1.19 | 1.27 | 1.25 | 0.206 | 67% |
| problem | 1.31 | 1.34 | 1.31 | 0.191 | 67% |
| task | 1.44 | 1.47 | 1.43 | 0.162 | 72% |
| coherence | 2.84 | 2.83 | 2.85 | 0.012 | 98% |
| language | 2.48 | 2.50 | 2.48 | 0.040 | 93% |

Pairwise repeat differences: all within about 1.6 SE (largest subfield 2−1, +0.08, SE 0.049);
a changed score moves by more than one level for at most 2% of papers. Coherence is near the
ceiling (258 of 300 scores are 3).
