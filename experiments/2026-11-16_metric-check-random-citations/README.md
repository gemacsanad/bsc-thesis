# Metric check: real vs random citations (2026-11-16)

**Question.** Do ROUGE-L and BERTScore notice when a generated abstract is about the wrong
paper? This is a check of the *metrics*, not of the thesis model, so the generator is an
off-the-shelf API model with no training (thesis Ch. 6, §6.2). It is not a baseline and its
scores are never compared with the model's.

**Design.**
- 200 of the 945 evaluation papers (`data/frozen/eval_nodes_capped.json`), sampled with seed 0.
- Two conditions with an identical prompt; only the listed publications differ:
  - `real` — titles and abstracts of all the paper's recorded references (joint graph);
  - `random` — the same number of papers drawn at random (seed 0) from the joint graph,
    excluding the paper and its real references.
- No reference cap: every reference is shown (median 48 per paper, max 99; prompts median
  ≈ 8,500 words).
- Generator `gpt-5.6-luna`, `reasoning_effort=low`, `max_completion_tokens=2048`; temperature
  not sent (the 5.6 models reject it). Prompt in `build_inputs.py` (`INSTRUCTION`). The
  generator is deliberately a different model from the judge (author), so the judge does not
  score its own text.
- Scoring against the original abstract (`score.py`): ROUGE-L F1 and BERTScore F1 (same
  settings as `src/evaluation/evaluate_detailed.py`), TF-IDF identification within the 200
  papers, and the judge's content-agreement scores (subfield, problem, task, 0–3;
  `evaluate_detailed.compare`, `gpt-5.4-mini`).

**Run** (from the repository root):

    python experiments/2026-11-16_metric-check-random-citations/build_inputs.py
    python experiments/2026-11-16_metric-check-random-citations/generate.py
    python experiments/2026-11-16_metric-check-random-citations/score.py

**Outputs.** `test_set.csv` (prompts), `real/` and `random/` (`generations.csv`, `scored.csv`),
`summary.csv` (paired real − random per metric), `generate.log`.

**Result** (200 papers × 3 conditions, 0 generation errors; `summary.csv`, notebook
`notebooks/metric_check.ipynb`).
`same_category` was added on 2026-11-16 after the first two conditions; it uses a separate seed, so
the `random` prompts are byte-identical to the first run (checked).

| metric | real | same category | random | AUC real vs same cat. | AUC real vs random |
|---|---|---|---|---|---|
| ROUGE-L F1 | 0.153 | 0.112 (73%) | 0.112 (73%) | 0.932 | 0.936 |
| BERTScore F1 | 0.841 | 0.822 (98%) | 0.820 (97%) | 0.920 | 0.945 |
| judge content agreement (0–3) | 2.03 | 0.25 (12%) | 0.07 (3%) | 0.959 | 0.982 |
| TF-IDF top-1 (of 200) | 76.0% | 2.5% | 0.5% | — | — |

Reading: ROUGE-L and BERTScore separate real from control citations (AUC 0.92–0.95) but on a
compressed scale, and they give the same-category and the random controls practically the same
score (0.112 vs 0.112; 0.822 vs 0.820): they do not register that one control is in the paper's
field. The judge registers it (0.25 vs 0.07; subfield 0.18 vs 0.03) and has an interpretable
scale. Caveat: an arXiv category is broad, so the same-category control is only slightly closer
to the paper (judge subfield 0.18 of 3).
