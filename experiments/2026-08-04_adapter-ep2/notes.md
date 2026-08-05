# 2026-08-04 — adapter, epoch 2

First trained adapter evaluated on the frozen 495-node set. Paired against
[`../2026-08-04_vicuna-baseline`](../2026-08-04_vicuna-baseline), which is the same model,
prompts, order and decoding with the prefix removed.

## Result

Means over the 432 nodes scorable in **both** runs (63 rows overflow the context in each and are
excluded by construction):

| metric | adapter | baseline | delta |
|---|---|---|---|
| rouge1_f | 0.3221 | 0.3263 | −0.0042 |
| rouge2_f | 0.0606 | 0.0540 | **+0.0065** |
| rougeL_f | 0.1747 | 0.1614 | **+0.0133** |
| bertscore_f1 | 0.8386 | 0.8405 | −0.0019 |
| llm_accuracy | 2.82 | 2.46 | **+0.36** |
| llm_coherence | 7.03 | 6.32 | **+0.70** |
| llm_language | 7.17 | 7.28 | −0.11 |

The prefix helps most on the judge's coherence score and on the longer n-gram overlaps
(ROUGE-2, ROUGE-L), and is flat-to-slightly-negative on unigram overlap, BERTScore and language
quality. `llm_accuracy` is low in absolute terms for both conditions — 2.8 and 2.5 out of 10 —
which says the task itself is far from solved, not that the prefix failed.

## Open questions

- **No significance test yet.** These are means over 432 paired nodes; the deltas are small
  relative to the 1–10 judge scale and nothing here establishes they are more than noise. A
  paired test on the per-node differences is the next step.
- **One checkpoint, one seed.** Epoch 2 of a 4,000-node training subset. Whether the gate opens
  reliably under a different seed is untested — see the bootstrap risk in `docs/adapter.md`.
- **The 63 excluded nodes are the high-degree ones**, i.e. exactly where the graph should have
  the most to add. The comparison is honest but it is not a comparison over the whole set.

## Gotcha when analysing

`llm_accuracy`, `llm_coherence` and `llm_language` load as **object**, not numeric — a few rows
carry non-numeric judge output. Use `pd.to_numeric(col, errors="coerce")` before aggregating, or
a plain `.mean()` will concatenate strings instead of raising.
