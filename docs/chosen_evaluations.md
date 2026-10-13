# Chosen evaluations for the thesis ablation table

Written 2026-10-10. This file picks which experiment folders go into the final ablation table
and lists what still needs judging. All rows use the same 945 eval papers, Qwen3-14B and the
same judge. Paired differences are per paper, with SE = sd / √945.

## The table

| # | role | folder (`experiments/`) | input to the LLM | judged? | accuracy | ≥3 | coherence |
|---|---|---|---|---|---|---|---|
| 1 | **Text baseline** | `2026-09-10-14364743_qwen14b-baseline` | reference abstracts as text, no prefix | ✅ | 3.59 | — | 7.90 |
| 2 | **End-to-end (full pipeline)** | `2026-10-09-14541449_run11-joint` | GNN → prefix, GNN + adapter + LoRA fine-tuned jointly | ✅ | 2.94 | 59.7% | 7.86 |
| 3 | Pieces trained separately | `2026-10-09-14541449_run11-reader-gnn` | GNN estimate → prefix, no joint stage | ✅ | 2.98 | 61.2% | 7.83 |
| 4 | No-learning graph baseline | `2026-10-09-14541449_run11-reader-refs-mean` | plain mean of reference embeddings → prefix | ✅ | 2.86 | 55.3% | 7.89 |
| 5 | Ceiling | `2026-10-09-14541449_run11-reader-oracle` | the paper's own embedding → prefix | ❌ **judge this** | — | — | — |
| 6 | Floor | `2026-10-09-14541449_run11-reader-zero` | zero vector → prefix | not needed | TF-IDF top-1 0.1% | | |
| 7 | Decoding check | `run11-looptest-ngram0` / `run11-looptest-ngram6` | oracle, 200 papers, without / with the n-gram ban | not needed | loops 14% → 0%, top-1 52.0% → 51.5% | | |

**Still to run:** only the judge on row 5. Row 6 is reported with TF-IDF identifiability: at
0.1% top-1 the reader writes nothing paper-specific, and a judge score adds nothing to that. Row
7 is a decoding decision, measured by loop rate and top-1, not by the judge.

## What each comparison answers

| comparison | question | result |
|---|---|---|
| 3 − 4 | Does the GNN beat the plain mean? | **+0.113 (SE 0.037)**, about 3 SE. TF-IDF top-1 18.5% vs 16.5%. |
| 2 − 3 | Does joint fine-tuning add anything? | −0.032 (SE 0.038): no. The GNN did not move in the joint stage (`cos_loss` flat at 0.20), so this tests reader adaptation only. |
| 2 − 4 | The full pipeline against the no-learning baseline | +0.081 (SE 0.038). |
| 1 − 2 | What one 32-token vector loses against thousands of tokens of text | −0.649 (SE 0.047). The end-to-end arm gets 82% of the baseline's accuracy. |
| 5 − 3 | How much is lost to the estimate rather than the reader | pending row 5 |
| 6 | Is the gain from the vector, not from LoRA? | yes: 0.1% top-1 without it |

## Why these and not others

- **The text baseline is the Qwen3-14B one (row 1)**, not Vicuna, because every graph arm uses
  Qwen3-14B.
- **Run 11 replaces run 10a.** Run 10a's arms looped (17.6% of oracle outputs), which depressed
  their scores. Run 11 has the same recipe with the loop fix, so its rows are the clean version.
- **Earlier soft-prompt runs (stages 7–8, runs 7–8) stay out of the table.** They scored
  1.3–1.5 and failed for reasons later fixed. They belong in the "how we got here" chapter, not
  in the ablation.
- **Row 2 is the only end-to-end run there is.** If a real end-to-end run is done (GNN lr ~1e-4,
  starting from the InfoNCE GNN, ~13 h; see `run9_embedding_decoder.md` §9), it replaces row 2,
  and row 2 moves to a footnote as "joint stage with a frozen-in-practice GNN".
