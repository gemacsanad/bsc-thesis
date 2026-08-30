# Blind-judge comparisons

Each round asks an LLM judge to compare two generations of the same target paper against its
gold abstract, blind to the accuracy scores and to how the nodes were sampled. One folder per
round; each holds the sampler, the judge, the resulting CSV, and a report.

| round | compares | question |
|---|---|---|
| `01_baseline_vs_adapter` | no-graph baseline vs adapter, 2026-08-15 | what does the graph add? |
| `02_prompt_rewrite` | 2026-08-15 vs 2026-08-20 | why did the rewrite raise the floor and destroy the top? |
| `03_prompt_fix` | 2026-08-20 vs 2026-08-21 | did removing the two blamed clauses bring the top back? |

Round 3 also carries `ref_overlap.csv` and `ref_overlap_all945.csv`: overlap of each generation
with the gold and with the most similar cited reference, the basis for the copying analysis.

Rounds 2 and 3 use the **same 48 nodes**, so a paper can be traced across all three prompts.

Each folder's `prompt.md` records the exact generation prompt behind each version it judges;
the judge's own prompt lives in that folder's `compare_abstracts.py`.

```
python 03_prompt_fix/create_dataset.py     # builds comparison_dataset.csv
python 03_prompt_fix/compare_abstracts.py  # fills its `comparison` column (needs OPENAI_API_KEY)
```

The builders refuse to overwrite a CSV that already holds judge output unless passed `--force`.
