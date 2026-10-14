# Documentation index

The written record of the project. Each file is append-only: a new section per piece of work,
rather than a rewrite of what came before.

| File | Covers |
|---|---|
| [`project_overview.md`](project_overview.md) | The non-specialist introduction: the research question, the data, the method in plain terms, the milestone timeline, and what has been found. Start here if you are not going to read the code. |
| [`adapter.md`](adapter.md) | Stages 1–4 of the current model: eligible nodes and the frozen eval set, the graph tensors and loaders, the GraphSAGE encoder, the prefix adapter, prefix injection into a frozen Vicuna, training, W&B telemetry, the container and the training job. |
| [`adapter_evaluation.md`](adapter_evaluation.md) | Stage 5: generating the two evaluation CSVs, and what is held fixed between the adapter and baseline conditions. |
| [`hpc_migration.md`](hpc_migration.md) | Moving the Komondor project onto this layout in place, keeping the Vicuna cache, the trained adapter and the S2 download. |
| [`model_swap_qwen.md`](model_swap_qwen.md) | Stage 7, **the plan**: swapping the frozen Vicuna-7B for Qwen3-14B, what must not change, and why. |
| [`qwen_migration.md`](qwen_migration.md) | Stage 7, **the record**: what was actually built at each step, with the measurements. |
| [`soft_prompt_only.md`](soft_prompt_only.md) | Stage 8: removing the reference abstracts from the prompt so the graph prefix is the only channel — why (the Qwen gap went negative), the decisions, and the measurements. |
| [`run9_embedding_decoder.md`](run9_embedding_decoder.md) | Run 9: the review of all soft-prompt runs, the embedding → abstract decoder it motivates, what changed and why, how to read the four evaluation arms, run 9's epoch-1 stop, run 10a (step 4's recipe on 90k papers, trained half on the reference mean), and run 11 (reader, GNN imitation grid, joint fine-tune, loop fix). |
| [`run11_model.md`](run11_model.md) | The run-11 model as one description: each component (masked GATv2 estimator, prefix adapter, LoRA on a frozen Qwen3-14B, decoding), the three training phases with their W&B curves, the 945-paper results and limitations. |
| [`other_datasets.md`](other_datasets.md) | Candidate non-citation datasets for the same pipeline (Goodreads, Amazon co-purchase, Wikipedia), what rules others out, and the CPU check to run before any GPU time. |
| [`chosen_evaluations.md`](chosen_evaluations.md) | Which experiment folders make up the thesis ablation table (text baseline, end-to-end, run-11 arms), the paired comparisons they answer, and what is still to judge. |
| [`detailed_evaluation.md`](detailed_evaluation.md) | `src/evaluation/evaluate_detailed.py`: the accuracy score split into blind facet labels, subfield/problem/task, claim precision, identification among hard distractors, pairwise preference and on-target margin; how each works, the reading rules and the cost. |
| [`thesis_context/`](thesis_context/00_index.md) | Ten self-contained, sourced context files for writing the thesis: literature with verified URLs, data and preprocessing, the final model, development history, evaluation methods (incl. planned validity experiments), results (not final), proposed other datasets, reproducibility, limitations. Start at `00_index.md`. |
| [`weekly/week_4.md`](weekly/week_4.md) | The reference-filtering pipeline — LLM relevance labels, Voyage embeddings, and the two cosine scores. Old generation. |
| `../src/data/s2/README.md` | The Semantic Scholar bulk download and the joint dataset build. |

## Reading order

If you are picking the project up cold: the root [`README.md`](../README.md) for the map, then
`adapter.md` Stage 1 for why the evaluation set is what it is, then `adapter_evaluation.md` for
how the current numbers were produced.

## The three findings that constrain everything downstream

- **S2 coverage** (`adapter.md`, Stage 1) — 56% of core nodes have a real bibliography, 44% only
  their arXiv-CS citations. Out-degree is therefore not comparable across nodes.
- **Prefix injection is silent when wrong** (`adapter.md`, Stage 4 step 1) — a wrong
  `cache_position` produces fluent text and a falling loss, not an error. This is why
  `verify_generation.py` runs before the GPU hours on every evaluation job.
- **The gate bootstraps from zero** (`adapter.md`, Stage 3b addendum) — at `gate = 0` the entire
  13.7M-parameter graph pathway receives exactly zero gradient, so it has to learn its way open
  from a random delta.

## The `final_model` branch cleanup (2026-10-13)

The `final_model` branch starts from `be93232` and removes only code that no experiment or job
depends on. **Every folder under `experiments/` is kept**, including `_archive/`: runs that look
superseded still back arguments in the thesis (for example `_archive/random-sample` is the
evidence that BERTScore cannot separate the arms).

Removed, all recoverable with `git show be93232:<path>`:

| path | why it went |
|---|---|
| `src/analysis/probe.py` | TF-IDF identifiability; the same instrument now lives in `src/adapter/identify.py` and in `src/evaluation/evaluate_detailed.py` (`tfidf_rank`) |
| `src/analysis/refsignal.py` | the 64.34% reference-mean retrieval; superseded by the run-11 GNN grid's `refs_mean` baseline and `retrieval_baseline.py` |
| `src/analysis/memo.py`, `selpolicy.py`, `domain_probe.py` | step-6-era one-off analyses; their results are in `soft_prompt_only.md` and `step6_failure_analysis.md` |
| `src/evaluation/evaluate_results_simple.py` | never used by any experiment |
| `archive/graphsage_classification/`, `hpc/komondor/jobs/run_train_graphsage_full.sh`, `hpc/komondor/containers/graphsage-gpu.def` | the ogbn-arxiv node-classification side project; it produced no experiment in this repo |

`.env_local` is no longer tracked and is now in `.gitignore`; the file itself stays on disk. It
holds real API keys and is still present in every earlier commit, so **rotate those keys before
this repository is pushed anywhere**.
