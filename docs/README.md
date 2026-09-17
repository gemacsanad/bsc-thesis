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
