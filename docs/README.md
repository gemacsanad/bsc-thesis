# Documentation index

The written record of the project. Each file is append-only: a new section per piece of work,
rather than a rewrite of what came before.

| File | Covers |
|---|---|
| [`adapter.md`](adapter.md) | Stages 1–4 of the current model: eligible nodes and the frozen eval set, the graph tensors and loaders, the GraphSAGE encoder, the prefix adapter, prefix injection into a frozen Vicuna, training, W&B telemetry, the container and the training job. |
| [`adapter_evaluation.md`](adapter_evaluation.md) | Stage 5: generating the two evaluation CSVs, and what is held fixed between the adapter and baseline conditions. |
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
