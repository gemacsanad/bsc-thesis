# DyTAG — graph-conditioned abstract generation

Generate a paper's abstract from its citation neighbourhood. A GraphSAGE encoder reads the
paper's reference neighbourhood, an adapter turns that into soft prefix tokens, and the prefix
is injected into a **frozen** Vicuna-7B. The prefix is the only trainable channel into the LLM.

```
neighbourhood → GraphSAGE → adapter → 10 prefix tokens → frozen Vicuna-7B → abstract
```

## Where things are

| Directory | What's in it |
|---|---|
| `src/` | All code. Nothing else — no data, no results, no notebooks. |
| `data/` | All datasets. Mostly gitignored; `data/frozen/` is not (see below). |
| `experiments/` | One directory per run: what produced it, what it generated, how it scored. |
| `notebooks/` | Exploration only — never a pipeline step. |
| `docs/` | The written record. Start at [`docs/README.md`](docs/README.md). |
| `checkpoints/` | Trained adapter weights (~196 MB/epoch). Gitignored. |
| `hpc/komondor/` | Singularity container definitions and SLURM job scripts. |
| `archive/` | Superseded work, kept for reference. Not part of the current pipeline. |

## The current experiment

The live comparison is two runs over the **same frozen 495-node evaluation set**, differing only
in whether the graph prefix is present:

| Run | Condition |
|---|---|
| `experiments/2026-08-04_adapter-ep2/` | GraphSAGE + prefix adapter + frozen Vicuna-7B |
| `experiments/2026-08-04_vicuna-baseline/` | The same frozen Vicuna-7B, no prefix |

Everything under `experiments/_archive/` is the **older generation**: it ran on a 200-node
ogbn-only test set that `docs/adapter.md` (Stage 1) records as superseded, and it is not
comparable with the two runs above. It is kept because the work was real, not because the
numbers still stand.

## Anatomy of an experiment

Every run directory holds the same five files, so any two runs can be compared without
remembering what each one called things:

```
experiments/<run-id>/
├── config.yaml        what produced this: checkpoint, decoding, data, judge
├── generations.csv    node_id, mag_id, original_abstract, generated_abstract
├── scored.csv         the above + ROUGE + BERTScore + the LLM-judge scores
├── analysis.ipynb     the plots and tables for this run
└── notes.md           what you concluded (current runs only)
```

`scored.csv` adds `rouge{1,2,L}_f`, `bertscore_{p,r,f1}` and the LLM-as-judge scores
`llm_accuracy`, `llm_coherence`, `llm_language` — each 1–10 from `gpt-5.4-mini`, each with a
free-text reason column.

Rows whose prompt overflowed the 4,096-token context are written with
`generated_abstract = "ERROR: ..."` and skipped by the evaluator. Both current runs carry the
same 63 error rows by construction, so they compare on the same 432 nodes.

## Running the pipeline

Local work uses the `DyTAG` conda env; GPU work runs on Komondor through the Singularity images
in `hpc/komondor/containers/`. Adapter scripts use flat imports, so run them **from inside
`src/adapter/`**.

```bash
# 1. datasets (once)
python src/data/download_ogbn.py                 # → data/raw/ogbn_arxiv/
sbatch hpc/komondor/jobs/run_download_dataset.sh # → data/processed/s2/
python src/data/build_joint.py                   # → data/processed/joint/

# 2. freeze which nodes each split may use  → data/frozen/
cd src/adapter && python prepare_data.py && python text.py

# 3. train, then generate  (Komondor)
sbatch hpc/komondor/jobs/run_train_adapter.sh    # → checkpoints/epoch*.pt
sbatch hpc/komondor/jobs/run_generate_eval.sh    # → experiments/<date>_*/generations.csv

# 4. score a run
python src/evaluation/evaluate_results.py \
    --input experiments/<run>/generations.csv \
    --output experiments/<run>/scored.csv
```

`data/frozen/` is deliberately versioned: `eval_nodes.json`, `splits_eligible.json` and
`training_sets.json` decide which nodes every experiment runs on, and a result cannot be
reproduced without them. Everything else under `data/` is large and regenerable.

Secrets live in a single `.env` at the repo root (`OPENAI_API_KEY`, `VOYAGE_API_KEY`,
`WANDB_KEY`, `S2_API_KEY`); it is gitignored.

## Known limitations worth carrying into any result

- **S2 coverage confound.** 44% of the 169,343 core nodes have no Semantic Scholar
  bibliography, so out-degree means different things for different nodes. Measured, not removed.
- **Context overflow is not random.** It concentrates at high out-degree — at out-degree 15 more
  than half of all candidates exceed 4,096 tokens — so surviving nodes skew toward terser
  neighbourhoods exactly where the graph has the most to say.

Both are quantified in `docs/adapter.md`.
