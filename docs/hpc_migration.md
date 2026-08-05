# Migrating the Komondor project to the new layout

**Date:** 2026-08-05

Bring `/project/c_gnn42/csanad_thesis/baseline` onto the same structure as the local repository,
keeping the Vicuna checkpoint, the trained adapter, every dataset **and** the Semantic Scholar
bulk download.

## The governing principle: move, never copy

Every `mv` below stays inside `/project`, so it is a **rename** — instant, regardless of file
size, and it never touches the network. The alternative that first suggested itself, "delete the
folder and re-upload the organised one", would have cost:

| | size | why re-uploading is wrong |
|---|---|---|
| `.hf_cache/` (Vicuna-7B) | ~13 GB | not on the laptop at all; compute nodes have no route to HuggingFace |
| `checkpoints/epoch{1,2}.pt` | ~400 MB | not on the laptop, not in git — a 12 h GPU job to reproduce |
| `*.sif` containers | GBs | rebuilding needs a login node, internet and time |
| `data/raw/s2/` | ~480 shards | days of API download |
| `data/processed/joint/` | 2 GB | identical copy exists locally, but uploading it is pointless |

Only the code moves over the network — about 15 MB.

## Target layout

Identical to the local repository, plus four things that exist only on the HPC (all gitignored,
so they never conflict with what is pushed):

```
baseline/
├── src/  docs/  notebooks/  experiments/  archive/  hpc/    ← pushed from git
├── data/
│   ├── frozen/                       ← from git (eval_nodes, splits_eligible, training_sets)
│   ├── raw/ogbn_arxiv/               ← mv  ogbn_arxiv_dataset/
│   ├── raw/s2/{abstracts,citations,papers}/   ← mv  s2_data/raw/*
│   ├── processed/ogbn_subset/        ← mv  the *_subset files
│   ├── processed/joint/              ← mv  joint_dataset/
│   └── processed/s2/*.parquet        ← mv  s2_data/processed/
├── checkpoints/epoch{1,2}.pt         ← mv  adapter/checkpoints/
├── duckdb_tmp/                       ← mv  s2_data/duckdb_tmp/
├── .hf_cache/                        ← UNTOUCHED, stays exactly where it is
├── adapter-gpu.sif  baseline-gpu.sif ← UNTOUCHED, job scripts expect them at the root
├── hpc/komondor/graphsage-gpu.sif    ← mv  Komondor_files/graphsage-gpu.sif
├── logs/                             ← keep the two current runs, archive the rest
├── .env                              ← merged from adapter/.env + Download_script/.env
└── src/data/s2/.env                  ← S2_API_KEY, read by s2_common.py
```

## Phase 0 — pre-flight

Nothing destructive. Establish what you have so you can prove afterwards that nothing vanished.

```bash
cd /project/c_gnn42/csanad_thesis/baseline
squeue -u $USER                       # nothing of yours may be running
du -sh .hf_cache adapter/checkpoints *.sif s2_data joint_dataset ogbn_arxiv_dataset
find s2_data/raw -name '*.jsonl.gz' | wc -l     # expect 480: 30 + 390 + 60
md5sum adapter/checkpoints/epoch2.pt | tee /tmp/epoch2.md5
```

Record those numbers. Phase 6 checks against them.

## Phase 1 — the skeleton

```bash
mkdir -p data/raw/ogbn_arxiv data/raw/s2 \
         data/processed/ogbn_subset data/processed/joint data/processed/s2 \
         checkpoints logs_keep
```

## Phase 2 — the irreplaceable artifacts

```bash
mv adapter/checkpoints/epoch1.pt adapter/checkpoints/epoch2.pt checkpoints/
mv Komondor_files/graphsage-gpu.sif hpc/komondor/graphsage-gpu.sif   # after Phase 5 creates hpc/

# W&B records of the runs that produced the current results
mkdir -p logs_keep/wandb
mv adapter/wandb                logs_keep/wandb/adapter
mv GraphSAGE_classification/wandb logs_keep/wandb/graphsage

# the only measured A100 timings anywhere in the project
mv logs/train_adapter_14152948.out logs/train_adapter_14152948.err logs_keep/
mv logs/generate_eval_14156862.out logs/generate_eval_14156862.err logs_keep/

# GraphSAGE artifacts that exist ONLY here — absent from the local repo
mkdir -p archive/graphsage_classification
mv GraphSAGE_classification/full_graph_embeddings.pt \
   GraphSAGE_classification/full_graph_weights.pt \
   GraphSAGE_classification/full_graph_training_curves.png archive/graphsage_classification/
```

`.hf_cache/`, `adapter-gpu.sif` and `baseline-gpu.sif` are **not touched** — they are already at
the paths the new job scripts expect (`HF_HOME=/app/.hf_cache`, `SIF=$BASE_DIR/adapter-gpu.sif`).

## Phase 3 — the datasets

```bash
# joint dataset
mv joint_dataset/nodes.csv joint_dataset/edges.csv joint_dataset/splits.json \
   joint_dataset/node_embeddings.npy joint_dataset/node_embeddings.progress.json \
   data/processed/joint/

# ogbn-arxiv: raw download vs the derived 1,936-node subset
mv ogbn_arxiv_dataset/nodes.csv ogbn_arxiv_dataset/edges.csv \
   ogbn_arxiv_dataset/node_features.npy ogbn_arxiv_dataset/splits.json \
   ogbn_arxiv_dataset/_titleabs.tsv ogbn_arxiv_dataset/_ogb_cache  data/raw/ogbn_arxiv/
mv ogbn_arxiv_dataset/nodes_subset.csv ogbn_arxiv_dataset/edges_subset.csv \
   data/processed/ogbn_subset/

# Semantic Scholar — kept in full
mv s2_data/raw/abstracts s2_data/raw/citations s2_data/raw/papers  data/raw/s2/
mv s2_data/processed/*.parquet                                     data/processed/s2/
mv s2_data/duckdb_tmp                                              duckdb_tmp
```

## Phase 4 — secrets

`.env` files are gitignored, so the upload in Phase 5 will not create them. Do this **before**
deleting the old directories, or the keys go with them.

```bash
cat adapter/.env Download_script/.env > .env      # WANDB_KEY + S2_API_KEY
# then add OPENAI_API_KEY and VOYAGE_API_KEY by hand if any HPC job ever needs them
grep -c . .env                                     # sanity: non-empty
```

`src/data/s2/.env` is created in Phase 5 by the upload (it exists locally and rsync carries it
if you do not exclude it — see the note there).

## Phase 5 — push the code

From the **laptop**, not the HPC. Never pass `--delete`: it would erase every directory that
exists only on the HPC, including `.hf_cache`.

```bash
cd ~/egyetem/DyTAG/baseline
rsync -av \
      --exclude '.git' --exclude 'data' --exclude 'checkpoints' \
      --exclude '*.sif' --exclude '__pycache__' --exclude '.DS_Store' \
      ./ vn01:/project/c_gnn42/csanad_thesis/baseline/
```

This carries `src/`, `docs/`, `notebooks/`, `experiments/`, `archive/`, `hpc/`, `README.md`,
`.gitignore` and — because it is not excluded — `src/data/s2/.env` with the S2 key. It does not
touch `data/`, the checkpoints, the containers or the model cache.

## Phase 6 — verify before deleting anything

```bash
cd /project/c_gnn42/csanad_thesis/baseline
md5sum -c /tmp/epoch2.md5                          # the adapter survived intact
find data/raw/s2 -name '*.jsonl.gz' | wc -l        # still 480
ls .hf_cache/hub/models--lmsys--vicuna-7b-v1.5/snapshots/*/   # weights still linked
du -sh data/processed/joint                        # ~2 GB

# the real test: the pipeline loads its data from the new paths
module load singularity
singularity exec -B $PWD:/app adapter-gpu.sif bash -lc 'cd /app/src/adapter && python graph.py'
```

`graph.py` reads `data/processed/joint/` and `data/frozen/`, and asserts edge direction, no
self-loops and seed-row placement. If it prints 384,756 nodes and 3,844,131 edges and exits 0,
every path in the new layout resolves.

## Phase 7 — delete, only now

```bash
rm -rf adapter cosine_filter Download_script GraphSAGE_classification random_sample
rm -rf ogbn_arxiv_dataset joint_dataset s2_data Komondor_files __pycache__ .venv_s2_download
rm -f  dataset_download.py generate_test_csv.py EDA.ipynb .DS_Store
rm -rf .hf_cache/xet .hf_cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
rm -rf logs && mv logs_keep logs
```

Every code file above is in git. `.venv_s2_download` is a virtualenv, rebuilt in one line by
`run_download_dataset.sh`. The Llama-3.1 cache entry holds only a LICENSE and a README — the
download never produced weights.

## What had to be fixed to make this work

**The S2 job scripts pointed at the wrong directories.** When the repository was reorganised,
`s2_data/` was rewritten to `data/processed/s2/` everywhere, which produced
`data/processed/s2/raw/` and `data/processed/s2/processed/` — the S2 tree has its own raw and
processed split that does not collapse into one directory. Harmless while the S2 data was being
discarded; wrong as soon as it is kept.

`run_download_dataset.sh`, `run_filter_ogbn_references.sh` and `run_add_titles_years.sh` now use
three explicit variables — `RAW_DIR=data/raw/s2`, `PROC_DIR=data/processed/s2`,
`TMP_DIR=duckdb_tmp` — instead of one `DATA_DIR` with a nested split. `download_shards.py`
writes to `<out-dir>/<dataset>/` rather than `<out-dir>/raw/<dataset>/`, so the caller controls
the whole path. All three target directories were already in `.gitignore`.

## Open item

`wandb/` is not in `.gitignore`, but `train.py --wandb` creates one wherever it runs — which,
after this migration, is `src/adapter/wandb/` inside the repository. Add it before the next
training run, or the run's telemetry lands in `git status`.
