# Migrating the Komondor project to the new layout

**Date:** 2026-08-05

Rebuild `/project/c_gnn42/csanad_thesis/baseline` on the same structure as the local repository:
delete everything that also exists locally and re-upload it, while preserving in place the things
that exist **only** on the HPC.

An earlier draft of this plan rearranged everything with `mv` to avoid the transfer entirely.
That was dropped in favour of the simpler approach below — ~3 GB over a fast link is minutes, and
a clean re-upload leaves no chance of a stale file surviving in a directory nobody thought to
check. The `mv` trick is still used, but only for the handful of things that cannot be re-uploaded.

## What is re-uploaded, and what is preserved

Measured on the local repository:

| Re-uploaded | size |
|---|---|
| `data/processed/joint/` | 2.0 GB |
| `data/raw/ogbn_arxiv/` | 673 MB |
| `data/processed/s2/` (the two filtered parquets) | 238 MB |
| `data/processed/ogbn_subset/` | 10 MB |
| code, docs, notebooks, experiments | ~22 MB |
| `archive/graphsage_classification/*.pt` (optional) | 187 MB |
| **total** | **~3.1 GB** |

| Preserved on the HPC — cannot be re-uploaded | why |
|---|---|
| `.hf_cache/` — Vicuna-7B-v1.5, ~13 GB | not on the laptop; compute nodes have no route to HuggingFace |
| `checkpoints/epoch{1,2}.pt` | not on the laptop, not in git — a 12 h GPU job to reproduce |
| `adapter-gpu.sif`, `baseline-gpu.sif`, `graphsage-gpu.sif` | rebuilding needs a login node, internet and time |
| `data/raw/s2/` — 480 shards | days of API download |
| `data/processed/s2/papers.parquet`, `references.parquet` | the **unfiltered** dumps; only the filtered pair is local |
| W&B run directories | raw record of the runs behind the current results |
| `logs/train_adapter_14152948.*`, `logs/generate_eval_14156862.*` | the only measured A100 timings in the project |
| `full_graph_embeddings.pt`, `full_graph_weights.pt`, `full_graph_training_curves.png` | absent from the local repo |

## The safety property

The deletion in Phase 2 is a **whitelist**, not a list of things to remove. Everything worth
keeping is moved into a single `_keep/` directory first; the delete loop then removes every
top-level entry *except* `_keep`. This means no `rm -rf` is ever typed next to a path that must
survive, and nothing can be missed by forgetting to list it.

## Phase 0 — pre-flight

```bash
cd /project/c_gnn42/csanad_thesis/baseline
squeue -u $USER                                   # nothing of yours may be running
du -sh .hf_cache adapter/checkpoints s2_data/raw
find s2_data/raw -name '*.jsonl.gz' | wc -l       # expect 480: 30 + 390 + 60
md5sum adapter/checkpoints/epoch2.pt | tee ~/epoch2.md5
```

Keep `~/epoch2.md5` outside the project directory — Phase 5 verifies against it.

## Phase 1 — stage everything that must survive

```bash
cd /project/c_gnn42/csanad_thesis/baseline
mkdir -p _keep/{sif,checkpoints,s2_processed,wandb,graphsage,logs,env}

# the model cache, moved whole and untouched
mv .hf_cache                       _keep/hf_cache

# containers
mv adapter-gpu.sif baseline-gpu.sif        _keep/sif/
mv Komondor_files/graphsage-gpu.sif        _keep/sif/

# the trained adapter
mv adapter/checkpoints/epoch1.pt adapter/checkpoints/epoch2.pt  _keep/checkpoints/

# Semantic Scholar: all raw shards, and the two parquets that are NOT local
mv s2_data/raw                              _keep/s2_raw
mv s2_data/processed/papers.parquet         _keep/s2_processed/
mv s2_data/processed/references.parquet     _keep/s2_processed/

# W&B records
mv adapter/wandb                            _keep/wandb/adapter
mv GraphSAGE_classification/wandb           _keep/wandb/graphsage

# GraphSAGE artifacts absent from the local repo
mv GraphSAGE_classification/full_graph_embeddings.pt \
   GraphSAGE_classification/full_graph_weights.pt \
   GraphSAGE_classification/full_graph_training_curves.png  _keep/graphsage/

# the two job logs worth keeping
mv logs/train_adapter_14152948.out logs/train_adapter_14152948.err  _keep/logs/
mv logs/generate_eval_14156862.out logs/generate_eval_14156862.err  _keep/logs/

# belt and braces — the local .env files carry the same keys, so this is a fallback only
cp adapter/.env _keep/env/adapter.env 2>/dev/null || true
cp Download_script/.env _keep/env/s2.env 2>/dev/null || true
```

Check the staging before deleting anything:

```bash
du -sh _keep/*
find _keep/s2_raw -name '*.jsonl.gz' | wc -l      # must still be 480
ls _keep/checkpoints _keep/sif
```

## Phase 2 — delete everything else

```bash
cd /project/c_gnn42/csanad_thesis/baseline
shopt -s dotglob nullglob        # include dotfiles; '*' still never matches . or ..
for entry in *; do
    [ "$entry" = "_keep" ] && continue
    echo "removing $entry"
    rm -rf -- "$entry"
done
shopt -u dotglob nullglob
ls -A                            # only _keep should remain
```

`dotglob` is what makes this catch `.venv_s2_download`, `.DS_Store` and the old `.env` files;
without it they would survive and the re-upload would land on top of stale dotfiles. `*` never
expands to `.` or `..`, so the loop cannot walk up out of the directory.

Verified on a mock tree containing `_keep`, four dotfiles and five directories: all nine
non-`_keep` entries were removed and `_keep`'s contents were untouched.

Read the `removing …` lines as they scroll. If `_keep` ever appears there, stop immediately.

## Phase 3 — upload the new structure

From the **laptop**. No `--delete` — there is nothing left to delete, and the flag would only
create risk if the phases were run out of order.

```bash
cd ~/egyetem/DyTAG/baseline
rsync -avh --progress \
      --exclude '.git' \
      --exclude 'checkpoints' \
      --exclude '__pycache__' \
      --exclude '.DS_Store' \
      --exclude '.ipynb_checkpoints' \
      ./ vn01:/project/c_gnn42/csanad_thesis/baseline/
```

This carries `src/`, `docs/`, `notebooks/`, `experiments/`, `archive/`, `hpc/`, `data/`,
`README.md`, `.gitignore`, and both `.env` files — the root one with all four API keys, and
`src/data/s2/.env` with `S2_API_KEY`. `checkpoints/` is excluded because the real ones come back
in Phase 4.

Two optional trims, if you would rather not send them:

```bash
--exclude 'data/processed/s2'                 # saves 238 MB; identical copies are in _keep
--exclude 'archive/graphsage_classification/*.pt'   # saves 187 MB; only needed to re-run GraphSAGE
```

## Phase 4 — put the preserved artifacts into the new layout

```bash
cd /project/c_gnn42/csanad_thesis/baseline
mkdir -p data/raw/s2 data/processed/s2 checkpoints logs archive/graphsage_classification

mv _keep/hf_cache                  .hf_cache
mv _keep/sif/adapter-gpu.sif  _keep/sif/baseline-gpu.sif  .
mv _keep/sif/graphsage-gpu.sif     hpc/komondor/graphsage-gpu.sif
mv _keep/checkpoints/epoch1.pt _keep/checkpoints/epoch2.pt  checkpoints/
mv _keep/s2_raw/*                  data/raw/s2/
mv _keep/s2_processed/*.parquet    data/processed/s2/
mv _keep/graphsage/*               archive/graphsage_classification/
mv _keep/logs/*                    logs/
mkdir -p logs/wandb && mv _keep/wandb/* logs/wandb/

# only if the root .env did not come across for some reason
[ -s .env ] || cat _keep/env/*.env > .env

rmdir _keep/sif _keep/checkpoints _keep/s2_raw _keep/s2_processed _keep/graphsage _keep/logs _keep/wandb
rm -rf _keep
```

## Phase 5 — verify

```bash
cd /project/c_gnn42/csanad_thesis/baseline
md5sum -c ~/epoch2.md5                              # adapter intact
find data/raw/s2 -name '*.jsonl.gz' | wc -l         # still 480
ls data/processed/s2/                               # four parquets
ls .hf_cache/hub/models--lmsys--vicuna-7b-v1.5/snapshots/*/
du -sh data/processed/joint                         # ~2 GB
ls -lh adapter-gpu.sif baseline-gpu.sif hpc/komondor/graphsage-gpu.sif checkpoints/

# the real test: the pipeline loads its data from the new paths
module load singularity
singularity exec -B $PWD:/app adapter-gpu.sif bash -lc 'cd /app/src/adapter && python graph.py'
```

`graph.py` reads `data/processed/joint/` and `data/frozen/`, and asserts edge direction, absence
of self-loops and seed-row placement. **384,756 nodes, 3,844,131 edges and exit 0** means every
path in the new layout resolves. Anything else means stop and read the error before running a job.

## Phase 6 — optional cleanup

```bash
rm -rf .hf_cache/xet                                          # staging + logs, junk
rm -rf .hf_cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct   # LICENSE + README only, no weights
```

## What had to be fixed to make this work

**The S2 job scripts pointed at the wrong directories.** When the repository was reorganised,
`s2_data/` was rewritten to `data/processed/s2/` everywhere, which produced
`data/processed/s2/raw/` and `data/processed/s2/processed/` — the S2 tree has its own raw and
processed split that does not collapse into one directory. Harmless while the S2 data was going
to be discarded; wrong as soon as it is kept.

`run_download_dataset.sh`, `run_filter_ogbn_references.sh` and `run_add_titles_years.sh` now use
three explicit variables — `RAW_DIR=data/raw/s2`, `PROC_DIR=data/processed/s2`,
`TMP_DIR=duckdb_tmp` — instead of one `DATA_DIR` with a nested split. `download_shards.py` writes
to `<out-dir>/<dataset>/` rather than `<out-dir>/raw/<dataset>/`, so the caller controls the whole
path. All three target directories were already in `.gitignore`.

## Open item

`wandb/` is not in `.gitignore`, but `train.py --wandb` creates one wherever it runs — which,
after this migration, is `src/adapter/wandb/` inside the repository. Add it before the next
training run, or the run's telemetry lands in `git status`.
