# Implementation, infrastructure and reproducibility

Sources: repository tree on branch `final_model` (2026-10-13), `README.md`, `docs/README.md`,
`docs/hpc_migration.md`, `docs/adapter.md` (stages 4d–4e, 1f), `docs/qwen_migration.md`,
`hpc/komondor/`, script docstrings, `.gitignore`.

---

## 1. Repository layout

| directory | contents |
|---|---|
| `src/data/` | ogbn-arxiv download (`download_ogbn.py`), Semantic Scholar bulk download and joins (`s2/`), the joint-dataset build (`build_joint.py`) |
| `src/adapter/` | the model and its pipeline: config, graph tensors, models, training, generation, preparation of node sets, verification scripts |
| `src/evaluation/` | judge + ROUGE + BERTScore scorers, the detailed evaluator |
| `src/analysis/` | `retrieval_baseline.py` (the no-LLM floor) |
| `src/filters/`, `src/generation/` | the early reference-filtering pipeline and text-only generation scripts (archived 200-node experiments) |
| `data/frozen/` | **versioned** node sets every experiment uses (eval, validation, training sets, frozen prompt references, cached gold facet labels) |
| `data/raw/`, `data/processed/` | large regenerable data (git-ignored) |
| `experiments/` | one folder per evaluation arm: `generations.csv`, `evaluated_result*.csv`, `analysis*.ipynb`, `notes.md`; plus the study folders. **All kept**, including `_archive/` |
| `docs/` | append-only engineering record (index in `docs/README.md`); `docs/thesis_context/` = this folder |
| `hpc/komondor/containers/` | Singularity definitions: `baseline-gpu.def` (Vicuna text generation), `adapter-gpu.def` (Vicuna adapter), `adapter-qwen-gpu.def` (current) |
| `hpc/komondor/jobs/` | SLURM scripts; `run11_all.sh` is the final model |
| `notebooks/` | exploration only (EDA, embedding job, token estimates) |
| `archive/` | superseded work |
| `thesis/` | the LaTeX thesis (BME VIK template) |
| `checkpoints/` | trained weights (git-ignored) |

## 2. The pipeline, script by script (final model)

| step | script | runs on | output |
|---|---|---|---|
| 1. ogbn-arxiv | `src/data/download_ogbn.py` | CPU | `data/raw/ogbn_arxiv/` |
| 2. S2 bulk download | `src/data/s2/download_shards.py` (`run_download_dataset.sh`) | HPC CPU | ~500 GB raw shards |
| 3. S2 joins | `s2/build_dataset.py` (DuckDB), `s2/filter_ogbn_references.py`, `s2/add_titles_years.py` | HPC CPU | `papers_filtered.parquet`, `references_subset.parquet` |
| 4. joint dataset | `src/data/build_joint.py` | CPU | `data/processed/joint/{nodes,edges}.csv`, `splits.json` |
| 5. embeddings | `notebooks/adapter/embed_nodes.ipynb` (Voyage API, resumable) | API | `node_embeddings.npy` |
| 6. node sets | `src/adapter/prepare_data_capped.py`, `prepare_run9_sets.py` | CPU | `data/frozen/*_capped.json`, `training_sets_run9.json` |
| 7. text-arm references | `freeze_prompt_refs.py` | CPU | `data/frozen/prompt_refs_capped.json` |
| 8. GNN grid | `pretrain_encoder.py` (×8), `select_encoder.py` | GPU | `checkpoints/run11/gnn/*.pt`, `winner.pt` |
| 9. reader, joint | `train.py` | GPU | `checkpoints/run11/{reader,joint}/epoch*.pt`, `best.pt` |
| 10. loop decision | `generate.py --subset 200`, `loop_check.py --choose 6` | GPU / CPU | `checkpoints/run11/ngram` |
| 11. generation | `generate.py --input {zero,stage-a,refs-mean,oracle}` | GPU | `experiments/<tag>_run11-*/generations.csv` |
| 12. scoring | `src/evaluation/evaluate_results_scores_only.py`, `evaluate_detailed.py` | CPU + OpenAI API | `evaluated_result*.csv` |
| 13. analysis | `analysis.ipynb`, `analysis_detailed.ipynb` per arm | CPU | figures/tables |

Supporting modules: `config.py` (all constants, with the reason for each value), `graph.py`
(PyG graph + `NeighborLoader`), `model.py` (GNN, estimator, prefix adapter, oracle/refs-mean
vectors), `text.py` (prompts, corpus, target cleaning), `lora.py` (in-place LoRA injection),
`prefix_generation.py` (the patched `generate()`), `identify.py` (TF-IDF identification),
`rank.py` (ranking loss, history only), `whiten_features.py` (history only).

## 3. Software stack

| component | version / setting |
|---|---|
| base image | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| transformers | **4.51.3** (pinned exactly; first line with Qwen3; prefix injection depends on its cache internals) |
| peft / accelerate | 0.15.2 / 1.6.0 |
| PyTorch Geometric | 2.8.0, with `torch_sparse 0.6.18+pt22cu121`, `torch_scatter 2.1.2+pt22cu121` (sampler backend for `NeighborLoader`) |
| numpy | < 2 (the base image's torch is built against 1.x) |
| others | pandas 2.2.2, wandb 0.28.0, sentencepiece 0.2.0 and protobuf 4.25.3 (to load the Vicuna tokenizer that defined the frozen references), scikit-learn (TF-IDF) |
| evaluation (local) | `rouge-score`, `bert-score`, `openai` (judge `gpt-5.4-mini`), python-dotenv |
| embeddings | Voyage AI `voyage-4-large` |
| tracking | Weights & Biases project `dytag-adapter` |
| local environments | conda `DyTAG` (transformers 4.39.3, CPU work) and `DyTAG-qwen` (transformers 4.51.3, Qwen tokenizer checks) |

## 4. Hardware and HPC (Komondor)

- Komondor (Hungarian national supercomputer), account `c_gnn42`, GPU partition, **one NVIDIA
  A100 40 GB** per job, 8 CPUs, 6 GB RAM per CPU.
- Singularity container built with `--fakeroot` on a login node into node-local `/tmp` (Lustre
  cannot host a fakeroot build), then copied to the project directory. Compute nodes use the
  pre-downloaded HF cache (`HF_HOME=/app/.hf_cache`, `HF_HUB_OFFLINE=1`).
- Jobs bind the project directory as `/app`; W&B key from an untracked `.env`.
- **Resumability by design:** training resumes from the newest epoch checkpoint (`sort -V`);
  generation appends only missing `node_id`s; a fixed evaluation tag file keeps resubmissions
  writing into the same experiment folders; a `STOPPED` file written by a stop rule blocks
  accidental continuation.
- Local laptop (16 GB) is used only for CPU preparation and verification; no model is
  instantiated at full size locally.

## 5. Verification before GPU time (run on CPU at the start of every job)

The project's central engineering lesson is that failures in this layer are **silent** (fluent
text, falling loss). Each risk has a CPU check, most with a negative control:

| script | proves |
|---|---|
| `verify_generation.py` | a prefix survives `generate()`: patched 20/20 tokens match a plain run; unpatched control fails; logit error ~1e-7; asserts the transformers version |
| `verify_sliced_loss.py` | the supervised-positions-only loss equals the standard loss (4.8e-07), including with prefix and padding |
| `verify_lora_prefix.py` | LoRA does not disturb the prefix path; GQA (kv ≠ q) exercised; gradients reach prefix and every LoRA tensor; save/reload bit-identical |
| `verify_graph_only_prompt.py` | the exact ChatML prompt; labels mask exactly the prompt; all nodes have targets |
| `verify_decoder_inputs.py` | 20+ checks for reader inputs: reference mean reads exactly the cited papers and never the target; refs-mix behaviour; epoch slicing; GNN head loading; empty text ranks last |
| `model.py` (smoke test) | target masking (randomising the target row leaves the output bit-identical); gate = 0 gives exactly `base`; K/V decoders move only their own role |
| `graph.py` assertions | no self-loops/duplicates; seeds in rows 0…B−1; a seed aggregates exactly the papers it cites in the raw CSV |

Checkpoints record their configuration (prompt type, features file, oracle flag, LoRA targets,
GNN type and depth, node sets, joint flag); resume and generation refuse a mismatch, because
mismatched checkpoints often have identical tensor shapes and would load silently.

## 6. Reproducing the final experiments

```bash
# data (once)
python src/data/download_ogbn.py
sbatch hpc/komondor/jobs/run_download_dataset.sh        # S2 shards, then the s2/ join scripts
python src/data/build_joint.py
# embeddings: notebooks/adapter/embed_nodes.ipynb (needs VOYAGE_API_KEY)

# node sets (CPU, from inside src/adapter/ — flat imports)
cd src/adapter && python prepare_data_capped.py && python prepare_run9_sets.py

# container (Komondor login node)
singularity build --fakeroot /tmp/adapter-qwen-gpu.sif hpc/komondor/containers/adapter-qwen-gpu.def

# the whole of run 11: GNN grid → reader → loop test → reader arms → joint → oracle
sbatch hpc/komondor/jobs/run11_all.sh                   # ~50 h estimated; resubmit to resume

# scoring (local)
python src/evaluation/evaluate_results_scores_only.py --input experiments/<arm>/generations.csv \
       --output experiments/<arm>/evaluated_result.csv
python src/evaluation/evaluate_detailed.py --input experiments/<arm>/generations.csv \
       --output experiments/<arm>/evaluated_result_detailed.csv
python src/analysis/retrieval_baseline.py               # no-LLM floor
```

Deterministic elements: frozen node sets (seed 42), greedy decoding, judge at temperature 0,
seeded epoch permutation, seeded distractor order. Non-deterministic: GPU kernels in training
(no full determinism flags), the judge API (temperature 0 is not a guarantee of identical
outputs across model updates), the Voyage API.

## 7. Cost and time (measured where stated)

| item | time / cost |
|---|---|
| run-11 reader training | 11.5 h (A100) |
| run-11 GNN grid (8 variants) | ~31 min |
| run-11 joint fine-tune | 5.7 h |
| one 945-paper generation arm | ~7 GPU-h (plan estimate) |
| run-10a reader training | ~12 h (measured) |
| old-rubric judge, one arm | 945 calls to `gpt-5.4-mini` |
| detailed evaluator, one arm | ~4.8× the old-rubric cost |

## 8. Publication on GitHub — what is still needed

Not yet published (the repository has **no remote**). Before publishing:
1. **Rotate the API keys** that are in `.env_local`: the file was committed in earlier history
   (first in `9c01db0`) and, although untracked on `final_model` and listed in `.gitignore`,
   it is still in every earlier commit. Either rotate all keys (OpenAI, Voyage, Semantic
   Scholar, W&B) or publish a history-rewritten/fresh repository.
2. Update the root `README.md` (it still describes the Vicuna-7B / GraphSAGE pipeline) to
   describe the run-11 pipeline and point to `docs/run11_model.md`.
3. Decide on data distribution: `data/frozen/` is versioned; the joint dataset and embeddings
   are large and regenerable (licences: ogbn-arxiv/MAG, Semantic Scholar terms, Voyage terms).
   Model checkpoints are not in git.
4. Add a licence file and a minimal environment file (`requirements.txt` or conda YAML) for the
   local evaluation scripts.
5. Optionally a single-command CPU smoke test (the verify scripts already form one).
