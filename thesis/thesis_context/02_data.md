# Data — the citation graph, its preparation and preprocessing

Sources: `docs/adapter.md` (stages 1–2, 1b–1f), `docs/project_overview.md` §2,
`src/data/s2/README.md`, `src/data/build_joint.py`, `notebooks/eda/enrichment_analysis.ipynb`,
`notebooks/adapter/embed_nodes.ipynb`, `docs/step8_representation_plan.md`,
`docs/soft_prompt_only.md` (step 8), `src/adapter/whiten_features.py`,
`src/adapter/prepare_run9_sets.py`, `src/adapter/text.py`, and statistics recomputed from
`data/processed/joint/` on 2026-10-13 (marked **[recomputed]**).

---

## 1. The task the data must support

Given a target paper, generate its abstract from its references only. So the data needs:
(a) papers with an abstract each (the target text), (b) the real reference list of each target,
(c) the referenced papers' text or embeddings, (d) a split that keeps evaluation papers unseen.

## 2. Sources

### 2.1 ogbn-arxiv (Open Graph Benchmark)

- 169,343 computer-science arXiv papers, 1,166,243 directed citation edges, built from the
  Microsoft Academic Graph (MAG). Each paper has a `mag_id`, title, abstract (from the OGB
  `titleabs` file), year and a subject label (40 arXiv CS categories; the label is dropped in
  the joint dataset).
- **Time-based split** (OGB's own): train = papers up to 2017, validation = 2018, test =
  2019–2020. **[recomputed]** train 90,941 (years 1971–2017), valid 29,799 (2018), test 48,603
  (2019–2020).
- **Limitation:** it records only citations *between* arXiv-CS papers. Out-degree mean 6.89,
  median 4; 17,440 papers have no references at all (enrichment notebook).

### 2.2 Semantic Scholar (S2) bulk datasets

Downloaded to recover the real bibliographies (`src/data/s2/`):

| S2 dataset | used for |
|---|---|
| `papers` | `externalids.MAG` → the `mag_id ↔ corpusid` mapping |
| `abstracts` | `corpusid → abstract` text |
| `citations` | `citingcorpusid → citedcorpusid` edges |

- About 500 GB of raw shards, downloaded resumably (`download_shards.py`, HTTP range resume).
- Joined out-of-core with **DuckDB** (`build_dataset.py`) into `papers.parquet`
  (`mag_id, corpusid, abstract`) and `references.parquet` (`citing_mag_id, cited_mag_id`).
- Only papers with a MAG id **and** an abstract survive; edges whose endpoints lack a MAG id
  are dropped. Then filtered to the ogbn-arxiv papers and everything they cite
  (`papers_filtered.parquet`: 314,224 papers; `references_subset.parquet`: 2,986,960 edges).

### 2.3 The joint dataset (`src/data/build_joint.py` → `data/processed/joint/`)

- ogbn-arxiv node ids 0…169,342 are kept unchanged; S2-only papers get new ids from 169,343 in
  sorted `mag_id` order. A boolean `core` column marks the original ogbn-arxiv papers.
- Where a paper exists in both, ogbn-arxiv's title/abstract/year win.
- Edge direction was verified to agree in both sources: `(src, dst)` = "src cites dst".
- Validation in the enrichment notebook, all PASS: no duplicate `mag_id` or `node_id`, no
  duplicate edges, all endpoints valid, every original ogbn edge preserved, `core` exactly
  equals the ogbn-arxiv set.

| | value |
|---|---|
| nodes | **384,756** (169,343 core + 215,413 S2-only) |
| directed edges | **3,844,151** (20 self-loops, dropped in the graph tensors → 3,844,131) |
| edges out of core papers | 2,125,355, of which 889,024 point to S2-only papers |
| new core→core edges found only via S2 | 70,088 |
| core out-degree before → after enrichment | mean 6.89 → 12.55, median 4 → 8 |
| core papers with no references | 17,440 → 8,675 |
| core papers with > 50 references | 499 → 4,284 |
| max out-degree / max in-degree **[recomputed]** | 832 / 16,504 |
| abstract length, words **[recomputed]** | core: median 157, mean 160.9, p10 91, p90 237. S2-only: median 157, mean 164.4 |
| missing abstracts **[recomputed]** | 1 |

### 2.4 The S2 coverage confound (a finding that shapes every result)

Among the 169,343 core papers, **94,756 (56%) have an S2 bibliography**, mean out-degree 17.0;
**74,587 (44%) do not** and keep only their arXiv-CS citations, mean out-degree 6.9
(`adapter.md` stage 1). So a low out-degree means *either* a short bibliography *or* an
unobserved one, and which cannot be known. This was **measured, not removed** — filtering it
would silently change the population the results describe. S2 coverage rises with out-degree
(31% at degree 5 → 64% at degree 15 in the first eval set).

## 3. Node features: Voyage embeddings

- Model **`voyage-4-large`**, `input_type="document"`, batch size 128
  (`notebooks/adapter/embed_nodes.ipynb`). **Text embedded = `title + "\n\n" + abstract`.**
  Empty texts replaced by a single space.
- Output `node_embeddings.npy`: float32, shape (384,756, 1024), row *i* = `node_id` *i*
  (asserted), 1.58 GB, unit-norm rows. Written to a memory map with a progress file so the job
  resumes after a crash.
- **Why not the OGB features:** ogbn-arxiv ships 128-d averaged word2vec vectors, which lose
  sentence structure. The reader has to recover what a paper is *about*, so a sentence-level
  embedding was used instead.
- **Consequence to state in the thesis:** the target's embedding encodes its **title** as well
  as its abstract, so the oracle input carries the title's information. The text baseline does
  **not** see the target's title (it shows only the references' titles and abstracts; corrected
  in `run9_embedding_decoder.md` §5).
- The first experiments (week 4, 1,936-node subset) used `voyage-4`; the joint dataset uses
  `voyage-4-large`.

### 3.1 Geometry of the embedding space (measured in step 8)

- **The raw features are anisotropic.** Over the 25,539 nodes reachable from the eval set:
  participation ratio 92.6 of 1024; norm of the mean unit vector 0.5975 (isotropic would be
  ~0.006) (`config.py`, `whiten_features.py`).
- **Reference means are even more alike.** The pooled reference vectors of eval papers sit at
  pairwise cosine **0.6602**, with cosine 0.8127 to the global mean (`soft_prompt_only.md`
  step 8).
- **How close the references are to the target:** the plain reference mean has cosine **0.823**
  to the target's own embedding on the 945 eval papers (0.816 on the 806 validation papers),
  and retrieves the target's own embedding among the 945 at **64.3%** top-1 (raw space;
  `run11_model.md`, `soft_prompt_only.md`).

## 4. Graph construction (`src/adapter/graph.py`)

- **Directed, references only.** `edge_index` is stored reversed, as (cited → citing), so in
  PyG messages flow from a paper's references into the paper. A node learns only from works it
  cites. Reason: at writing time a paper's references exist but no one has cited it yet;
  aggregating citing papers would leak the future.
- **Undirected edges were implemented first and reverted**: temporal leak, plus hub papers
  (in-degree up to 16,513) inflated 2-hop neighbourhoods to 29,650 nodes at p99 and 94,721 at
  worst.
- Self-loops dropped (20), edges coalesced. All neighbours taken at every hop
  (`num_neighbors = [-1] * layers`), no sampling cap.
- Measured on early eligible train seeds (out-degree 5–15): 1-hop mean 8.6, 2-hop mean 50.6
  (median 36, p99 215); only 0.3% of seeds have an empty 2-hop neighbourhood; 10% of 1-hop
  neighbours are dead ends.
- **Assertions run on every build:** no self-loops, no duplicate edges, seed nodes occupy rows
  `0…batch_size−1` (needed by the mask), and **a seed aggregates exactly the papers it cites in
  `edges.csv`** (the direction check against the raw file).
- Data artifacts recorded: mutual citation pairs (~1.5% of a 500k-edge sample) and the 20
  self-loops — impossible in a strict citation DAG.

## 5. Node sets: which papers are trained and evaluated on

All frozen sets are versioned in `data/frozen/` so every experiment reads the same nodes.

### 5.1 History of the sets

| set | when | definition | sizes |
|---|---|---|---|
| first eligible / eval | 2026-07-27 | core, out-degree 5–15 (what fit Vicuna's 4,096-token context with all references as text); eval 45 per degree | eligible 65,609 (36,093 / 12,020 / 17,496); eval 495 |
| **capped** | 2026-08-06 / -11 | core, out-degree **5–100**; quotas per out-degree (train 200, valid 10, eval 10), drawn with `random.Random(42)`; quotas are ceilings since high degrees run out | eligible 111,538 (48,479 / 22,296 / 40,763); **train 7,750 / valid 806 / eval 945** |
| **run-9 training set** | 2026-10-07 | **every** train-split core paper, minus stubs and near-duplicates (below); validation unchanged | **train 90,461 / valid 806**; eval 945 unchanged |

- **The 945 eval papers** (`eval_nodes_capped.json`) come from the test split (years 2019:
  750, 2020: 195 **[recomputed]**), cover all 96 out-degrees 5–100 (at most 10 per degree),
  out-degree mean 51.8, median 52. Gold abstracts: median 170 words, mean 174.3.
- **The 806 validation papers** are from 2018, out-degree median 45, mean 45.4
  **[recomputed]**.
- **The run-9/11 training papers**: out-degree median 5, mean 7.9; 7,202 have no references
  at all, 83,259 have ≥ 1 **[recomputed]**.
- **Train/eval shape mismatch (state it):** training papers are mostly low-degree, eval papers
  high-degree, because the graph simply lacks enough high-degree training papers. Already true
  of the capped set (train mean degree 25.5 vs eval 51.8).

### 5.2 Filters applied to the run-9 training set (`prepare_run9_sets.py`)

From 90,941 train-split core papers to 90,461:
- the 945 eval and 806 validation seeds removed (asserted none are train-split anyway);
- abstracts under 30 words removed (477 stubs);
- near-duplicates of an eval paper removed (3): same normalised title, or raw Voyage cosine
  > 0.95 to an eval paper. A training target that *is* an eval paper under another id would be
  a leak.

### 5.3 GNN training set

The run-11 GNN grid trains on the **83,259** training papers with at least one reference. The
others (8%) are dropped: their estimate would be the `[MASK]` row alone, the same vector for
all of them.

## 6. The text baseline's reference selection (prompt budget)

The text arm cannot show every reference. Which references a prompt shows:
- score each reference by its distance to the centroid of the paper's own reference set,
  `(1 − cos(ref, centroid)) / 2` on unit Voyage embeddings (`neighbour_cap.py`, the same score
  as `src/filters/centroid_cosine_distance.py`);
- keep at most **15** closest, then drop the farthest until the prompt fits
  `4,096 − prefix − max(320, target length)` tokens.
- Frozen once under the Vicuna tokenizer in `data/frozen/prompt_refs_capped.json` (9,501
  papers, mean 12.87 references) so Qwen and Vicuna read byte-identical bibliographies (48/48
  stored prompts matched string for string).
- **What this means [recomputed]:** an eval paper's prompt shows mean 13.5 (median 14)
  references, a **median 27%** (mean 39%) of its references. The titles+abstracts of *all* its
  references total a median **8,605 words** (max 21,072) — roughly 11k tokens, far beyond the
  4,096-token context. The soft-prompt arms see **all** references through one vector.
- Above out-degree ~16 the text prompt stops growing (~14 references, ~3,700 Vicuna tokens),
  so anything a graph arm gains at high degree has to come through the graph channel.

## 7. Preprocessing applied for training

### 7.1 Target masking (all runs)

The target's own row is replaced by one learned `[MASK]` vector before message passing,
because its Voyage embedding *is* the answer. Enforced by a test: randomising the seed rows of
the input leaves the encoder output bit-identical. Masking is by row position, so when two seeds
in one batch cite each other one of them also aggregates the mask (0.19% of seed appearances at
batch 8); kept deliberately because it also closes the mutual-citation self-leak.

### 7.2 ZCA whitening of the Voyage features (runs 8 and 9 only; **not** used by the final model)

- **Why it was tried:** the near-constant reference means (pairwise cosine 0.66) produced a
  near-constant prefix and one generated string for 945 papers in step 7. The adapter trunk's
  LayerNorm normalises per sample and cannot remove a component shared across samples.
- **What:** `W = V diag(λ + floor)^−½ Vᵀ` with `floor = 1e-3 · λmax`, applied to centred
  features, rows re-normalised to unit length (`whiten_features.py`). Fitted on the 383,005
  nodes that are not eval/validation seeds, streamed in 32k-row blocks (laptop, ~15 s, ~1.1 GB).
  ε swept 1e-4…1e-1 on 945 held-out training nodes: dev retrieval moved only 69.3% → 70.6%.
- **Measured effect (train-only fit):**

  | | raw | whitened |
  |---|---|---|
  | eval retrieval by reference mean (top-1 / 945) | 64.34% | 72.59% |
  | pooled pairwise cosine | 0.6602 | 0.0547 |
  | participation ratio | 122 / 1024 | 948 / 1024 |
  | norm of the mean unit vector | 0.5458 | 0.0078 |

- **Outcome:** both whitened runs failed to use the vector (run 8 collapsed for a different
  root cause; run 9 never opened its gate). Run 10a returned to **raw features** and worked;
  the final model (run 11) uses **raw** features. Whitening is the leading suspect for run 9's
  failure: in raw space most variance sits in a few topic directions that even an untrained
  adapter passes through, giving the gate an early reason to open (`run9_embedding_decoder.md`
  §6). This is a suspicion, not an isolated ablation.
- The retrieval floor `src/analysis/retrieval_baseline.py` reports both spaces (27.5% raw /
  32.0% whitened top-1 for "nearest training abstract to the reference mean").

### 7.3 Cleaning the training targets (run 8 onward, **used by the final model**)

- **Why:** the system prompt asks for "no markup, mathematical notation or headings", while
  13.5% of training abstracts contained LaTeX — the instruction and the supervision disagreed.
- **What `text.clean_abstract` does:** unwraps `\textit{…}`, `\emph{…}`, `{\em …}` etc., strips
  unescaped `$ \( \) \[ \]` delimiters, resolves escapes and accents, removes the backslash of
  remaining control words **but keeps their names** (`$\Omega(N \log\log N)$` →
  `Omega (N log log N)`), drops grouping braces. Residual markup across 40,000 abstracts fell
  from ~18% to 1.0%; ~21 s at startup.
- A rule converting literal `\n`/`\t` was removed after measurement: the corpus has 10 real
  `\n`/`\r` escapes and 5,172 LaTeX commands starting with those letters (`\times`, `\tilde`…).
- **Training targets are cleaned; evaluation golds are not** (`generate.py` uses
  `clean=False`), so scores stay comparable with all earlier runs.
- **Declared mismatch:** embeddings were computed on the raw text; re-embedding 384,756
  documents was not done.

### 7.4 Normalisation

All vectors fed to the reader are unit length (Voyage rows, the reference mean, the GNN head
output after `F.normalize`).

## 8. Data analyses already done (with results)

| analysis | result | source |
|---|---|---|
| S2 enrichment before/after | out-degree mean 6.89 → 12.55; 70,088 new core–core edges; 889,024 edges to new nodes | enrichment notebook |
| coverage confound | 56% / 44%, mean out-degree 17.0 / 6.9 | `adapter.md` stage 1 |
| relevance of references (LLM labels, 2,100 edges of the early subset) | 845 relevant vs 1,255 not (~60% arguably noise) | `docs/weekly/week_4.md` |
| cosine of citing–cited pairs (early subset) | min 0.23, mean 0.62, max 0.97 | week 4 |
| context overflow vs degree | at out-degree 15, 58% of candidates did not fit 4,096 tokens | `adapter.md` stage 4c |
| embedding anisotropy, whitening | table in §7.2 | step 8 |
| reference-mean retrieval of the target | 64.3% top-1 / 945 (raw), 96.0% top-10; first 512 dims 64.6%; uniform 1+2-hop mean 33.2%; best single reference (oracle choice) 84.1% | `soft_prompt_only.md` step 1 |
| which references to keep when capping at 64 (345 eval papers where the cap bit) | all 70.43%, 64 nearest centroid 66.38%, random 69.57%, farthest 71.88% | step 7 |
| identity in an abstract's opening (TF-IDF, first N words vs remainder) | 16 words 47.8%, 24 words 57.4%, 32 words 63.2%, 48 words 72.3% top-1 / 945 | step 7 |
| TF-IDF neighbour signal (1,000 nodes, own text vs concatenated neighbours) | citations 46.9% top-1; cos 0.243 vs 0.102 random | `other_datasets.md` |

## 9. Further data analyses recommended for the thesis (not yet done)

Proposals, listed so the data chapter covers the dataset fully. Each is CPU-only and cheap
unless noted.

1. **Degree distributions** — out- and in-degree histograms (log-log) for core and S2-only
   nodes, before/after enrichment (the notebook has the before/after table; a figure is
   needed). Mark the train / valid / eval degree profiles on the same axes to show the shift.
2. **Year distribution per split** and the citation-age distribution (target year − reference
   year). Shows the temporal split and how old the evidence is.
3. **Abstract length distributions** (gold eval vs training targets vs generated outputs). The
   generated abstracts are shorter (run 11 ≈ 95–101 words vs gold median 170), which affects
   ROUGE.
4. **Subject-category composition.** The ogbn-arxiv 40-class labels exist in
   `data/raw/ogbn_arxiv/nodes.csv`: category distribution of train vs eval, and category
   agreement between a paper and its references (a homophily rate).
5. **Homophily in embedding space:** distribution of cos(target, each reference), of
   cos(target, reference mean), and the same for random paper pairs; per-degree. The 0.823 mean
   is already known; the distribution and its tail (papers whose references do not describe
   them) is what explains where the model fails (`06_results.md`, quintile analysis).
6. **Embedding visualisation:** a 2-D projection (UMAP or PCA) of eval-paper embeddings
   coloured by category, with each paper's reference mean and GNN estimate overlaid for a few
   examples.
7. **Coverage confound, stratified:** S2-covered vs uncovered among the 945 eval papers, and
   whether scores differ between them.
8. **Near-duplicate / version analysis:** how many references are other versions of the
   target (cosine ≥ 0.95); the detailed evaluator already excludes these as distractors.
9. **Information budget table:** words/tokens of all references vs what the text prompt shows
   vs 32 prefix positions (numbers in §6).
10. **Anisotropy figure:** eigenvalue spectrum of the raw vs whitened embedding covariance
    (numbers in §7.2).
