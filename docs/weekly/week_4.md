# Week 4 — Reference Filtering Pipeline

> **Note — the repository was reorganised on 2026-08-05.** Paths below are as they were when
> each section was written. The code is now under `src/`, datasets under `data/`, results under
> `experiments/`, and job scripts under `hpc/komondor/`. See [`README.md`](../../README.md) for the
> current mapping. The reasoning and the numbers are unaffected.

**Date:** 2026-06-30 / 2026-07-01

## Overview

This week's work focused on building a filtering pipeline for the citation graph. The goal is to identify which reference papers (cited by a test node) are genuinely relevant to the test paper before using them as input for abstract regeneration. Four scripts were added under `cosine_filter/` and a new analysis notebook was created.

---

## Dataset context

- **Graph:** ogbn-arxiv citation network (CS arXiv papers)
- **Task:** Regenerate a test paper's abstract from the abstracts of the papers it cites
- **Edge direction:** `src_node_id` → `dst_node_id` means the src paper cites the dst paper
- **Subset:** `nodes_subset.csv` — 1,936 nodes (200 of which are test nodes, `is_test=1`); `edges_subset.csv` — 2,100 edges, all with a test node as `src`
- **Node features:** `node_features.npy` — pre-computed 128-dim word2vec embeddings from OGB (downloaded via `scripts/dataset_download.py`); **not used for filtering** — replaced by Voyage AI embeddings below

---

## Scripts created

### 1. `cosine_filter/relevance_filter.py`

**Purpose:** LLM-based binary relevance label for every edge.

For each edge `(src → dst)`, sends both papers' titles and abstracts to `gpt-5.4-mini` and asks whether the reference (`dst`) is directly relevant to the source paper (`src`) — relevant meaning it contributes meaningfully to the topic, methodology, or core contributions. The model responds with `True` or `False`.

**Output:** adds a `relevant` (boolean) column to `edges_subset.csv`.

**Implementation details:**
- Skips already-labelled rows on re-run (resumable)
- Checkpoints every 100 rows
- Exponential backoff retry on `RateLimitError` (up to 6 attempts)
- Uses `max_completion_tokens=5`, `temperature=0`

**Results on the 2,100 edges:**
| Label | Count |
|-------|-------|
| True (relevant) | 845 |
| False (not relevant) | 1,255 |

---

### 2. `cosine_filter/embed_abstracts_voyage.py`

**Purpose:** Generate high-quality semantic embeddings of all node abstracts using Voyage AI.

Embeds the abstract of every node in `nodes_subset.csv` using the `voyage-4` model with `input_type="document"`. Processes texts in batches of 128 with exponential backoff on rate limits.

**Output:** `ogbn_arxiv_dataset/node_subset_features.npy` — shape `(1936, D)`, float32, rows ordered to match `nodes_subset.csv`.

**Why Voyage AI instead of the OGB word2vec features:** The OGB `node_features.npy` provides 128-dim word2vec averages which lose sentence structure. `voyage-4` produces contextual embeddings that capture semantic meaning more accurately, which is important for the cosine-based filtering steps below.

---

### 3. `cosine_filter/cosine_similarity.py`

**Purpose:** Compute the direct cosine similarity between the two endpoint embeddings of each edge.

For each edge `(src → dst)`, computes `cos_sim(embed(src), embed(dst))` using the voyage-4 embeddings. All embeddings are L2-normalised once upfront so cosine similarity reduces to a dot product, vectorised across all 2,100 edges simultaneously.

**Output:** adds a `cosine_similarity` (float32) column to `edges_subset.csv`.

**Summary stats:** min=0.23, mean=0.62, max=0.97

---

### 4. `cosine_filter/reference_cosine_score.py`

**Purpose:** Score each reference by how central it is within the full set of references for its test paper.

**Method — per test node:**
1. Collect all reference node IDs (`dst_node_id` where `src_node_id == test_node`)
2. Fetch their voyage-4 embeddings and build an N×N cosine similarity matrix (every reference vs every other reference)
3. Sum each column → raw score: how similar reference *i* is to all other references in the set
4. Normalise the raw scores

**Normalisation methods** (selectable via `--normalization`):

| Flag | Formula | Column | Range |
|------|---------|--------|-------|
| `min_max` | `(x − min) / (max − min)` | `cosine_score_min_max` | [0, 1] |
| `mean` | `col_sum / N` then min-max rescaled | `cosine_score_mean` | [0, 1] |
| `softmax` | `exp(x) / Σexp(x)` then min-max rescaled | `cosine_score_softmax` | [0, 1] |
| `all` | all three (default) | — | — |

**Interpretation:** a high score means the reference is thematically central to the citing paper's full reference set — it is similar to many of the other references, suggesting it is genuinely relevant rather than tangentially cited.

**Output:** adds `cosine_score_min_max`, `cosine_score_mean`, `cosine_score_softmax` columns to `edges_subset.csv`.

**Verified properties of `cosine_score_min_max`:**
- Every test node with 3+ references produces exactly one 0.0 (least central ref) and one 1.0 (most central)
- Test nodes with 1 reference produce a single 1.0
- Test nodes with exactly 2 references: in theory equal column sums, but float32 asymmetry in the dot product causes 7 of 10 such nodes to produce one 0.0 and one 1.0 anyway; the remaining 3 produce two 1.0s

---

## Analysis notebook

### `cosine_filter/analysis.ipynb`

**Cell 1:** Load `edges_subset.csv`.

**Cell 2:** Drop rows with missing scores, print row count.

**Cell 3:** Compute and display the **mean absolute difference (MAD)** between `cosine_similarity` and each of the three cosine scores.

**Cell 4:** For each of the three scores, produce two side-by-side plots to support threshold selection:
- **Left — Score distribution by relevance label:** overlapping histograms of the score for `relevant=True` vs `relevant=False` edges, showing the degree of separation between the two groups
- **Right — Retention rate vs threshold:** as the minimum score threshold increases, what fraction of relevant and not-relevant edges survive; includes a dashed equal-retention marker

---

## Final state of `edges_subset.csv`

| Column | Type | Description |
|--------|------|-------------|
| `src_node_id` | int | Citing paper (test node) |
| `dst_node_id` | int | Cited paper (reference) |
| `relevant` | bool | GPT-5.4-mini relevance label |
| `cosine_similarity` | float32 | cos_sim(embed(src), embed(dst)) |
| `cosine_score_min_max` | float32 | Centrality score, min-max normalised |
| `cosine_score_mean` | float32 | Centrality score, mean normalised |
| `cosine_score_softmax` | float32 | Centrality score, softmax normalised |
