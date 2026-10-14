# Other datasets — proposed for future runs

Sources: `docs/other_datasets.md` (survey 2026-10-10; measurements 2026-10-13). These datasets
are **planned for future runs** of the same pipeline. Nothing below has been embedded or
trained on. Measurements were taken on CPU from downloaded samples.

---

## 1. Purpose

To show that the method — *write a node's own text from its neighbours through one soft
prompt* — is a general **text-attributed-graph (TAG)** method, not something specific to
scientific abstracts:

- **Generalisation:** does the pipeline transfer to other kinds of node text and edges?
- **Where a GNN earns its place:** on citation graphs neighbours are *similar* to the node, so
  the plain mean of their embeddings is already near the limit (`06_results.md` §6.2). On a
  graph whose neighbours are *complements* or link to the node for other reasons, a learned
  aggregator has a reason to beat the mean.

"The same pipeline" = Voyage embeddings → GNN → prefix adapter → Qwen3-14B + LoRA, **retrained
per dataset** (the paper-trained reader is not expected to transfer to other text as is).

## 2. Requirements a dataset must meet

1. Each node has its **own text of ~100–300 words** to generate (titles, or user nodes
   described by a job title, are not enough).
2. Neighbours whose text is informative about the node but does not contain it (a neighbour
   quoting the target is leakage).
3. **Tens of thousands of nodes** to train the reader (run 11 used 90k).
4. An accuracy rubric that carries over: what the item is, its domain, its scope.

## 3. One cheap probe for every graph

For 1,000 nodes with ≥ 80 words of text and ≥ 3 neighbours: TF-IDF of the node's own text
against the **concatenated texts of its neighbours**; top-1 = share of nodes whose neighbours
are closer to their own text than to any of the other 999. A model-free measure of "how much
the neighbours say about this node".

| graph | edges mean | top-1 of 1,000 | cos(neighbours, own) vs random pair |
|---|---|---|---|
| **arXiv/S2 citations (this thesis)** | references | **46.9%** (of 945) | 0.243 vs 0.102 |
| Amazon books co-purchase (CS-TAG Books-History) | also bought | 31.1% | 0.221 vs 0.062 |
| WikiCS, **lead sections re-fetched**, in-links | articles linking *to* the target | 24.1% | 0.213 vs 0.055 |
| WikiCS, re-fetched, out-links | articles the target links to | 10.8% | 0.218 vs 0.069 |
| WikiCS, re-fetched, both directions | | 10.8% | 0.219 vs 0.073 |
| WikiCS, re-fetched, target name masked in neighbours | | 10.6% | |
| WikiCS, original stored text (preprocessed tokens) | hyperlinks | 5.2% | 0.227 vs 0.118 |
| Goodreads, children's books | `similar_books` | 4.2% | 0.091 vs 0.045 |

The citation graph is the most informative of those measured. The candidates span a range of
neighbour signal strength, which is itself useful: the thesis can relate generation quality to
how much the neighbourhood says.

## 4. Proposed datasets

### 4.1 Wikipedia (WikiCS, expanded) — **primary candidate**

| | |
|---|---|
| node → text to generate | article → its **lead section** |
| neighbours | hyperlinked articles (in-links carry the most signal) |
| access | `github.com/pmernyei/wiki-cs-dataset` (`dataset/data.json` + `metadata.json`, MIT licence; 83 + 56 MB). Lead sections from the MediaWiki API (TextExtracts, `exintro`, plain text) |
| size | WikiCS: 11,701 nodes, 297,110 directed links |
| stored text | lower-cased, stop-words removed tokens (median 219) — not prose, so leads must be re-fetched by title |
| re-fetch result | **11,024 of 11,701 leads (94%)** in ~3 minutes (20 titles per request) |
| lead length | median 95 words, mean 120, p90 249; 58% ≥ 80 words |
| usable nodes now | **5,923** (lead ≥ 80 words and ≥ 3 linked articles with a lead) |
| leakage | 8% of linked articles' stored text contained the target's title; 5% of linked leads name the target; masking the name barely moves the probe (10.8% → 10.6%) |

Steps needed:
1. **Expand to tens of thousands of articles**, e.g. a breadth-first crawl of the MediaWiki API
   from the WikiCS seeds (the link graph would then come from current Wikipedia, not the 2020
   snapshot).
2. **Memorisation control:** Qwen3-14B was pretrained on Wikipedia; if the soft prompt only
   conveys *which* article it is, the model may recall the lead. Add a **title-only arm** (the
   article title as text, no graph) and compare.
3. Use **in-links** as neighbours (twice the signal of out-links); test out-links as an ablation.
4. Mask the target's name in neighbour text before embedding.

### 4.2 Amazon co-purchase products — **second domain, the "complements" test**

Two concrete options:

| option | access | size | node text | notes |
|---|---|---|---|---|
| **CS-TAG Books-History** | Hugging Face `Sherirto/CSTAG`, `History/History.csv` | 63 MB; 41,551 nodes / 358,574 edges | Amazon book description + title: median 131 words, mean 220, 68% ≥ 80 | Ready to use: neighbour lists are in the CSV. Degree median 3, mean 8.6. 16,564 nodes with ≥ 80 words and ≥ 3 neighbours. Same-title neighbours 0.10%. Probe 31.1%. |
| **TAGLAS Product-subset** | `github.com/JiaruiFeng/TAGLAS` | 54,025 nodes / 144,638 edges | title + description, 113.9 words on average (paper) | Co-purchase products from TAPE / ogbn-products: the closest ready-made graph where neighbours are complements (a phone case's neighbour is a phone). Not yet measured. |

Related sources (larger, more preprocessing):
- **CS-TAG Books-Children** (same repository, 110 MB, 76,875 nodes / 1,554,578 edges, same
  format as History; not measured).
- **Amazon 2018 metadata** (`mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/
  meta_<Category>.json.gz`; no explicit licence, cite the paper): Books 1.22 GB gz (2.9M items),
  description median 45 words, 41% ≥ 80, `also_buy` on 62% of items (mean 16); Electronics
  1.20 GB gz, median 59 words, `also_buy` on only 10%. Descriptions contain HTML.

The author does not intend to use two book datasets; choose one Amazon option.

### 4.3 Goodreads (UCSD Book Graph) — optional contrast case

| | |
|---|---|
| access | `mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/` (direct links; academic use only, no redistribution) |
| size | 1.9 GB gz for all 2.36M books; per-genre files 27 MB (poetry) – 348 MB (romance) |
| graph | book–book edges from each book's `similar_books` list |
| text | blurb: median 69 words (children), 75 (poetry); 12–21% empty; only 26–31% coded English |
| children's genre | 47% of books have any `similar_books`, 44% ≥ 3 within the genre file; **27,179 usable**; 0.00% of neighbours are another edition of the same work |
| probe | **4.2%** — neighbours are about the genre, not the book |

The extreme case of "the domain is recoverable, the specifics are not". It is a book dataset,
so it competes with 4.2 for the "books" slot.

## 5. Screened out (unsuitable as node-text generation targets)

| candidate | why unsuitable |
|---|---|
| CS-TAG Goodreads (676,084 nodes) | titles only, no descriptions |
| CS-TAG Ele-Computers / Ele-Photo / Sports-Fitness | node text is reviews or titles, not a description of the node |
| Amazon Reviews 2023 (HF `McAuley-Lab/Amazon-Reviews-2023`) | `bought_together` empty on all 20,000 sampled books — no graph |
| DTGB (Enron, GDELT, ICEWS, Stack Exchange, Google Maps, Amazon movies, Yelp) | rich text is on **edges**; node text is a name or job title |
| knowledge graphs (FB15K-237, WN18RR) | entity descriptions are one or two sentences |
| other citation graphs (ogbn-arxiv alone, PubMed, Cora) | same domain; useful only as a zero-retraining transfer check |

## 6. Protocol for each new dataset (before any GPU time)

Compute on CPU the three numbers run 11 computed for papers (`06_results.md` §6.2):
1. cosine of the neighbour mean to the node's own embedding;
2. the best non-negative neighbour weighting fitted to the answer;
3. the same fit against an unrelated node (control).

On papers, (2) gained no more over (1) than (3) did, so the mean was at the limit. If (2) beats
(3) clearly on a dataset (expected on complements graphs), a GNN has room to win there.

Then: Voyage embeddings for all nodes → reader (~13 h on Komondor at run-11 scale) → GNN grid
(minutes) → evaluation with the same ladder (zero / mean / GNN / oracle / text baseline) and a
dataset-appropriate rubric (what the item is, its domain, its scope).

## 7. Related work

TANS (Wang et al. 2024, https://arxiv.org/abs/2412.10136) generates node text from neighbours
with GPT-4o-mini (text prompts) on Cora, PubMed and airport graphs, evaluated via node
classification; it finds neighbour information matters most when a node has little text of its
own. Dataset sources: CS-TAG (NeurIPS 2023 D&B), TAGLAS (https://arxiv.org/abs/2406.14683),
DTGB (https://arxiv.org/abs/2406.12072), TEG-DB (https://github.com/Zhuofeng-Li/TEG-Benchmark).
