# Other datasets: where else the soft-prompt model could run

Written 2026-10-10. A survey of candidates, not yet acted on. Nothing has been downloaded or
embedded.

## Why

Run 11 showed that on the citation graph the plain mean of a paper's references already
reaches the limit of what the neighbourhood says about it (cosine ~0.82–0.83 to the paper's own
embedding; see `run9_embedding_decoder.md` §9). Running the same pipeline on other graphs would
test two things:

- **Generalisation:** does "write a node's own text from its neighbours, through one soft
  prompt" work outside scientific papers?
- **Where a GNN earns its place:** on a graph whose neighbours are *not* similar to the node,
  the mean should not be enough, and a learned aggregator has a reason to win.

"The same model" here means the same pipeline (Voyage embeddings → GNN → prefix adapter →
Qwen3-14B + LoRA), retrained per dataset. The paper-trained reader is not expected to transfer
to other kinds of text as is.

## What a dataset needs

1. **Each node has its own text of ~100–300 words** to generate. Titles alone, or user nodes
   described by a job title, are not enough.
2. **Neighbours with text that is informative about the node** but does not contain it. A
   neighbour quoting the target text is leakage.
3. **Tens of thousands of nodes** to train the reader (run 11 used 90k).
4. **An accuracy rubric that carries over:** what the item is, its domain, its scope.

## Recommended

| dataset | node → text to generate | neighbours | what it tests |
|---|---|---|---|
| **Goodreads books** | book → its description (blurb) | similar or co-shelved books | Another domain, the same "what is this item about" task. Neighbours are similar items, as with citations, so this is a generalisation test. |
| **Amazon products** | product → its description | co-purchased products | **Neighbours are complements, not look-alikes** (a phone case's neighbour is a phone), so the mean of neighbours is not the target. This is where a learned GNN should beat the mean if it can beat it anywhere. |
| **Wikipedia** (WikiCS, or a larger slice from a dump) | article → its lead section | hyperlinked articles | Clean, high-quality text and a natural rubric. Linked articles often name or describe the target, so leakage has to be filtered. WikiCS alone (11.7k nodes) is small for training the reader. |

**Where to get them.**

- **Goodreads:** book descriptions are in CS-TAG and TEG-DB. In TEG-DB the edges are user
  reviews, so the graph is user–book. The original UCSD Goodreads dump has a per-book
  `similar_books` list, which would give a book–book graph. Check that field before committing.
- **Amazon:** TAGLAS has a co-purchase product subset (54,025 nodes, 144K edges, title +
  description). The McAuley Amazon review metadata is larger. Some descriptions are short or
  templated, so filter by length.
- **Wikipedia:** WikiCS (11,701 nodes, 216K edges) is in CS-TAG and TAGLAS.

## Not a good fit

- **DTGB** (Enron, GDELT, ICEWS, Stack Exchange, Google Maps, Amazon movies, Yelp): the rich
  text is on edges (emails, reviews, answers). Node text is a job title, entity name or
  business name. Its generation task writes the text of an interaction, which would need a
  different model.
- **Knowledge graphs** (FB15K-237, WN18RR): entity descriptions are one or two sentences, too
  short for this setup.
- **Other citation graphs** (ogbn-arxiv, PubMed, Cora): same domain. Useful only as a cheap
  check that the existing model transfers without retraining, not as a different dataset.

## Check before any GPU time

For each candidate, compute on CPU the three numbers run 11 computed for papers (§9 of
`run9_embedding_decoder.md`):

1. the cosine of the neighbour mean to the node's own embedding;
2. the best non-negative neighbour weighting, fitted to the answer;
3. the same fit against an unrelated node, as a control for how much any target can be fitted.

On papers, (2) beat the mean by no more than (3) did, so the mean was already at the limit.
If on Amazon (2) beats (3) by a clear margin, the GNN has room to win, and that is known before
anything is trained. The thesis comparison is then: a GNN adds nothing where neighbours look
like the node, and something (or not) where they complement it.

## Cost per dataset

- Voyage embeddings for every node (API cost).
- Reader training: ~13 h on Komondor.
- The GNN grid: minutes.
- Judge runs.

## Related work

TANS ("Can LLMs Convert Graphs to Text-Attributed Graphs?", arXiv 2412.10136) generates node
text from neighbours with GPT-4o-mini, on Cora, PubMed and three airport graphs, and evaluates
it through node classification. Neighbour information matters most when a node has little text
of its own.

## Sources

- [DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs](https://arxiv.org/html/2406.12072v3)
- [TEG-DB / TEG-Benchmark](https://github.com/Zhuofeng-Li/TEG-Benchmark)
- [CS-TAG: A Comprehensive Study on Text-attributed Graphs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/37d00f567a18b478065f1a91b95622a0-Abstract.html) and [TAG-Benchmark repo](https://github.com/sktsherlock/TAG-Benchmark)
- [TAGLAS: An atlas of text-attributed graph datasets](https://arxiv.org/html/2406.14683)
- [TANS: Can LLMs Convert Graphs to Text-Attributed Graphs?](https://arxiv.org/html/2412.10136)

## Measured, 2026-10-13: access, size, text length and neighbour signal

Samples were downloaded to a scratch directory and measured on CPU. Nothing was embedded or
added to `data/`.

### One probe for every graph

For 1,000 nodes with ≥ 80 words of text and ≥ 3 neighbours: TF-IDF of the node's own text
against the concatenated texts of its neighbours. Top-1 is the share of nodes whose neighbours
are closer to their own text than to any of the other 999. This is a cheap, model-free version
of "how much do the neighbours say about this node", the quantity run 11 found to be the ceiling.

| graph | edges mean | top-1 of 1,000 | cos(neighbours, own) vs random |
|---|---|---|---|
| **arXiv/S2 citations (ours)** | references | **46.9%** (of 945) | 0.243 vs 0.102 |
| Amazon books, co-purchase (CS-TAG History) | also bought | 31.1% | 0.221 vs 0.062 |
| WikiCS | hyperlinks | 5.2% | 0.227 vs 0.118 |
| Goodreads, children's books | `similar_books` | 4.2% | 0.091 vs 0.045 |

**Our citation graph is the most informative of the four.** No candidate measured here gives
the model more to work with than the dataset it already has.

### Per dataset

| dataset | how to get it | download | nodes / edges | node text | findings |
|---|---|---|---|---|---|
| **CS-TAG Books-History** | HF `Sherirto/CSTAG`, `History/History.csv` | 63 MB | 41,551 / 358,574 | Amazon book description + title: median 131 words, mean 220, 68% ≥ 80 | Ready to use, with neighbour lists in the CSV. Degree median 3, mean 8.6. 16,564 nodes have ≥ 80 words and ≥ 3 neighbours. Same-title neighbours 0.10%. |
| CS-TAG Books-Children | same repo, `Children/Children.csv` | 110 MB | 76,875 / 1,554,578 | same format | Not measured; the same construction as History. |
| CS-TAG Goodreads | same repo | 95 MB | 676,084 / 8,582,324 | **titles only** | Unusable as a target: there are no descriptions in the file. |
| CS-TAG Ele-Computers / Ele-Photo / Sports-Fitness | same repo | 53 / 77 / 39 MB | 87k / 48k / 173k nodes | reviews (Computers, Photo), titles (Fitness) | Not a description of the node; unusable. |
| **Goodreads (UCSD Book Graph)** | `mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/`, direct links. Academic use only, no redistribution. | 1.9 GB gz all books; per genre 27 MB (poetry) to 348 MB (romance) | 2.36M books; edges from `similar_books` | blurb: median 69 words (children), 75 (poetry); 12–21% empty | 47% of children's books have any `similar_books`, 44% have ≥ 3 inside the genre file. 27,179 usable children's books. No leakage: 0.00% of neighbours are another edition of the same work. Neighbours are genre-level, hence the 4.2% probe. Only 26–31% are coded English. |
| **Amazon 2018 metadata** | `mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_<Category>.json.gz`, direct. No explicit licence; cite the paper. | Books 1.22 GB gz (2.9M items); Electronics 1.20 GB gz; all categories 24 GB | built from `also_buy` / `also_view` | Books description median 45 words, 41% ≥ 80; Electronics median 59, 42% ≥ 80 | Books: `also_buy` on 62% of items, mean 16. Electronics: `also_buy` on only **10%**. Descriptions contain HTML. |
| Amazon 2023 (Reviews'23) | HF `McAuley-Lab/Amazon-Reviews-2023` | Books meta 14.7 GB | — | — | **No graph:** `bought_together` was empty on all 20,000 sampled books. |
| WikiCS | `github.com/pmernyei/wiki-cs-dataset`, `dataset/data.json` + `metadata.json`. MIT licence. | 83 + 56 MB | 11,701 / 297,110 directed links | **preprocessed tokens: lowercased, stop words removed**, median 219 | Not prose, so it cannot be a generation target as is. Text would have to be re-fetched from Wikipedia by title. 8% of a node's linked articles contain its title (leakage). |
| TAGLAS Product-subset | `github.com/JiaruiFeng/TAGLAS` | — | 54,025 / 144,638 | title + description, 113.9 words on average (paper) | Not measured. From TAPE / ogbn-products: co-purchase, so the closest ready-made "complements" graph. |

### What this means for the recommendations above

- **Best second domain: Amazon books co-purchase** (CS-TAG Books-History, then Children). It is
  the only candidate with a strong neighbour signal (31.1%), it's a 63 MB download with the
  graph included, and the text is a description of the item. Expect a weaker result than on
  papers, not a stronger one.
- **Goodreads is a contrast case, not a win.** Its neighbours are about the genre, not the book
  (4.2%). On the run-11 argument that is useful: it is the extreme case of "the domain is
  recoverable, the specifics are not".
- **The "complements" hypothesis is untested.** The co-purchase graph that exists in quantity is
  book-to-book, and those neighbours look like the node (similar books). Electronics, where
  complements would appear, has `also_buy` on only 10% of items. TAGLAS Product-subset is the
  ready-made place to test it.
- **Drop WikiCS** unless the article text is re-fetched. Drop Amazon 2023 for graph work.

### WikiCS with the text recovered (2026-10-13)

The WikiCS text is unusable as stored, but its titles are intact. Fetching each article's lead
section from the MediaWiki API (TextExtracts, `exintro`, plain text, 20 titles per request,
about 3 minutes) recovered **11,024 of 11,701 leads (94%)**.

- **Lead length:** median 95 words, mean 120, p90 249; 58% have ≥ 80 words. Shorter than an
  abstract (median 174) but in the right range.
- **Usable nodes** (lead ≥ 80 words, ≥ 3 linked articles with a lead): **5,923**. That is far
  too few to train a reader; run 11 used 90k papers.
- **Leakage:** 5% of linked leads name the target. Removing the name barely moves the probe
  (10.8% → 10.6%), so the signal does not come from the name.

| neighbours | top-1 of 1,000 | cos vs random |
|---|---|---|
| out-links | 10.8% | 0.218 vs 0.069 |
| **in-links** | **24.1%** | 0.213 vs 0.055 |
| both | 10.8% | 0.219 vs 0.073 |

In-links (articles that link *to* the target) carry about twice the signal of out-links, which
tend to point at broad topics. Concatenating both dilutes the in-link signal.

Two open problems before it can be used:
1. **Size.** It needs expanding to tens of thousands of articles, for example by a
   breadth-first crawl of the API from the WikiCS seeds. The link graph would then come from
   today's Wikipedia rather than the 2020 snapshot.
2. **Memorisation.** Qwen3-14B was trained on Wikipedia. If the soft prompt only conveys
   *which* article it is, the model may recall the lead from pretraining. That would make the
   reader look better than the graph signal justifies. A title-only control is needed: prompt
   the model with the article's title alone, no graph, and score the result.
