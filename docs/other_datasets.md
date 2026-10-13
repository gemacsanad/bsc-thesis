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
