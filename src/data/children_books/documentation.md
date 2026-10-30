# Children's books dataset

## Step 1: cleaning the CSV into the pipeline's format (2026-10-21)

### Why

The children's books are the second domain for the run-11 model: write a book's description
from its co-purchased books, as the model writes a paper's abstract from its references. The
choice and its reasoning are in `docs/other_datasets.md` ("Final recommendation, 2026-10-19").
This step converts the raw CSV into the `nodes.csv` / `edges.csv` pair that
`data/processed/joint/` uses, so the adapter code (`text.Corpus`, `graph.load_graph`) can
read the books by changing only the paths.

### Source

CS-TAG Books-Children, from the Amazon co-purchase graph. Public, no login:

```bash
curl -L https://huggingface.co/datasets/Sherirto/CSTAG/resolve/main/Children/Children.csv \
     -o data/children_books/Children.csv
```

One row per book: `category`, `text` ("Description: …; Title: …"), `label`, `node_id`,
`neighbour` (a list of co-purchased `node_id`s). `data/children_books/` is gitignored.

### How to run

```bash
cd src/data/children_books && python build_children.py
```

It writes `data/children_books/nodes.csv` and `data/children_books/edges.csv`.

### Decisions

- **The description goes in the `abstract` column** and the title in `title`, so `Corpus`
  reads the books unchanged. As with papers, only the description is the target, and the
  title is not shown.
- **Edges are made symmetric.** Co-purchase has no direction, but the CSV's neighbour lists
  are not symmetric. Every node now aggregates every book it was bought with. The 889 books
  with an empty neighbour list appear in other books' lists, so after symmetrising no node is
  isolated.
- **Only HTML entities and whitespace are cleaned.** 18.5% of rows have entities (`&#8220;`)
  and there are no tags. Reviewer quotes ("— Publishers Weekly") and School Library Journal
  grade tags ("PreS-Gr 1In this picture book…") are part of the blurbs as sold and are kept.
- **Seller notes are flagged, not removed** (`seller_note`): descriptions of a copy's
  condition ("Good condition. Ships same day."). The pattern looks only at the first 150
  characters, because a blurb can mention "highlighting" or "ships" anywhere. A broader
  pattern flagged 459 rows, mostly real blurbs. Nodes are never dropped: a book with a
  useless description is still a neighbour of other books.

### Results

| | |
|---|---|
| nodes | 76,875 |
| edges | 1,162,522 undirected pairs (2,325,044 directed rows) |
| degree | median 11, mean 30.2, none isolated |
| description | median 127 words, mean 190.4, p90 399; 55 empty; 64.1% ≥ 80 words |
| seller notes | 63, of which 1 has ≥ 80 words |
| **usable** (≥ 80 words, ≥ 3 neighbours, no seller note) | **38,992** |
| categories | 24, very uneven: "Growing Up & Facts of Life" 24,679 down to "Computers & Technology" 81 |

The usable count is higher than the 38,537 in `docs/other_datasets.md`, because that count
used the broader seller-note pattern.

### Known limitations

- **`text.clean_abstract` damages book text.** It is the LaTeX stripper for abstracts and
  treats `$` as a maths delimiter: "$1 million" becomes "1 million", "$16" becomes "16", and
  braces are dropped. It changes 469 descriptions (383 contain a `$`). The books must be
  loaded with `Corpus(..., clean=False)` (step 5).
- **21 descriptions contain pasted MS Word CSS** (`st1\:*{behavior:url(#ieooui) }`). Left
  in; too few to matter.
- **55 empty descriptions** cannot be embedded as they are (step 3 decides: embed the title,
  or a zero vector).

## Step 1, revised: empty books dropped, shared descriptions flagged (2026-10-21)

- **The 55 books with an empty description are dropped.** There would be nothing to compare a
  generation with. The rest are renumbered 0..N-1, so row *i* of the embedding matrix is
  `node_id` *i*. The original id is kept as `cstag_id`. 9 books whose only neighbours were
  empty books are now isolated.
- **24% of books share their exact description with another book** (`shared_description`:
  18,567 books in 4,052 groups, matched after normalising case and punctuation). These are
  mostly not editions: only 45 groups have one title. The rest are publisher or author
  boilerplate, such as "DK was founded in London in 1974…" (257 books), Speedy Publishing (230)
  and McGraw Hill (184). Such a text describes the publisher, not the book. 14,494 books have
  a neighbour with an identical description, so the answer would be among the inputs. These
  books **stay in the graph as neighbours but are never targets**, the same rule run 9 applied
  to near-duplicate papers.

| | |
|---|---|
| nodes | 76,820 |
| edges | 1,161,167 undirected pairs |
| shared descriptions | 18,567 books (6,218 of them ≥ 80 words) |
| **usable** (≥ 80 words, ≥ 3 neighbours, no seller note, description not shared) | **33,639** |

## Step 2: frozen node sets (2026-10-21)

`prepare_sets.py` is the counterpart of `src/adapter/prepare_run9_sets.py`. CS-TAG has no fixed
split, so it draws one, seed 42. The output is versioned, like `data/frozen/`:
`data/frozen/children_books/eval_nodes.json` and `training_sets.json`.

```bash
cd src/data/children_books && python prepare_sets.py
```

- **Evaluation 945 and validation 806**, the sizes of run 11. Both are drawn from the usable
  books, minus 994 that have a neighbour with the same normalised title. Another edition would
  hand the model its own text. That leaves a pool of 33,084.
- **Training is every other book, as in run 9,** minus:
  - descriptions under 30 words (7,833);
  - seller notes and shared descriptions (18,619);
  - same title as a held-out book (30).

  That leaves **52,301**. 79% have ≥ 80 words, and all but 7 have a neighbour for the GNN.
- **The voyage cosine > 0.95 filter is still to run:** it needs step 3. Re-running the script
  afterwards applies it to training only. The evaluation and validation books stay the same.
  The file records `"cosine_filtered"`.

| | words, median | degree, median |
|---|---|---|
| evaluation | 235 | 20 |
| training | 188 | 10 |

**Train and evaluation books differ, as papers did.** Evaluation books are longer and
better connected, because they were selected for ≥ 80 words and ≥ 3 neighbours. The gap is
smaller than for papers (5 vs 45–52 references). Evaluation over-represents "Growing Up &
Facts of Life" (38% vs 32% overall) and under-represents "Activities, Crafts & Games" (2% vs
5%), which have short descriptions.

## Step 3: embeddings (2026-10-21)

`embed_children.py` is a script version of `notebooks/adapter/embed_nodes.ipynb`, with the
same settings as the papers: voyage-4-large, `input_type="document"`, 128 texts per request.
The text embedded is **`title + "\n\n" + description`**, as papers embedded `title + "\n\n" +
abstract`. The generation target stays the description alone.

```bash
# VOYAGE_API_KEY in the repository's .env
cd src/data/children_books && python embed_children.py && python prepare_sets.py
```

- **Output:** `data/children_books/node_embeddings.npy`, float32 (76,820 × 1024). It is
  written to a memory map with a progress file, so an interrupted run resumes.
- **Size:** about 15.2M words, roughly 20M tokens. The largest batch is about 89k tokens.
- **Very long descriptions:** 9 have more than 5,000 words. Voyage truncates at the model's
  context length.

**Run 2026-10-21:** 76,820 rows embedded in 8.1 min with no retries. Output (76,820 × 1024),
no zero rows, every row unit norm. Re-running `prepare_sets.py` removed **38** more training
books (voyage cosine > 0.95 to an evaluation or validation book), leaving **52,274** for
training. Evaluation (945) and validation (806) are unchanged, because they are drawn before
the filter. `training_sets.json` now records `"cosine_filtered": true`.

## Step 4: does the GNN have room to beat the mean? (2026-10-21)

### Why

On papers, run 11's GNN beat the plain neighbour mean by only +0.010 cosine. The argument for
books was that their neighbours might hold information a mean hides. That can happen in two
ways:
- **Selection:** a few neighbours are informative, and attention could pick them out.
- **Translation:** the neighbours point at the book without resembling it, and a learned map
  could get from one to the other.

This step measures both on CPU, before any GPU time.

### How

```bash
cd src/data/children_books
python neighbour_check.py --dataset children
python neighbour_check.py --dataset papers
```

For each of the 945 evaluation nodes, the script compares the neighbours' voyage embeddings
with the node's own:
- **The rows of `docs/thesis_context/06_results.md` §6.2:** the mean, the best single
  neighbour, the mean of the 3 neighbours closest to the answer, and the best non-negative
  weighting fitted to the answer (NNLS).
- **Plus a linear map:** ridge regression from mean to own embedding, fitted on the training
  nodes. Its regulariser is chosen on validation. This is the "mean + linear head" baseline.
- **A control for each fit:** the same fit against another evaluation node's embedding.

The **selection room** is how much more NNLS gains over the mean for the real node than for
the control. The **translation room** is the linear map's gain over the mean on held-out
nodes. The paper run reproduces §6.2 exactly (0.823 / 0.814 / 0.853 / 0.894, control 0.497 →
0.583), which validates the script.

### Results

| estimate (945 eval nodes, cosine to own embedding) | papers | children's books |
|---|---|---|
| eval degree, median | 52 | 20 |
| plain mean | **0.823** (top-1 64.3%) | **0.725** (top-1 45.4%) |
| best single neighbour (chosen with the answer) | 0.814 | 0.719 |
| closest-3 mean (chosen with the answer) | 0.853 | 0.765 |
| NNLS (fitted to the answer) | 0.894 | 0.803 |
| control: mean → NNLS against an unrelated node | 0.497 → 0.583 | 0.483 → 0.548 |
| **selection room** (real NNLS gain − control gain) | **−0.016** (SE 0.002) | **+0.013** (SE 0.002) |
| linear map (fitted on train) | 0.826 (top-1 65.5%) | 0.731 (top-1 46.9%) |
| **translation room** (linear map − mean) | **+0.003** (SE 0.000) | **+0.006** (SE 0.001) |
| run-11 GNN (GATv2), for reference | 0.833 | not trained |

### Reading

- **The neighbours say much less about a book than references say about a paper.** The
  mean's cosine is 0.725 against 0.823, and top-1 is 45% against 64%. The reader will get a
  weaker input, so expect lower absolute generation quality than on papers.
- **There is a little room for selection on books and none on papers.** On books, fitting
  weights to the answer gains 0.013 more than chance fitting does. On papers it gains less
  than chance. The difference is clear (SE 0.002), but small.
- **Translation room is small on both.** A linear map adds +0.006 on books and +0.003 on
  papers. Co-purchased children's books are mostly similar books, not complements, so there
  is little to translate.
- **What to expect from the GNN.** On papers, GATv2 gained +0.010 over the mean, more than
  either room predicts; part of that is the linear head and input-dependent weighting
  together. On books, both rooms are larger, so expect a GNN margin somewhat above papers'
  +0.010, but still a few hundredths of cosine, not tenths. Books-Children is a
  generalisation test, not the dataset where the GNN wins big. That remains the products
  hypothesis.

### Limitations

- NNLS fitted to the answer is an upper bound on non-negative weighting only. A GNN also
  applies a learned transform, so it can exceed what NNLS's room suggests, as GATv2 did on
  papers.
- The control's fitting freedom grows with degree (books 20 neighbours, papers 52). The room
  is computed per node against its own control, so this is accounted for within a dataset,
  but cross-dataset comparisons of the raw gains are rough.

## Step 5: the training code reads the books (2026-10-21)

### How to switch

```bash
export DYTAG_DATASET=children          # unset, or "papers", is the citation pipeline as before
cd src/adapter
python pretrain_encoder.py --sets children --features raw ...
python train.py           --sets children --features raw ...
python generate.py        --sets children --features raw ...
```

**One environment variable, not a `--dataset` flag.** Every path is a module-level constant in
`config.py`, imported by all the scripts, so one switch there moves them all. A flag would have
had to be threaded through train, generate, pretrain and the verify scripts. `--sets children`
must go with `DYTAG_DATASET=children` and no other set with it. `config.check_sets` makes
train, generate and pretrain refuse any mismatch, so a book checkpoint cannot be scored on
paper nodes or the reverse.

### What `DYTAG_DATASET=children` changes

| | papers | children's books |
|---|---|---|
| `JOINT_DIR` (nodes, edges, embeddings) | `data/processed/joint/` | `data/children_books/` |
| `FROZEN_DIR` | `data/frozen/` | `data/frozen/children_books/` |
| node sets (`--sets`) | `run9` → `training_sets_run9.json`, `eval_nodes_capped.json` | `children` → `training_sets.json`, `eval_nodes.json` |
| target cleaning (`CLEAN_TARGETS`) | LaTeX stripper on | **off**: it strips `$` from prices |
| `MAX_NEW_TOKENS` | 320 | **1,024** |
| prompt (`text.GRAPH_ONLY_SYSTEM`) | paper abstract | book description, see below |
| id column in `generations.csv` (`SOURCE_ID`) | `mag_id` | `cstag_id` |

The output columns stay `original_abstract` and `generated_abstract`, so `loop_check.py` and
the evaluators read book runs unchanged.

### The book prompt

It follows the paper prompt sentence by sentence: what the context encodes, what to
reconstruct, what to do when the context is undetermined, voice, format. It covers the kind of
book and its readers (format, age), its subject, characters or story, and what makes it worth
reading. It never mentions the co-purchased books. The user turn is "Write the description."
There is **no word count**: descriptions run from 80 to over 700 words, while abstracts sit
near 170. The length is left to training. The prompt is 218 tokens, against 241 for papers.

### The 1,024-token cap, and step 2 re-run

Half the evaluation descriptions are longer than 320 Qwen tokens (median 317, p95 901, max
4,263 before the cap), so the paper cap would have cut them. 1,024 covers 96% of evaluation and
98% of training descriptions. A non-stopping generation then costs about 50 s per book
instead of about 15 s.

**Every target must fit the cap**, in all three sets: description + EOS ≤ 1,024 tokens
(`prepare_sets.py`, `MAX_TARGET_TOKENS`). A longer target could not be generated in full, and
the longest (27,579 tokens) would not fit the 4,096-token context. This drops 1,303 books
(every one has ≥ 80 words), and the sets were redrawn. `prepare_sets.py` now needs the Qwen3
tokenizer, so it runs in the `DyTAG-qwen` env.

| | before the cap | with the cap |
|---|---|---|
| eligible for eval/valid | 33,084 | 31,996 |
| evaluation / validation | 945 / 806 | 945 / 806 (redrawn) |
| training | 52,274 | **50,999** |
| eval words / tokens, median | 235 / 317 | 222 / 306 (max 1,014) |
| longest training sequence incl. prefix | — | 1,274 tokens |

Step 4 on the redrawn evaluation set: mean 0.734 (top-1 48.1%), NNLS 0.808, control 0.490 →
0.554. **Selection room +0.010** (SE 0.002), **translation room +0.005** (SE 0.001). The
conclusions are unchanged.

### Verified (locally)

- `verify_graph_only_prompt.py` passes for both datasets (`DYTAG_DATASET=children` and
  unset). Every book target is non-empty, masked correctly, and fits the cap and the context.
  It now reads the book sets and checks the cap for books.
- `verify_decoder_inputs.py`, `verify_sliced_loss.py`, `verify_lora_prefix.py` and
  `verify_generation.py` pass under `children`. They use tiny models and no dataset files.
- **A 1-epoch GNN run on books** (`pretrain_encoder.py --sets children`, CPU, 55 s): 50,993
  training nodes, refs-mean baseline 0.7253 valid / 0.7339 eval. The eval figure matches
  step 4. After one epoch the GNN is at 0.708, below the mean, as papers were at epoch 1.
- `--sets run9` with `DYTAG_DATASET=children` is refused by `check_sets`.
- `train.py` and `generate.py` import with the book settings, and the paper defaults are
  unchanged. They cannot be run end to end locally without loading Qwen3-14B.

### Not done here

- **The text baseline.** It came from the old text path (neighbour texts in the prompt,
  centroid-capped, frozen under the Vicuna tokenizer). `generate.py --mode baseline` today
  is frozen Qwen with the graph-only prompt. Rebuilding the text arm for books is its own
  step.
- **The job script** (step 6).

## Step 5b: the text arm (2026-10-21)

**Why.** Run 11's comparison arm on papers was frozen Qwen reading the references as text. The
books need the same comparison, in the same job, because every queue wait on Komondor can take
days.

**Same prompt, evidence as text.** The papers' 09-10 text baseline used the older era-1 wording,
which differed from the graph prompt, and `docs/soft_prompt_only.md` measured that confound as
large. The books' text arm avoids it. Its system prompt (`text.BOOK_TEXT_SYSTEM`) is the book
graph prompt with "the context you have been given encodes" re-pointed at "the books listed
below". The user turn lists the neighbours (`- Title: … / Description: …`), then "Write the
description." The evidence is the only difference between the two arms.

**Which neighbours are listed** is frozen in `data/frozen/children_books/prompt_refs_text.json`
by `src/adapter/freeze_book_refs.py`. It uses the selection the papers' text arm used
(`neighbour_cap.fit_refs`): the 15 neighbours closest to the neighbourhood centroid, with the
farthest dropped until the prompt fits. The budget is the one `generate.py` checks, 4,096 − 32
prefix − 1,024 generation = 3,040 prompt tokens, under the Qwen3 tokenizer. No frozen prompt can
produce an ERROR row. It covers only the 945 evaluation books, since nothing is trained on it.

```bash
cd src/adapter && DYTAG_DATASET=children python freeze_book_refs.py    # DyTAG-qwen env
DYTAG_DATASET=children python generate.py --mode text --sets children --no-repeat-ngram 6 --output …
```

| | |
|---|---|
| neighbours listed | mean 8.65, median 8 (min 2, max 15) of a median degree of 18 |
| all neighbours listed | 30.9% of books |
| prompt tokens | median 2,650, max 3,040 (= budget) |

**The graph arm reads every neighbour; the text arm reads about half.** This is the same
asymmetry as on papers, where the text arm saw a median 27% of the references.

## Step 6: the job script (2026-10-21)

`hpc/komondor/jobs/run11_children.sh` is `run11_all.sh` with the dataset swapped:
`SINGULARITYENV_DYTAG_DATASET=children`, `--sets children`, checkpoints in
`checkpoints/run11_children/`, experiments named `<tag>_run11-children-*`, W&B runs `run11c-*`.
The recipe is unchanged: the 8-variant GNN grid, the reader at 6 × 15,000 with refs-mix 0.5,
and the joint fine-tune at 3 × 15,000. 6 × 15,000 is about 1.8 passes over the 50,999 training
books, where run 11 saw its 90k papers about once. I kept the steps rather than the passes:
run 11's reader was still improving at epoch 6, so more passes should not hurt, and the
schedule stays identical.

**Two changes from run 11, both to fit one queue wait:**

1. **Arms ordered by importance:** refs-mean and GNN (the comparison), text, zero, oracle, then
   the joint fine-tune and the joint arm. A timeout then costs the least needed parts. The
   joint stage goes last because on papers it added nothing measurable over reader ← GNN.
2. **No loop test.** The papers' decision (`--no-repeat-ngram 6`: loops 14.0% → 0%, top-1 −0.5
   points) is reused, which saves about 6 h at book lengths.

**Walltime: 72 h may not be enough.** An arm took about 7 h on papers at roughly 150 generated
tokens. Book descriptions are about twice as long, so expect about 14 h per arm, or about 70 h
for the five long arms on top of about 19 h of training. Resubmitting resumes every phase and
every arm, so a timeout costs a queue wait, not work.

**To run:**

```bash
# laptop: send only what the job needs (no --delete)
rsync -avh --progress --relative src/ hpc/ data/children_books/ data/frozen/children_books/ \
      vn01:/project/c_gnn42/csanad_thesis/baseline/
# Komondor
sbatch hpc/komondor/jobs/run11_children.sh
```

The container (`adapter-qwen-gpu.sif`), the Qwen3-14B cache and `.env` are already on Komondor
from run 11.

## Step 6b — the GNN grid ran out of memory (job 14618627)

The first Komondor job failed in phase 2 with `CUDA out of memory` in `pretrain_encoder.py`
(`loss.backward()`, 38.7 of 39.5 GB in use). The 1-layer variants trained; the 2-layer GATv2
did not.

**Cause:** the encoder reads the *full* neighbourhood at every hop (`num_neighbors = [-1] *
layers`, no sampling). The co-purchase graph is much denser than the citation graph (mean
degree 30.2, max 2,753), so the 2-hop subgraph of a batch reaches most of the graph. Measured
on 30 random batches of training books:

| seeds per batch | nodes in subgraph | edges, mean | edges, max |
|---:|---:|---:|---:|
| 256 (run 11 default) | 49,751 | 681,207 | 810,712 |
| 64 | 36,676 | 242,584 | 336,560 |
| 32 | 28,412 | 129,119 | 252,616 |
| 16 | 18,799 | 63,218 | 150,455 |

**Fix:** `--batch-size 32` for every variant in the grid, so they stay comparable. The
worst-case batch is about a third of the default's typical batch. This keeps every neighbour
rather than sampling the second hop, which would have changed what the GNN sees compared with
the papers. Costs: about 8× more steps per epoch and about 1.5× more edge computation, and
InfoNCE has 31 in-batch negatives instead of 255. That matters little next to the 4,096
sampled negatives, and only the `cosnce` variants use InfoNCE at all.

Resubmitting resumes: variants with a checkpoint in `checkpoints/run11_children/gnn/` are
skipped. The failed variant crashed in its first epoch, before any checkpoint was written.

## Step 6c — restart from zero, with no deliberate stops (after job 14626133)

**What happened.** Job 14626133 trained the GNN grid and two reader epochs, about 10 h in all.
Then the identification stop rule ended it. From its own embedding, the reader picked out 0.062
of 128 validation books by TF-IDF (8 of 128), against a floor of 10× chance (0.078). The same
log shows that the vector is being read:

- validation loss is 2.3998 with the vector and 2.4270 without it, a gain of 0.027 nats (the
  papers recipe gained 0.022 at the same stage);
- oracle MRR is 0.151, against roughly 0.04 by chance.

The threshold was set on paper abstracts, whose technical vocabulary makes a text easy to
match. Book descriptions share genre wording and often open generically, and the check reads
only the first 96 generated tokens. Training is seeded, so resubmitting without a change would
have stopped at the same point.

**Decision:** keep the papers recipe and restart the whole run from zero. The failed run's
folder is renamed, not deleted: `checkpoints/run11_children_stopped_14626133`.

**Changes:**

1. `train.py --stop-rules warn`. Both reader rules (identification and canary) are still
   computed every epoch. A failure is printed and appended to `reader/WARNINGS.txt`, and
   training continues. The default stays `stop`, so the papers pipeline is unchanged. Model
   selection (`best.pt` = highest validation refs-mean MRR) is untouched. The papers run passed
   both rules, so making them warnings changes nothing about comparability.
2. **Phase 3 leaves out very large subgraphs.** The cap is
   `config.JOINT_MAX_SUBGRAPH_EDGES = 60_000` for the books (`None` for the papers), applied
   through `graph.within_subgraph`. The joint fine-tune backpropagates through the GNN next to
   Qwen3-14B. The largest 2-hop subgraph of a training book has 223k edges, about the size of
   the largest batch-32 subgraph in the GNN grid. The cap removes 108 of 50,993 training books
   (0.21%) when the winner has 2 layers, and none when it has 1. Evaluation books are not
   filtered (max 85k edges, run under `no_grad`).
3. **The job no longer exits on a failed evaluation arm.** The failure is recorded, the next
   arm runs, and the job exits 1 at the end with the list of failed arms. Resubmitting
   re-runs only their missing rows. The `STOPPED` file and its refusal guard are gone. The CPU
   checks still exit at once, since they cost no GPU time.

The prompts are unchanged. The reader learns the output style from the targets through LoRA,
and a prompt more different from the papers' would cost comparability.

**Deviations from the papers run, all forced by the data:** GNN grid batch size 32 (step 6b),
`MAX_NEW_TOKENS` 1024, no-repeat n-gram 6 reused without a loop test, stop rules as warnings,
and the phase-3 subgraph cap.
