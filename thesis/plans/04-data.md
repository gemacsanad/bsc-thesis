# Chapter 4 — Data (detailed plan)

**DRAFT v3 2026-10-30, not yet approved.** File `content/04-data.tex`, title "Data".
Sources: `thesis_context/02_data.md` (numbers, with its section), `01_literature.md` (bib URLs),
and where stated `../docs/step8_representation_plan.md` §4.1 and `../docs/soft_prompt_only.md`
(step 8, cleaning). Numbers marked [recomputed] in `02_data.md` were recomputed from
`data/processed/joint/` on 2026-10-13. **[new]** = needs a small CPU analysis (separate task,
the author decides).

## Decisions (author, 2026-10-30)

- **D1 one dataset.** Only the citation graph; the children's books are not planned for the
  thesis now (future work, Ch. 10). No reserved section.
- **D2 data exploration, no design history.** The chapter is about the data: what it contains,
  its limits, its properties. Earlier or reverted designs are not told.
- **D3 no text baseline here.** Its reference selection and the information budget go to Ch. 6.
- **D4 figures after the text.** Candidates are listed below and chosen once the text is written.
- **D5 tools.** No word2vec at all. DuckDB only if really necessary; otherwise "SQL queries".
- **D6 order of the argument.** First the raw ogbn-arxiv numbers, to show that it lacks
  references; then the enrichment.
- **D9 only what the final model uses.** Dropped: the early LLM relevance labels and filtering,
  ZCA whitening, earlier node sets, the reverted undirected graph.
- **D10 no time or artefact exploration.** The year is not a model input: no years per split, no
  reference age. Time appears only as the reason for using references (4.3) and as the split
  criterion (4.6). Mutual citations are not discussed. The enrichment is described by its
  matching logic, not by how the data was stored.
- **D11 review applied (2026-10-30):**
  - order: ogbn-arxiv → enrichment → the graph used by the model → exploration → embeddings →
    splits and node sets (with the reference-mean analysis at its end) → cleaning; the
    reference-mean analysis comes after the evaluation set is defined;
  - the 17.0 vs 6.9 comparison is in 4.2, not 4.1;
  - wording: "self-loops", not "self-citations"; never "full reference list" / "full
    bibliography is known" (Semantic Scholar adds citation records, subject to its coverage and
    the MAG-id and abstract requirements); "no outgoing citations recorded in ogbn-arxiv", not
    "no references at all";
  - training-set arithmetic: only the stubs and near-duplicates reduce 90,941 to 90,461
    (validation and evaluation papers are in other temporal splits; the non-overlap is a
    separate check);
  - the degree mismatch is explained by the sampling procedure first;
  - make explicit which papers can be targets (core) and what added papers are for
    (neighbourhood information), and that the splits divide targets, not the graph;
  - trim bookkeeping (intermediate 314,224 / 2,986,960 counts, id ordering, detailed integrity
    checks, overlapping degree plots); keep explanations;
  - the example paper is in, as an illustration (not claimed to be representative);
  - in-degree gets less space than out-degree;
  - the embedding analysis is interpreted carefully (random baseline, what 64.3% means,
    similarity ≠ full information).

**Not here:** the formal task (Ch. 3; one-line forward reference where needed), target masking
with `[MASK]` (Ch. 5), the text baseline's input (Ch. 6), evaluation instruments (Ch. 6).

**Budget ≈ 5–6 pages** including figures and tables.

---

### 4.1 ogbn-arxiv and its limitations (≈ 0.7 page)

- **P1 the dataset.** 169,343 computer-science arXiv papers and 1,166,243 directed citation
  edges, built from the Microsoft Academic Graph; each paper has a MAG id, title, abstract and
  year. Its subject labels are not used. Cite: Hu et al. 2020; Wang et al. 2020. `02_data` §2.1.
- **P2 its recorded references.** Out-degree mean 6.89, median 4; 17,440 papers (10%) have no
  outgoing citations recorded in ogbn-arxiv; only 499 have more than 50. `02_data` §2.1, §2.3.
- **P3 why this is too little (from the restricted coverage alone).** ogbn-arxiv records only
  citations between papers in the dataset, so a reference to a journal or conference paper not
  on arXiv, a book, or a paper from another field is missing. A low recorded out-degree is
  therefore mostly a property of the dataset, not of the paper. For a task that writes a paper
  from its references (forward reference to Ch. 3), most of the evidence would be missing → more
  citation records are needed. No enrichment results here (D11).

### 4.2 Enrichment with Semantic Scholar (≈ 1 page)

- **P1 the matching logic.** Every ogbn-arxiv paper has a MAG id, and Semantic Scholar records
  the MAG id of its papers, so each ogbn-arxiv paper is matched to its Semantic Scholar record
  through it. Semantic Scholar provides additional citation records; a referenced paper is added
  when it has the identifier and the abstract the pipeline requires. Joined with SQL queries (D5).
  Cite: Kinney et al. 2023. No intermediate counts, sizes or file names. `02_data` §2.2.
- **P2 the two kinds of papers.** *Core papers* are the original ogbn-arxiv papers; only they
  are prediction targets. *Added papers* come from Semantic Scholar and serve only as additional
  neighbourhood information (they are references of core papers). Where a paper exists in both
  sources, the ogbn-arxiv text and year are kept. Edge direction checked to agree in both
  sources (one clause). Result: 384,756 nodes (169,343 core + 215,413 added), 3,844,151
  directed edges. `02_data` §2.3.
- **P3 what changed** (candidate Tab. 4.1 before/after): core out-degree mean 6.89 → 12.55,
  median 4 → 8; core papers without recorded references 17,440 → 8,675; with > 50 references
  499 → 4,284; 70,088 new citations between core papers that ogbn-arxiv lacked; 889,024
  citations to added papers.
- **P4 the coverage limitation.** Semantic Scholar contributes citation records for 94,756 core
  papers (56%), whose mean out-degree is 17.0; the other 74,587 (44%) keep only their ogbn-arxiv
  citations, mean 6.9. So an observed low out-degree does not necessarily mean a short
  bibliography: it can also mean an unobserved one, and the two cannot be told apart for a given
  paper. Measured, not removed: filtering would change the population the results describe.
  Forward references to Ch. 7 (per-degree results) and Ch. 9. Coverage by degree: O4.
  `02_data` §2.4.

**4.2 as written (2026-10-30, second review applied):** two paragraphs (matching via Corpus ID inside Semantic Scholar, MAG id to ogbn-arxiv; then the four expansion steps, all ogbn-arxiv papers and edges kept, not recursive), roles of ogbn-arxiv vs added papers, Tab. 4.1 (`tab:enrichment`), coverage paragraph. Figures: `fig:enrichment` (draw.io, `../diagrams/figures/data/enrichment.*`, two panels) and `fig:enrichment-degree` (`../notebooks/eda/thesis_enrichment.ipynb` → `../diagrams/figures/data/enrichment_degree.pdf`). Coverage definition verified in the notebook: covered = Semantic Scholar added ≥ 1 citation (94,756); 98,811 have a usable record, 4,055 of them gained no citation.

### 4.3 The graph used by the model (≈ 0.5 page)

*Written 2026-11-02 (as in `content/04-data.tex`, `sec:model-graph`); O6 and O7 resolved below.*

- **P1 what an edge means.** An edge runs from a cited paper to the citing paper, and messages
  flow along it: a paper is described only by the works it cites, never by the papers that cite
  it. Reason: when an abstract is written, its references exist, but no one has cited the paper
  yet; citing papers would be information from the future. Builds on Fig. 2.1.
- **P2 one graph, split targets.** All experiments use the same joint graph. The training,
  validation and evaluation splits (4.6) divide the *prediction targets*, not the graph: any
  paper in the graph, core or added, can supply reference information to a target that cites
  it. How the target's own text and embedding are hidden is left to Ch. 5. (To verify before
  writing, O7: whether an evaluation paper can appear as a reference of another target.)
- **P3 details used by the final model.** The 20 self-loops (a paper citing itself) are
  removed; all references are used at every hop, with no sampling. Neighbourhood sizes: O6.

### 4.4 Exploring the joint graph (≈ 0.8 page)

*Written 2026-11-02 (`sec:exploration`, `tab:example`). O1 resolved: author chose a cherry-picked example, Hacohen & Weinshall 2019 (node 46116, eval set), stated as chosen. O8: author decided not to mention the empty abstract.*

- **P1 out-degree** (main): distribution after enrichment, heavy-tailed, max 832 [recomputed].
  The before/after comparison is in 4.2 (one plot there, not several overlapping ones).
- **P2 in-degree** (short, context only): a few very highly cited papers, max 16,504
  [recomputed]; one or two sentences.
- **P3 abstracts** (one paragraph or a compact table): core median 157 words, mean 160.9,
  p10 91, p90 237; added papers similar (median 157). The single empty abstract: embedded as a
  single space (`02_data` §3); whether it is a core paper / a target: O8.
- **P4 one example.** One evaluation paper: its abstract (shortened) and the titles of some of
  its references. Shows that the references establish the topic and the prior work, while the
  abstract also states the paper's own contribution, which no reference contains. Presented as
  an illustration, not as representative. Selection: O1.

### 4.5 Node text and embeddings (≈ 0.5 page) — approved 2026-11-02

Explain for a reader who has not met them: what an embedding model is, and what similarity
between two vectors (cosine) means. Ch. 2 uses both without defining them.

- **P1 embedding models and similarity.** An embedding model maps a text to a fixed-length
  vector so that texts with similar meaning get vectors pointing in similar directions; cosine
  similarity = cosine of the angle (dot product for unit vectors), 1 = same direction.
- **P2 the representation.** Every paper (ogbn-arxiv and added) → `title + "\n\n" + abstract`
  embedded with `voyage-4-large`, document input type, 1024-d, unit length. Why a sentence-level
  embedding: the model has to recover what a paper is about. Raw, uncleaned text (link to 4.7).
  `02_data` §3. Cite `voyage2026voyage4` (blog, 15 Jan 2026) and `voyage2026embeddings` (docs,
  input type, dims).
- **P3 role.** As a neighbour, a paper's embedding is the information it contributes to a
  target. (Author: the GNN estimating the target's embedding is left to the GNN part, Ch. 5.)
- **P4 geometry (O2 resolved: here).** Unrelated papers are already fairly similar: random pairs
  of papers have median cosine 0.294 (5th–95th percentile 0.173–0.440; 100,000 random pairs,
  raw embeddings, recomputed 2026-11-02; confirmed by all-pairs check on 2,000 papers: 0.297). Needed to read the 0.823 in 4.6.
- **Not here (author):** the title in the target's embedding / oracle consequence (→ Ch. 6);
  whitening (not used by run 11: `--features raw` in `run11_all.sh`); word2vec; storage; empty
  abstract (O8 dropped).

### 4.6 Splits and node sets (≈ 1 page)

*Written 2026-11-02 (`sec:splits`, `fig:reference-mean-cosine`, notebook `notebooks/eda/thesis_splits.ipynb`). Author: eval-set sampling justified by analysing generation across a broad range of neighbourhood sizes; baseline = reference means of the other 944 eval papers (mean 0.496; valid 0.488); NOT included: cross-set citations, a table of the sets, the 64.3% retrieval number.*

- **Moved here from 4.3 (review 2026-11-02): cross-set citations.** 17 of the 90,461 training
  targets cite ≥ 1 of the 945 eval papers (14 distinct), 105 cite ≥ 1 of the 806 validation
  papers (63 distinct) [recomputed 2026-11-02]. Training papers (≤ 2017) citing 2019–2020 papers
  means the reversed edge direction alone does not make the graph strictly chronological. Cause
  not established (metadata dates, later paper versions or data errors are possibilities only);
  distinguish the intended temporal rationale from the limitations of the actual graph. Here or
  in a short temporal-limitations discussion (Ch. 9).

- **P1 the split.** By year, from ogbn-arxiv: training ≤ 2017 (90,941 core papers), validation
  2018, test 2019–2020 [recomputed]; this separates training from evaluation in time. Targets
  are core papers only (4.2).
- **P2 evaluation set.** 945 papers from the test split (2019: 750, 2020: 195), out-degree 5–100
  with at most 10 per degree, drawn with a fixed seed; out-degree mean 51.8, median 52; gold
  abstracts median 170 words. Why stratified by degree: to measure how results depend on the
  number of references. `02_data` §5.1.
- **P3 validation set.** 806 papers from 2018, out-degree median 45 (drawn with the same per-degree
  quotas; check wording against `02_data` §5.1 when writing).
- **P4 training set.** Removing 477 short abstracts (< 30 words) and 3 near-duplicates of
  evaluation papers (same normalised title or embedding cosine > 0.95) from the 90,941
  training-split papers leaves 90,461. Separately: the three sets are checked not to overlap.
  The GNN is trained on the 83,259 with at least one recorded reference. `02_data` §5.2–5.3.
- **P5 degree mismatch, explained by the sampling.** The evaluation set deliberately spreads its
  papers over out-degrees 5–100 (≤ 10 each), while the training set keeps almost every
  training-split paper, so it follows the natural, low-degree distribution (median 5, mean 7.9
  vs evaluation mean 51.8). Secondary: high-degree papers are rare in the training split. Used
  again in Ch. 7 and 9. One sentence: all sets are frozen to files.
- **P6 how close the references are to a paper (moved here, D11).** On the 945 evaluation
  papers, the plain mean of a paper's reference embeddings has cosine 0.823 to the paper's own
  embedding (0.816 on validation). Interpretation needs:
  - a baseline: the cosine to the reference mean of a random other paper [new; the pairwise
    cosine of reference means, 0.66, already shows that unrelated embeddings are similar];
  - one plain sentence that unrelated embeddings in this space also have substantial
    similarity (anisotropy, O2);
  - what the retrieval number means: the reference mean identifies the paper's own embedding
    among the 945 candidates at 64.3% top-1 (96.0% top-10); this is retrieval of an embedding,
    not generation of the abstract;
  - framed as evidence of semantic similarity that motivates recovering a paper's content from
    its references, not as evidence that the references contain every detail of the target.
  `02_data` §3.1, §8. Candidate figure: cosine(paper, own reference mean) vs random [new].

### 4.7 Cleaning the training targets (≈ 0.4 page)

- **P1 why.** The instruction to the LLM asks for no markup or mathematical notation, but in an
  audit of training abstracts 13.5% contained inline mathematics (`$…$`), and others contained
  LaTeX commands (7.0%) or super-/subscripts (6.2%): instruction and supervision disagreed.
  Source: `../docs/step8_representation_plan.md` §4.1 (audit of the 7,750 papers of an earlier
  training set; state the sample or recompute, O9).
- **P2 what.** Formatting commands unwrapped, math delimiters removed, accents resolved, command
  names kept without the backslash (one example: `$\Omega(N \log\log N)$` →
  `Omega (N log log N)`). Effect: the share of abstracts with any remaining markup fell from
  ≈ 18% to 1.0% on 40,000 abstracts (`soft_prompt_only.md`). The two numbers differ because
  13.5% counts inline maths only, ≈ 18% counts any markup; say so, or report only one (O9).
- **P3 scope.** Training targets are cleaned; evaluation gold abstracts are not, so scores stay
  comparable across runs; the embeddings use the raw text. `02_data` §7.3.

---

### Candidate figures and tables (chosen after the text, D4)

Diagram in draw.io with the creator/reviewer loop of Ch. 2; plots in matplotlib with
`../diagrams/thesis_style.py`. [new] analyses are CPU-only.

| candidate | § | shows |
|---|---|---|
| diagram: core and added papers | 4.2 | ogbn-arxiv papers matched to Semantic Scholar via the MAG id; added papers as references of core papers; core = targets, added = neighbourhood only |
| out-degree of core papers before vs after enrichment (+ split by coverage) | 4.2 | the gain and the coverage limitation in one plot |
| Tab.: ogbn-arxiv vs joint graph | 4.2 | the enrichment in numbers |
| example paper + reference titles (table or box) | 4.4 | references give topic and prior work; the abstract adds the contribution |
| out-degree per node set + Tab. of the sets | 4.6 | the mismatch caused by the sampling; one place to look up the sets |
| cosine(paper, own reference mean) vs random | 4.6 | similarity with its baseline |

### Citations needed (none in `mybib.bib` yet)

| key (proposed) | record | for |
|---|---|---|
| `hu2020open` | Hu et al., NeurIPS 2020, Open Graph Benchmark (NeurIPS bib) | ogbn-arxiv |
| `wang2020microsoft` | Wang et al., Quantitative Science Studies 2020, Microsoft Academic Graph (Crossref) | MAG ids |
| `kinney2023semantic` | Kinney et al. 2023, The Semantic Scholar Open Data Platform (arXiv 2301.10140) | Semantic Scholar data |
| `voyage2025voyage4` | Voyage AI documentation, `voyage-4-large` (web, owner + access date) | embeddings |

### Terms introduced (glossary, `00_index.md`)

core paper, added paper, joint graph, Semantic Scholar coverage, reference mean, evaluation /
validation / training set, gold abstract.

### Still open

- **O1 example selection.** Random with a fixed seed (honest, may be a dull example) or chosen by
  the author for clarity (stated as chosen)? Table or box?
- **O2 anisotropy.** One plain sentence in 4.6 P6 (needed to read the 0.823), or more?
- **O4 coverage by degree.** The 31% → 64% numbers come from an early evaluation set. Recompute
  for the final sets [new] or leave out.
- **O6 neighbourhood sizes — resolved 2026-11-02.** The final GNN has one layer, so a target's
  neighbourhood is exactly its recorded references; its size is the out-degree (reported in 4.4 /
  4.6). The early 2-hop numbers are not used. (Recomputed for reference: 2-hop within-reach nodes,
  eval mean 598.9, train median 25.)
- ~~O6 (old)~~ Measured on early out-degree 5–15 seeds. Re-measure on the final
  sets [new] or leave out.
- **O7 graph protocol check — resolved 2026-11-02.** Yes: on the joint graph without self-loops,
  17 of the 90,461 training targets cite ≥ 1 of the 945 eval papers (14 distinct eval papers),
  105 cite ≥ 1 of the 806 validation papers (63 distinct). Their abstracts are never training
  targets. The seed [MASK] (`model.py`) keeps the target's own row out of its input. Stated in 4.3.
- ~~O7 (old)~~ Verify in the code / `verify_decoder_inputs.py` whether an
  evaluation or validation paper can be a reference of another target, and state it in 4.3 P2.
- **O8 the empty abstract.** Find which node it is; if it is a core training paper, the stub
  filter removes it as a target (say so); otherwise it is only a neighbour embedded as a space.
- **O9 cleaning numbers.** The 13.5% audit was on an earlier 7,750-paper training set; recompute
  on the 90,461 [new], or state the sample; report one or both numbers.
- **Conflict:** max in-degree 16,504 ([recomputed], `02_data` §2.3) vs 16,513 (`02_data` §4).
  Plan uses 16,504.
