# DyTAG — an introduction to the project

*A guided overview of what this research project does, how it was built week by week, and what it
has found. Written for a technical reader who does not work on graphs or language models. For the
full engineering record see [`README.md`](README.md) and the stage documents it indexes.*

**Status:** 2026-09-07. Ten weeks of work, from 2026-06-30 to now.

---

## 1. The question

Every scientific paper carries a bibliography — the list of papers it builds on. That bibliography
is not chosen at random. It is a fingerprint: of the problem the paper attacks, the methods it
borrows, and the research community it belongs to.

**The question this project asks is: how much of a paper can you recover from its bibliography
alone?**

Concretely the task is: *given only the titles and abstracts of the papers that paper X cites,
write X's abstract.* X's own title is never shown. If a model can do this, then a citation graph
carries real semantic content about its nodes — which matters for anything built on academic
graphs: recommendation, trend detection, automated literature review, filling in missing metadata.

The second question is a *methods* question, and it is what most of the engineering serves:

> Does giving a language model the **graph structure** around a paper help it, beyond simply giving
> it the text of the cited papers?

That distinction is the whole experiment. A language model can already read a list of abstracts.
The claim under test is that a graph neural network — which sees the *whole* neighbourhood,
including references that never fit in the prompt, and the neighbours of those neighbours — adds
something the text alone does not carry.

The experiment is therefore always a **difference between two arms**, never an absolute score:

| arm | condition |
|---|---|
| baseline | the frozen language model, no graph |
| adapter | the same frozen model, with a graph-derived prefix |

Everything else is held identical. The number that matters is the *gap*.

---

## 2. The data

Two datasets, joined.

**ogbn-arxiv** is a standard public benchmark: ~169,000 computer-science arXiv papers, citation
edges between them, and a fixed train/validation/test split. Its limitation is that it records only
citations *between arXiv CS papers* — everything else a paper cites is invisible.

**Semantic Scholar** was downloaded in bulk to fill that gap with papers' real reference lists.
Three of its datasets were needed (`papers` for the ID mapping, `abstracts` for text, `citations`
for edges), read out-of-core with DuckDB because the raw shards do not fit in memory.

Joining them gives the **joint dataset**:

| | |
|---|---|
| nodes | 384,756 |
| directed edges | 3,844,151 |
| "core" nodes (original ogbn-arxiv, with a known split) | 169,343 |
| node features | 1,024-d Voyage AI (`voyage-4-large`) embedding of each abstract |

Voyage embeddings replaced the OGB-supplied 128-d word2vec features, which average word vectors and
lose sentence structure — a poor input for a model that has to judge what a paper is *about*.

**One measurement here shapes everything downstream.** Only 56% of core nodes have a real Semantic
Scholar bibliography; the other 44% have only their arXiv-CS citations. The two groups have very
different mean out-degree (17.0 vs 6.9). So "this paper cites 8 works" means either *a short
bibliography* or *an incompletely observed one*, and it is not knowable which. This confound was
deliberately **measured rather than removed** — filtering it out would have silently changed what
population the results describe — and it is printed alongside every frozen node set.

---

## 3. The method

```
citation neighbourhood → GraphSAGE encoder → prefix adapter → soft KV tokens
                                                            → FROZEN language model → abstract
```

**A graph neural network (GraphSAGE) reads the neighbourhood.** Each node starts as its 1,024-d
abstract embedding. Each layer lets a node average information from its neighbours, so two layers
summarise a paper's references *and* those references' own references. Output: one 1,024-d vector
per target paper.

Three design points, each of which is a decision that could have gone otherwise:

- **Edges point from cited to citing.** Messages flow *from* a paper's references *into* the paper,
  so a node only ever learns from work it cites. Aggregating incoming citations would feed the model
  papers published *after* the target — a temporal leak. Undirected edges were implemented first and
  reverted for exactly this reason (and because hub papers with in-degree up to 16,513 blew up
  neighbourhood sizes to 94,721 nodes in the worst case).
- **The target's own features are masked out** and replaced with a single learned `[MASK]` vector.
  The target's embedding *is* an encoding of the abstract being generated, so leaving it visible
  would hand the model the answer. This is enforced by a test that randomises the target's input row
  and asserts the output is bit-identical.
- **Mean aggregation**, because unit-norm semantic embeddings averaged over ~9 references give a
  genuine centroid, and the result does not depend on how many references there are.

**The language model is frozen.** Its weights never move. If it were fine-tuned, an improvement
could not be attributed to the graph rather than to the model simply getting better at writing
abstracts.

**A soft prefix is the only channel between them.** The adapter converts the graph vector into
learned key/value vectors injected into the attention cache of *every* layer, as if a few invisible
tokens sat in front of the prompt. They are not words — they are continuous vectors the graph
controls. Both arms read the same textual prompt; the prefix is the only difference.

Two details recur throughout the results:

**The size problem, and the factorisation.** The prefix is 2,621,440 numbers per paper (32 layers ×
2 roles × 32 heads × 10 tokens × 128). A naive map from the graph vector to that would cost
**2.68 billion** parameters — the cost is in the *output*, so shrinking the input saves nothing.
The design gives each (layer, role) pair a small 96-d "code" and expands all 64 of them through
**one shared decoder**: 12.1M parameters, **221× smaller**. With the encoder, about **16.3M
trainable parameters** against 7 billion frozen ones.

**The gate.** 64 scalars — one per layer and role — scale the graph's contribution:

```
prefix[l,r] = base[l,r] + gate[l,r] * delta[l,r]
              ^ same for every paper   ^ depends on this paper's neighbourhood
```

`gate` starts at **exactly zero**, so training begins as ordinary prefix tuning with the graph
disconnected. This buys a *free, exact ablation* — set the gate to zero on the trained weights and
you have the same model with the graph removed, no second training run — and it is logged as
`valid/gate0` on every validation pass.

It also creates the project's most interesting risk. Because `d(prefix)/d(delta) = gate`, at
`gate = 0` the entire graph pathway — 13.7M of the 16.3M trainable parameters — receives *exactly
zero* gradient. It is frozen by arithmetic. The gate has to learn its way open from a randomly
initialised delta, and only will if a random projection of the neighbourhood is slightly better than
nothing. It did, visibly: the telemetry shows the gate flat for ~50 steps, then an S-curve, with a
gradient-norm spike at step ~85 as the pathway unfreezes.

---

## 4. How results are measured

The evaluation set is **frozen and versioned in the repository** (`data/frozen/`), because a result
cannot be reproduced without knowing exactly which papers produced it. Currently 945 test-split
papers, stratified by reference count.

Both arms are generated **in the same job, on the same GPU**, with greedy (deterministic) decoding
and identical numerical precision, so nothing but the prefix can differ between the two output
files.

Scoring uses three metric families:

- **ROUGE** — n-gram overlap with the real abstract.
- **BERTScore** — embedding similarity.
- **LLM-as-judge** — `gpt-5.4-mini` reads the generated abstract beside the real one and scores
  *accuracy*, *coherence* and *language* from 1 to 10, with a written reason for each.

`llm_accuracy` is the headline number, because it is the only one that asks *is this about the right
paper?* — and getting the right paper is the whole claim.

---

## 5. What was built, week by week

> The repository's own weekly numbering starts at "Week 4" — the first three weeks predate this
> codebase. Dates below are the dates in the stage documents and the git history.

### Jun 30 – Jul 1 — reference filtering: which citations actually matter?

The starting hypothesis was that not every reference is informative, so the pipeline began by trying
to *filter* them. Four scripts on a 1,936-node / 2,100-edge subset:

- **An LLM relevance label per edge.** `gpt-5.4-mini` was asked, for each (citing, cited) pair,
  whether the reference contributes meaningfully to the citing paper. Result: **845 relevant vs
  1,255 not** — roughly 60% of references are arguably noise. This single number motivates a lot of
  later design, and is the standing argument for trying attention-weighted aggregation instead of a
  plain mean.
- **Voyage embeddings** of every abstract, replacing OGB's word2vec features.
- **Direct cosine similarity** between the two endpoints of each edge (min 0.23, mean 0.62,
  max 0.97).
- **Centroid centrality** — for each paper, build the similarity matrix between all its references
  and score each reference by how similar it is to the rest. A high score means the reference is
  thematically central to the bibliography rather than tangential.

That last score is the one that survives. It is not used as a filter in the end; it becomes, in
August, the rule that decides **which references fit in the prompt**.

### Jul 14 – Jul 26 — repository and datasets

Repository initialised. ogbn-arxiv downloaded. The Semantic Scholar bulk download and the joint
dataset build, which is the larger job: ~500 GB of raw shards, resumable downloads, DuckDB joins
keyed on `mag_id`.

### Jul 27 – Jul 28 — the foundations: node sets, graph tensors, encoder

Three stages in two days, and the order matters — each one had to be pinned down before the next
could mean anything.

**Stage 1 — which papers are we even talking about.** Eligibility was fixed as `core` papers with
out-degree 5–15, giving **65,609 eligible nodes** (36,093 train / 12,020 valid / 17,496 test), and
an evaluation set of **495 papers, 45 at each out-degree from 5 to 15**. Equal bins so quality can be
plotted against reference count without bin size confounding the trend; 495 rather than 500 because
500 does not divide by 11.

This is also where the S2 coverage confound was discovered, and it killed the original plan: the
intended 200-node evaluation set (kept for comparability with earlier baselines) turned out to span
out-degree 1–65 in the joint graph, with about half its prompts overflowing the context. **Backward
comparability was given up deliberately** and everything rerun.

The 5–15 band was never a modelling choice — it is what fits in a 4,096-token context. That is
recorded here as a known problem and fixed in August.

**Stage 2 — the graph tensors.** Edge direction settled (references only, reversed for PyG), 20
self-loops dropped, edges coalesced. Measured neighbourhood sizes: **8.6 references at 1 hop, 50.6
nodes at 2 hops** on average. Hop 2 is genuinely populated — only 0.3% of papers have an empty
2-hop neighbourhood — which matters, because the prefix is supposed to carry 2-hop information the
1-hop prompt text does not have.

Four assertions run on every graph build, including one that checks a paper aggregates exactly the
works it cites *against the raw CSV*. Edge direction is the one thing that could be silently
backwards and still run.

**Stage 3a — the GraphSAGE encoder.** Two layers, 1024→1024→1024, mean aggregation, dropout 0.2,
**4,197,376 parameters**. Plus the `[MASK]` addendum the next day, which worked out what the mask
actually does (it is a *placeholder*, not a prior — it carries no node-specific information) and
measured a subtle side effect: when two papers in the same batch cite each other, the loader
deduplicates them into one row and one of them gets masked. Measured at 0.19% of appearances at
batch size 8, 1.79% at batch 64. Recorded and deliberately not fixed, because the cheap fix would
reopen a worse leak.

### Jul 30 – Aug 3 — the adapter, and the gate

**Stage 3b.** The factorised adapter described in §3 above — 12,108,096 parameters against a naive
design's 2.68 billion. Prefix tuning was chosen over input-only soft tokens so the graph signal
enters at all 32 layers instead of having to survive propagation up from the embedding layer.

**Stage 3b addendum (Aug 3).** A diagram of the whole architecture, and a careful account of the
gate — including the bootstrap problem, which had not been noticed when the design was written. The
scale analysis is the useful part: propagating the default initialisation through trunk → codes →
decoder gives a `delta` about **17× larger** than `base`, so ungated, step 0 would be almost pure
noise from an untrained decoder. This is *why* the gate must start at zero, rather than a nice
property it happens to have.

### Aug 1 – Aug 2 — proving the prefix is actually read

**Stage 4 step 1** is the most consequential piece of engineering in the project, and it produced
the finding the whole codebase is now organised around:

> **Prefix injection is silent when it is wrong.** A wrong cache offset produces fluent text and a
> falling training loss — not an error.

A hand-built key/value cache has to be treated by the library exactly as if real tokens had been
processed. The verification method is the clever part: split a token sequence into `prefix ++ body`,
inject the **real** key/values of `prefix`, generate from `body`, and compare against a plain run
over the whole sequence. Because both runs use the *same* model, the weights are irrelevant — so it
runs on a 4-layer random model on a laptop CPU in ten seconds.

What it found:

| | requirement | what happens otherwise |
|---|---|---|
| cache type | `DynamicCache`, built via `.update()` | a legacy tuple raises |
| attention mask | length `P + L` (covering the prefix) | raises |
| `cache_position` | passed explicitly | **max logit error 0.74, silently** |
| `generate()` | requires a monkeypatch | different tokens; passing the argument is ignored |

The negative control is the number worth quoting: without the patch, **19 of 20 generated tokens are
wrong**, with no error raised. It also corrected a mistaken prediction in Stage 3b — the expected
failure was a `position_ids` offset, but RoPE is relative so that turns out to be harmless; the real
hazard was elsewhere.

**Stage 4 — training.** The training loop, with each element of the recipe traceable to a specific
silent failure found above. Loss is cross-entropy on the target abstract's tokens only. AdamW,
lr 1e-4, warmup 100 steps, no weight decay (decaying `base` and `gate` would pull the prefix toward
the ablation). The gate is left to learn freely — no forced opening — because *if it stays at zero
that is a real result about the graph*, not an artefact of the schedule. Validation reports the loss
twice, with the gate learned and with it forced to zero.

**Stage 4b** added Weights & Biases telemetry, and the choice of series is deliberate: `valid/delta`
(the graph-vs-no-graph gap) is the thesis's central claim, watched live during training rather than
discovered at the end; `gate/per_layer` shows *where in the stack* the graph gets used.

**Stage 4c** measured the frozen subsets and found the second structural problem: **context overflow
is not random.** At out-degree 15, more than half of all candidate papers do not fit 4,096 tokens.
So the surviving papers are those with terser references — precisely at the high-degree end where
the graph should have the most to contribute.

**Stages 4d–4e** built the Singularity container and the SLURM training job for the Komondor
supercomputer, including two things that recur: the pre-flight verification scripts run on CPU
*before* the GPU hours, and the job auto-resumes from the newest checkpoint.

### Aug 4 – Aug 5 — the first real result, then a reorganisation

**Stage 5** produced the first two comparable CSVs: the trained adapter and the frozen baseline over
the same 495 papers, same prompts, same greedy decoding, same precision, back to back in one job.
63 papers overflowed the context in *both* arms by construction, leaving **432 scorable**:

| metric | adapter | baseline | delta |
|---|---|---|---|
| ROUGE-2 F | 0.0606 | 0.0540 | **+0.0065** |
| ROUGE-L F | 0.1747 | 0.1614 | **+0.0133** |
| BERTScore F1 | 0.8386 | 0.8405 | −0.0019 |
| `llm_accuracy` | 2.82 | 2.46 | **+0.36** |
| `llm_coherence` | 7.03 | 6.32 | **+0.70** |

The prefix helps on longer n-gram overlap and on judged coherence, and is flat on unigram overlap
and embedding similarity. Both accuracy scores are low in absolute terms — 2.8 and 2.5 out of 10 —
which says the task is far from solved, not that the prefix failed.

**Aug 5** was housekeeping with a purpose: the repository was reorganised into `src/` (code only),
`data/`, `experiments/` (one directory per run, always the same five files), `docs/` and `hpc/`, and
the same layout migrated onto Komondor in place, keeping the 13 GB model cache and the trained
checkpoint.

### Aug 6 – Aug 12 — the capped-prompt redesign

This is the week that changed what the experiment is capable of showing, and it is the most
important design change in the project.

**The problem.** The 5–15 reference band existed only because the whole reference list went into the
prompt, and 15 abstracts is what a 4,096-token context holds. So the model had *only ever been
trained where the graph has least to say.*

**The fix** separates two things the context limit was conflating:

| | before | after |
|---|---|---|
| what the GNN may read | out-degree 5–15 | out-degree 5–100, the **whole** neighbourhood |
| what the prompt shows | all of them | the ~15 most central, as many as fit |
| papers lost to overflow | ~13% of eval | none |

Which references get shown is decided by the **centroid centrality score from Week 4** — keep the
15 closest to the bibliography's centroid, then drop the farthest one at a time until it fits.
Alternatives were considered and rejected in writing: keeping the *farthest* references (keeps the
outliers, throws away the core), stratifying across the range (no precedent in this project's own
filtering results), and dropping any overflowing paper (which was the first version, and cost 45% of
every out-degree above 15 — biased, because the survivors are those whose central references happen
to have short abstracts).

**The measurement that makes the experiment work:**

| out-degree | references shown | prompt tokens |
|---|---|---|
| 5–9 | 7.0 | 2,016 |
| 10–15 | 12.1 | 3,315 |
| 16–29 | 13.9 | 3,724 |
| 40–59 | 14.3 | 3,670 |
| 80–100 | 14.4 | 3,495 |

**Above out-degree ~16 the prompt stops growing.** A paper citing 16 works and one citing 100
produce visually identical prompts. So anything the model gains at high degree *has to come through
the GNN prefix* — the only channel still reading the other 85 references. That is precisely the
effect the adapter is meant to demonstrate.

The rest of the week was making this real rather than merely frozen:

- **Stage 1c** raised the training quota, growing the train set from 4,597 to **7,750** papers. The
  honest cost is recorded too: all 3,153 added papers sit at out-degree ≤ 39 (higher degrees were
  already exhausted), so train's mean degree *fell* from 29.1 to 25.5 while eval stayed at 51.8. The
  train/eval mismatch got wider, not narrower.
- **Stage 1d** made one shared function build every prompt — the freeze, training, and generation.
  Copying the logic into three places would have let them drift, and *a mismatched prompt budget is
  silent*: it produces a fluent abstract written from a different set of sources than the paper was
  selected under.
- **Stage 1e** wired training and generation to the capped sets behind a single `--sets` flag, with
  three guards: every checkpoint records which node sets it was trained on and refuses to load
  against the other; asking for more generation tokens than the freeze reserved is refused at
  startup; and an overflowing prompt now *aborts the run* instead of writing another
  plausible-looking error row.
- **Stage 1f** added a CPU pre-flight check (20 seconds, gating a multi-hour GPU job), early
  stopping with patience that **survives a job resubmit** (otherwise a 10-epoch run split across
  jobs could never stop early), and `best.pt` — because when early stopping fires the *last*
  checkpoint is by definition worse than the best one.

### Aug 15 — the reference run

The first full training run on the capped sets: 7,750 training papers, 945 evaluation papers.

| | mean `llm_accuracy` |
|---|---|
| frozen Vicuna-7B, no prefix | **2.815** |
| GraphSAGE prefix adapter | **3.288** |
| **gap** | **+0.473 ± 0.098** (95% CI) |

This is still the project's headline result, and every later experiment is defined relative to it.

### Aug 18 – Aug 21 — depth, and three prompt rewrites

**Stage 6a** made GNN depth a per-run argument rather than an edited constant, for a non-obvious
reason worth repeating: `sbatch` snapshots the *job script* at submit time, but the Python is read
when the job actually *starts*. Editing the source between two submissions means the first job can
run the architecture intended for the second. Depth now travels as an argument, is recorded in every
checkpoint, and generation reads it *out of* the checkpoint — so a 3-layer checkpoint cannot be
paired with a 4-layer encoder. Measured cost of deeper hops: 2 hops touch 199 nodes on average,
3 hops 876, 4 hops 2,645 — none of which is a memory problem, because the run is bound by the
language model, not the graph.

**Stages 6b–6c** rewrote the generation instruction three times, each rewrite driven by a diagnosis
rather than intuition. The original prompt turned out to specify the *wrong task*: it asked for "a
new application paper" that "builds on and combines these works", when the target is the existing
paper's own abstract. Three measured consequences — surveys failed as a class, accuracy rose
monotonically with how close a paper sits to its own bibliography's centroid, and generated
abstracts were far shorter than real ones.

The rewrites did not work:

| era | mean accuracy |
|---|---|
| 1 (Aug 15) | 3.288 |
| 2 (Aug 20) | 3.259 |
| 3 (Aug 21) | 3.214 |
| 4 (Aug 23) | 3.143 |

**Each rewrite changed 70–90% of the generated text and moved the mean by less than judge noise —
slightly downward.** Two measurements explain why: bigram overlap with the *real* abstract is 0.027,
while overlap with the nearest *cited* reference is 0.146 — output is ~5× closer to a paper the
target cites than to the target. And **10.4% of outputs are near-verbatim copies of a single cited
abstract**, making up 19–24% of everything scoring 6 or above. A fifth of the top tier was
regurgitation.

The conclusion drawn: **prompting controls *how* the abstracts are written, not *which paper* they
are about.**

One subtler result from this week deserves its own line. The measured gap *shrank* across the eras —
+0.473, +0.431, **+0.343** — and this was progress, not regression. Era 2's prompt had made the
*no-graph baseline* start describing the bibliography instead of the paper (third-person "the
authors" went 15.4% → 58.1%), inflating the gap. Repairing the control lowered the headline number.
**A change that lowers your headline result can be the more honest measurement.**

### Aug 24 – Aug 29 — diagnosis instead of tuning

Having established that prompting was not the lever, the work turned to figuring out *what* the
system does and *how* it is being measured.

**Blind-judge comparisons** (three rounds). An LLM judge compares two generations of the same paper
against the real abstract, blind to the scores and to how the papers were sampled. Round 1 compared
baseline against adapter and produced the clearest sentence in the project — quoted in §6 below.
Round 2 explained why a rewrite that barely moved the mean was actually a large change: the mean was
the sum of two opposite effects, **+0.81 on the floor tier and −2.00 on the top tier**. The blind
judge, having never seen the scores, picked the old version on 15 of the 15 collapsed-top papers.
Round 3 found the blind judge and the scoring judge disagreeing — and both being right: the rewrite
made abstracts better *written* and more hedged, which a preference judge rewards and an accuracy
rubric does not. **Implication: do not tune on a preference judge.**

**Stage 6d — LoRA.** With prompting exhausted, the remaining failures were register and style —
stock openers, LaTeX markup, length calibration — none of which a prompt can fix in a 7B model, and
all of which are learnable from 7,750 real abstracts. LoRA adds ~8.4M trainable parameters to the
LLM's query and value projections (chosen because the prefix arrives as extra keys and values, so
those two projections decide how much of it gets read). Two decisions worth noting: it is *injected*
in place rather than wrapped, because the wrapper would override the very method the prefix patch
monkeypatches; and its parameters are held in fp32 because AdamW on bf16 parameters silently loses
small updates. The standing risk is stated plainly — 8.4M LoRA parameters against a 64-element gate,
and if LoRA learns the task faster than the gate learns the graph, the measured graph effect
collapses. The telemetry to catch it is already logged.

**Prompt sensitivity under a stronger model (Aug 29).** The null result on prompts had a confound:
maybe Vicuna-7B simply cannot follow instructions of that length, in which case the four eras
measured instruction-following, not prompts. So all four were rerun through `gpt-5.6-luna` on the
same 48 papers, with the reference lists held byte-identical.

| | era 1 | era 2 | era 3 | era 4 |
|---|---|---|---|---|
| gpt-5.6-luna | 4.583 | 4.938 | 5.417 | **5.812** |
| Vicuna-7B, same papers | 4.292 | 3.771 | 3.667 | — |

**The prompts separate under the stronger model, and in the opposite direction** (+1.23, p=0.0002).
The era-4 instruction — *worst* of the four under Vicuna — is the best here. Prompting was exhausted
for the 7B, not for the task. This is the direct motivation for changing the language model.

That run also had to rule out the obvious objection — that a strong model might simply *recall*
these arXiv papers. Three lexical checks say no: zero papers leak a name that appears in the target
but in none of the shown references; overlap with the real abstract stays at 0.027 (Vicuna's own
figure) while accuracy rises 2.15 points; and the near-verbatim copy rate goes from 10.4% to **0%**.
The model scores far higher while sitting *further* from the target's text. That is better
inference, not retrieval.

### Sep 3 — how much of the result is the ruler?

Every accuracy number in the project comes from one judge prompt, and nobody had asked what that
choice was worth. The **rubric sensitivity study** took the same two sets of generated abstracts —
unchanged — and scored them under four different judge rubrics. Same generations, same judge model,
only the rubric moves.

| rubric | scores | graph gap | control gap |
|---|---|---|---|
| R1 — the current rubric (objective, domain **and methodology**) | 2.83 / 3.21 | +0.387 | not significant |
| **R2 — problem and task only** | 3.17 / 4.01 | **+0.833** | **significant** |
| R3 — R2 decomposed into domain/problem/task/scope | 2.98 / 3.45 | +0.473 | borderline |
| R4 — R2 plus a penalty for describing a cited work | 2.50 / 3.24 | +0.740 | not significant |

**The rubric decides roughly half the measured effect.** That is the main result, and it applies to
every accuracy figure in the project.

Two design choices made it trustworthy. A **negative control** — two runs that differ only in a
decoding setting and should show no graph effect — caught R2 spreading apart things that should be
identical. And reporting all four rubrics, including the flattering ones, caught R4: it has the
biggest effect size, but its entire gain comes from pushing the *baseline* down, and the penalty
does **not** land on the papers where the baseline actually copied a cited work. R4 was written from
the baseline's observed failure, and it turned out to be simply harsher rather than more perceptive
— exactly the thumb-on-the-scale outcome the design was built to catch. It is reported as **checked
and rejected**, which is a stronger methodological story than one that was adopted.

R3's decomposition is the explanatory figure: scores fall steadily as the question gets more
specific (domain 7.8, problem 4.7, task 3.9, scope 2.6) and **the adapter's advantage grows the
harder the question gets** — +4.7% on domain rising to +19.7% on scope.

### Sep 3 – Sep 6 — swapping the language model for Qwen3-14B

The GPT result said the ceiling was the 7B, not the task. So: replace the frozen language model with
**Qwen3-14B** — twice the size, far more modern — and change nothing else, so the gap is directly
comparable to +0.473. A written plan came first (Sep 3), then seven implementation steps.

**Step 1 — container and model.** The expected dependency cascade did not happen: reading the
package metadata showed `transformers==4.51.3` declares no PyTorch pin at all, so the base image,
CUDA and both compiled graph extensions could stay byte-identical. Checking that is what made the
step small. Two dead ends are recorded so they are not retried. The container's self-test was
extended to build a live Qwen3 and assert its cache layout — **the first time grouped-query
attention is exercised anywhere in this project.**

**Step 2 — freezing the reference lists.** The most important step, and one the plan called *"what
makes 'the data is the same' true rather than assumed."* Which references a prompt showed was decided
at call time by counting tokens — so it was a function of the *tokenizer*. Qwen's BPE is more
efficient on English than Vicuna's SentencePiece, so more references would have survived the same
budget and **the two arms would silently have read different evidence.** The lists are now computed
once under the Vicuna tokenizer and written to disk: 9,501 papers, mean 12.87 references. The check
that matters is a hard gate against 48 prompts stored back in August — **48/48 match string for
string**, which also confirmed that one configuration constant had been correctly chosen.

**Step 3 — prefix injection, re-derived.** The plan's instruction was *re-derive the patch, do not
port it*, and that was right: the bug still exists but it is **a different bug**. The old library
under-counted the prefix and shifted every attention mask row. The new one assumes the cache holds a
prefix of the input, and silently **drops the first 32 body tokens** — the model then writes a fluent
continuation of a truncated prompt. Porting the old fix would have been wrong in both halves. The
negative control is what tells you which world you are in.

**Step 4 — geometry.** Qwen3-14B has 40 layers and, crucially, **grouped-query attention**: 40 query
heads served by only 8 key/value heads. Since the prefix *is* a key/value tensor, holding the token
count at 10 would have narrowed the graph's channel into the model by 3.2× as a side effect — and a
fallen gap could not then be attributed to the model rather than the narrower prefix. So the prefix
grows from 10 tokens to **32**, restoring the volume exactly:

```
Vicuna, T=10:  32 × 2 × 32 × 10 × 128 = 2,621,440
Qwen3,  T=32:  40 × 2 ×  8 × 32 × 128 = 2,621,440   (identical)
```

What it *cannot* restore is addressability — 8 heads serving 40 means the prefix can no longer say
different things to different heads within a group. There is no workaround for a GQA model, so it is
recorded as a limitation rather than fixed.

**Step 5 — the chat wrapper.** The prompt was doing two jobs at once: the experimental content, and
the model-specific chat formatting. Only the second may move, so they were split. The reason this
matters is asymmetric: the adapter arm is *trained* and would learn any format, but the baseline arm
is frozen — a frozen Qwen3 given non-ChatML input is off-distribution precisely on instruction
following, and the degradation would land entirely on the control and **inflate the measured gap**.
That is the same artifact class already caught back in Stage 6c. The wrapper is frozen in a test so
a library update cannot move every prompt in both arms without anyone noticing.

**Step 6 — memory.** The first step motivated by hardware rather than correctness. Qwen3-14B is
29.6 GB in bf16 on a 40 GB card, and what does not fit alongside it is the *logits*, because Qwen's
vocabulary is 151,936 tokens — 4.75× Vicuna's. Measured over 300 papers: sequences average 3,075
tokens but only **219 of them carry a training label** — the rest is prompt. So **92.9% of the logits
were computed, upcast to fp32, and discarded.** Computing the loss only on the supervised positions
saves ~2.4 GB. Three details make it the *same* number rather than a similar one (the off-by-one
shift, the fp32 upcast, the denominator), and each is checked to ~5e-07 against the original path.

**Step 7 — making the tests test what they claim.** Two of the three pre-flight scripts were passing
for the wrong reason. One built its test model with equal query and key/value head counts, so the
cache layout could have been wrong in exactly the dimension Qwen3 introduces and every assertion
would still have held. The other compared two things that, after Step 2, could no longer disagree —
a test that had silently stopped testing. Both were rewritten. This is the project's recurring theme
applied to its own tooling.

**Steps 8–9 — the training run.** Set up Sep 6, not yet run. The planned separate pilot was folded
into the real run, because every number a pilot would produce lands in the first hour of a
checkpointing run anyway, and a second GPU allocation on a shared queue is the scarce resource. What
to watch in that first hour is written down: `gate/mean_abs` rising. A falling training loss against
a flat gate means the model is learning the task **without** the graph — the one failure worth
killing the job over.

---

## 6. What has been found

**The graph helps, and the gap is real but modest.** +0.473 accuracy points on a 1–10 scale
(2.815 → 3.288, 945 papers, 95% CI ±0.098).

**What it helps *with* is topic anchoring.** From the blind-judge reading of 57 pairs:

> Without the graph, the model typically writes a fluent abstract of *one of the cited papers* — a
> real, recognisable, wrong paper. With the graph, it writes about the right research problem, the
> right method family, the right vocabulary. But it still cannot recover the paper's actual
> contribution — its named method, its numbers, its conclusions — and fills that gap with plausible
> invention.
>
> **Graph conditioning converts "a fluent abstract of the wrong paper" into "a fluent abstract of a
> plausible-but-nonexistent paper on the right topic."**

Both systems drift; the graph shrinks the drift radius rather than eliminating it. And it costs
something exactly when a paper is an outlier from its own citation neighbourhood: surveys fail as a
class (a survey's neighbours are method papers, so the model writes a method paper), and so do
papers whose contribution sits at a different level of abstraction than their references — a loss
function among segmentation applications, a hardware accelerator among attention papers.

**There is a hard ceiling, and it was measured.** Comparing gpt-5.4-mini against Vicuna-7B on the
same papers: a paper's specific *methodology* is wrong 70% of the time for the small model and
**95%** for the strong one, while *domain* goes 58% → 91% and *task* 22% → 53%. Methodology is not
recoverable from a bibliography by any model — a paper's contribution is by definition its departure
from its references, so it is the one thing the conditioning signal cannot contain. R3's ordering
(domain 7.8 → scope 2.6) says the same thing from the other direction.

**Prompt wording is not the lever — for a 7B.** Four rewrites moved the mean by 0.145 in total, the
wrong way. The same four rewrites through a stronger model moved it +1.23 the right way, reversing
the ordering entirely.

**The measuring instrument decides half the effect.** Same generations, four rubrics, gaps from
+0.387 to +0.833. Any single LLM-judge number is a property of the ruler as much as of the model.

**A fifth of the good scores were regurgitation.** 10.4% of outputs are near-verbatim copies of a
single cited abstract, and those make up 19–24% of everything scoring ≥6 — a caveat on every ceiling
claim made before it was measured.

---

## 7. Known limitations

- **Coverage confound.** 44% of papers have incomplete bibliographies, so reference count is not
  comparable across papers. Measured, not fixed.
- **Train and eval have different shapes.** Training papers average 25.5 references, evaluation
  papers 51.8, because the graph simply does not contain enough high-degree papers to fill a quota.
  A quota is the wrong instrument above out-degree ~34.
- **The gate may never open.** The graph pathway starts with exactly zero gradient and has to
  bootstrap. It worked under this seed; it is not guaranteed to under another, or with LoRA changing
  the loss landscape. A seed sweep is owed before the design is defended.
- **The ablation is "graph off", not "prefix absent".** At `gate = 0` the prefix positions still
  absorb attention. Honest, but worth stating precisely.
- **Silent failure is the norm in this layer.** A wrong cache offset produces fluent text and a
  falling loss. This is why every job runs CPU verification before spending GPU hours, and why the
  documentation is as long as it is.
- **One judge, one vendor.** Generator and judge come from the same model family, so part of any gap
  may be register preference rather than accuracy. Unmeasured.
- **Qwen3-14B results do not exist yet.** That arm is currently plumbing, arithmetic and
  verification. No training run has completed, and peak GPU memory remains an estimate.

---

## 8. What comes next

1. **Run the Qwen3-14B training** and re-measure the +0.473 gap under a model that can actually
   follow the instruction. If the gap holds or grows, the finding is about the method; if it
   collapses, it was about the 7B.
2. **Re-score under rubric R2** — research problem and task, methodology excluded — as the
   sensitivity study recommends, reported alongside the old numbers rather than mixed with them
   (R1 and R2 cannot share a table).
3. **Isolate the graph from the text.** The strongest remaining criticism is that nothing yet
   separates *graph structure* from *access to more neighbour text*. A baseline given the same
   references, well summarised, might show the same topical anchoring. Until that arm exists, the
   headline gap is evidence for conditioning, not specifically for the graph.
