# Development history — how the final design was reached

Sources: `docs/project_overview.md`, `docs/adapter.md`, `docs/adapter_evaluation.md`,
`docs/qwen_migration.md`, `docs/soft_prompt_only.md`, `docs/step6_failure_analysis.md`,
`docs/step8_representation_plan.md`, `docs/run9_embedding_decoder.md`, `docs/run11_model.md`,
`notes.md` files in `experiments/`, and the per-run means recomputed from every
`experiments/*/evaluated_result.csv` on 2026-10-13. Dates are 2026.

All judge numbers in this file are the **old rubric** (`gpt-5.4-mini`, accuracy 1–10, see
`05_evaluation_methods.md`). TF-IDF top-1 is identification of the generation's own gold among
the 945 eval golds (chance 0.11%). "Distinct" = number of different generated strings out of
945.

---

## 0. The arc in five sentences

1. The project started as **text + graph**: a frozen LLM reads the reference abstracts as text,
   and a GNN prefix adds structure; with Vicuna-7B the prefix helped (+0.47).
2. With the stronger **Qwen3-14B** the prefix stopped helping (−0.07), so the question became
   **can the graph replace the text** — the soft prompt as the *only* channel.
3. Seven end-to-end soft-prompt runs (steps 1–8) **failed in instructive ways**: a frozen LLM
   could not read the prefix; with LoRA the model learned to ignore an uninformative prefix
   (conditioning collapse); more capacity, per-reference tokens, ranking losses and whitening
   did not fix it.
4. The fix was to **split the problem**: train a *reader* (vector → abstract) on 90k papers
   from each paper's own embedding, and separately a *GNN* that estimates that embedding from
   the references; then fine-tune them together (runs 10a, 11).
5. Run 11 is the final model.

---

## 1. Before the codebase: reference filtering (30 Jun – 1 Jul)

On a 1,936-node / 2,100-edge ogbn-arxiv subset (200 test papers):
- `gpt-5.4-mini` labelled each (citing, cited) edge: **845 relevant vs 1,255 not**.
- Voyage embeddings replaced OGB's word2vec features; citing–cited cosine min 0.23, mean 0.62,
  max 0.97.
- "Centroid centrality" score per reference (similarity to the rest of the bibliography). It
  was not kept as a filter but became the rule for which references fit in the text prompt.

Archived results on that 200-paper set (Vicuna-7B unless noted; `experiments/_archive/`; *not*
comparable with later sets because those papers span out-degree 1–65 in the joint graph):

| run | judge accuracy | note |
|---|---|---|
| `gpt-5.4-mini` with all references | 4.145 | strong API model |
| `vicuna-7b-8bit`, all references | 2.503 | 8-bit, sampled at T = 0.35; 53 context-overflow rows |
| `cosine-centroid-30/60/90` (keep 30/60/90% of refs) | 2.611 / 2.585 / 2.550 | filtering did not help |
| `cosine-mean-30/60/90` | 2.383 / 2.598 / 2.479 | |
| **`random-sample`**: each test paper's references **replaced by random papers** | **1.028** | 138 of 142 judged rows score 1; BERTScore F1 still 0.820 (see `05`) |

## 2. Text + graph with a frozen Vicuna-7B (27 Jul – 29 Aug)

### 2.1 Building the apparatus (stages 1–5)

- **Stage 1–2:** joint dataset, eligibility, directed reference-only graph (`02_data.md`).
- **Stage 3a:** 2-layer GraphSAGE encoder 1024→1024→1024, mean aggregation, dropout 0.2,
  4,197,376 parameters, learned `[MASK]` on the target row.
- **Stage 3b:** factorised prefix adapter for Vicuna (32 layers × 2 × 32 heads × 10 tokens ×
  128 = 2,621,440 numbers per paper); trunk 768, code 96, **one** shared decoder; 12,108,096
  parameters (a naive map: 2.68B). Gate initialised to 0 (64 scalars).
- **Stage 4, step 1 — prefix injection is silent when wrong.** Verified by splitting a real
  token sequence into prefix + body, injecting the real KV of the prefix, and comparing with a
  plain run over the whole sequence on a 4-layer random model (weights irrelevant, CPU, ~10 s).
  Findings (transformers 4.39.3): a `DynamicCache` built via `.update()` is required; the
  attention mask must cover prefix + body; **`cache_position` must be passed explicitly when
  `use_cache=False`** (otherwise max logit error 0.74, silently — and gradient checkpointing
  forces `use_cache=False`); `generate()` overwrites `cache_position` and needed a patched
  `prepare_inputs_for_generation`. Negative control: without the patch **19 of 20** generated
  tokens are wrong, with no error. With it, 20/20 match, logit error 1.9e-07. The expected
  hazard (`position_ids` offset) turned out harmless because RoPE is relative (3.6e-07).
- **Stage 4:** training recipe: read-only `PrefixCache` (a mutating cache grows 10 → 26 → 42
  under checkpoint recomputation), explicit `cache_position`, non-reentrant checkpointing,
  AdamW lr 1e-4, warm-up 100, weight decay 0, clip 1.0, effective batch 16. The gate learns
  freely from 0. **Bootstrap observed:** gate flat ~50 steps, then an S-curve, gradient-norm
  spike at step ~85.
- **Stage 5:** first comparable pair on 495 papers (63 overflowed in both arms → 432 scored):
  adapter 2.82 vs baseline 2.46 accuracy; coherence 7.03 vs 6.32; ROUGE-L +0.013; BERTScore
  −0.002.

### 2.2 The capped-prompt redesign (6–12 Aug)

The 5–15 out-degree band existed only because all references had to fit 4,096 tokens. The
redesign let the GNN read **all** references (out-degree 5–100) and the prompt show the ~15
most central that fit. Above out-degree ~16 the prompt stops growing, so any high-degree gain
must come through the prefix. Train 7,750 / valid 806 / eval 945 (`02_data.md` §5).

### 2.3 The reference result (15 Aug, job 14207605)

| arm | judge accuracy | ≥ 3 | coherence |
|---|---|---|---|
| frozen Vicuna-7B, references as text | 2.815 | 53.3% | 7.17 |
| + GraphSAGE prefix | **3.288** | 65.1% | 7.14 |
| **paired gap** | **+0.473 (SE 0.050)** | | |

### 2.4 Prompt rewrites and other variants (18–29 Aug)

| run | change | adapter accuracy |
|---|---|---|
| 08-15 (era 1) | original prompt | 3.288 |
| 08-20 (era 2) | rewritten instruction (task specified as "this paper's abstract") | 3.259 (baseline 2.829) |
| 08-21 (era 3) | two clauses removed | 3.214 (baseline 2.871) |
| 08-23 `14233259` (era 4) | era-4 prompt, repetition penalty 1.15 | 3.143 |
| 08-23 `14233391` (era 4) | era-4 prompt, repetition penalty 1.0 | 3.204 |
| 08-24 `14233674` | condition not recorded in `notes.md` | 3.270 |
| 08-24 `14235097` | first LoRA run (Vicuna, ~8.4M LoRA params) | 3.306 |
| 08-25 `14242534` | 3-layer GNN | 3.290 |

- Each rewrite changed 70–90% of the generated text and moved the mean by less than judge
  noise. The graph gap shrank +0.473 → +0.431 → +0.343, partly because era 2 had made the
  baseline describe the bibliography ("the authors" 15.4% → 58.1%), inflating the gap.
- Bigram overlap with the real abstract 0.027 vs 0.146 with the nearest *cited* reference;
  **10.4% of outputs were near-verbatim copies of one cited abstract**, 19–24% of all outputs
  scoring ≥ 6.
- **Prompt sensitivity under a stronger model** (`gpt-5.6-luna`, same 48 papers, same
  references): era 1 → 4 = 4.583 → 5.812 (+1.23, p = 0.0002), the opposite order to Vicuna.
  Prompting was exhausted for the 7B, not for the task. Lexical checks found no sign the
  strong model recalled the papers (near-verbatim copy rate 0%).
- **Rubric sensitivity study** and **blind-judge comparisons** were run in this period
  (`05_evaluation_methods.md`).

## 3. Swap to Qwen3-14B (3–10 Sep)

Planned to change nothing else so the gap stays comparable (`model_swap_qwen.md`,
`qwen_migration.md`):
- container: `transformers 4.51.3` (first Qwen3 support), torch/CUDA/PyG unchanged;
- **reference lists frozen** under the Vicuna tokenizer (`prompt_refs_capped.json`; Qwen's
  tokenizer would otherwise fit more references and the arms would read different evidence);
- **prefix injection re-derived**: under 4.51.3 the bug is different — the library assumes the
  cache holds a prefix *of the input* and silently drops the first 32 body tokens;
- **prefix length 10 → 32 tokens**: GQA (8 KV heads instead of 32) would otherwise narrow the
  channel 3.2×; 40 × 2 × 8 × 32 × 128 = 2,621,440, identical volume to Vicuna's. Lost:
  per-head addressability within a KV group (a GQA limitation);
- ChatML wrapper split from the experimental prompt and frozen in a test;
- loss computed only at supervised positions (−2.4 GB);
- verification scripts fixed to actually exercise GQA.

**Result (10 Sep, job 14364743):**

| arm | judge accuracy | ≥ 3 | coherence | TF-IDF top-1 |
|---|---|---|---|---|
| frozen Qwen3-14B, references as text (**the text baseline**) | **3.592** | 79.6% | 7.90 | 42.9% |
| + GraphSAGE prefix | 3.520 | 71.4% | 7.87 | 36.5%* |
| paired gap | **−0.072 (SE 0.051)** | | | |

(*from `soft_prompt_only.md` step 2 table.) The gate opened, so the prefix was used; it just did
not help once a stronger model could read the abstracts itself. **The question became: can the
graph replace the text?**

## 4. The graph as the only channel: end-to-end soft-prompt runs (16 Sep – 5 Oct)

All runs: the prompt carries **no references**, only a fixed instruction (era-4 derived); the
prefix is the only paper-specific input; GNN + adapter (+ LoRA) trained end-to-end on the LM
loss over 7,750 papers.

| run | what changed | LLM | judge acc | TF-IDF top-1 | distinct |
|---|---|---|---|---|---|
| text baseline (ref.) | references as text | frozen | 3.592 | 42.9% | 945 |
| **step 1** (14393649) | references removed from prompt; 2-layer SAGE, adapter trunk 768 / code 96 | frozen | **1.954** | 4.2% | 879 |
| step 2 (14395431) | bigger adapter: two decoders (K/V), code 192, trunk 1024 (12.5M → 32.1M) | frozen | 1.737 | 2.4% | 809 |
| step 3 (14411691) | **oracle**: own embedding in, 2 epochs | frozen | 1.249 | 0.3% | 198 |
| step 4 (14422070) | **oracle + LoRA r16**, 10 epochs | LoRA | 2.712 | 12.0–16.5%† | 944 |
| step 5 (14437531) | graph + LoRA r16 (one change vs step 2) | LoRA | 1.490 | 1.0% | 453 |
| step 6 (14450409) | one input-embedding token per reference + GATv2 + LoRA r64 (+k_proj) | LoRA | 1.314 | 0.3–0.5% | 342 |
| step 7 (14462189 / 14463510) | ranking loss, pretrained encoder (InfoNCE), gates opened at init, LoRA r16 | LoRA | 1.421 / 1.377 | 0.1–0.3% | 34 / 195 |
| run 8 (14482476) | ZCA-whitened features, VICReg, cleaned targets, canary | LoRA | 1.519 | 0.0–0.2%‡ | 59 |
| gate = 0 controls | graph disconnected | LoRA | 1.29–1.35 | chance | 1 |

† 12.0% in the step-4 write-up; 16.5% when rescored with the run-9 instrument. Same run.
‡ The step-8 scorecard reports 0.00%, the run-9 rescoring 0.2%. Ranges in this table reflect
the same runs scored by slightly different versions of the TF-IDF probe; they never change
which run is better.

### What each run established

- **Step 1:** the prefix alone does not replace ~13 reference abstracts. Outputs are fluent
  (coherence 7.5) and distinct (879) but rarely the right paper.
- **Step 2 — more capacity made it worse** (−0.218, SE 0.065). Outputs became more generic:
  distinct word types 4,817 → 4,194; rows sharing the most common 8-word opening 60 → 232. The
  identifying content of the graph vector had **no measurable effect**: accuracy 1.743 where the
  reference mean retrieves the right paper vs 1.724 where it does not. Lower training loss and
  lower accuracy are consistent: cross-entropy over ~219 abstract tokens is dominated by generic
  scientific English.
- **Step 3 — the frozen oracle was at chance.** Given the target's own embedding, a frozen
  Qwen produced 198 distinct texts (one 600-token image-classification abstract 75 times).
  Concluded at the time: the prefix → LLM channel is the bottleneck.
- **Step 4 — that conclusion was wrong.** The same channel, same 32 tokens, with LoRA and 10
  epochs: 944 distinct, judge 2.71, identification ~100× chance. **A frozen Qwen cannot read
  this prefix; a LoRA-adapted one can.**
- **Step 5 — conditioning collapse.** Graph + LoRA was worse than graph frozen on every metric
  (distinct 809 → 453; largest identical cluster 11 → 61 rows). A decoder able to adapt, given a
  prefix that carries little, minimises loss by ignoring it and writing a good generic abstract
  (the posterior-collapse pattern of Bowman et al. 2016). Within the arm the graph was still
  worth ~+0.5 over gate = 0.
- **Step 6 — per-reference tokens failed.** Lowest training loss of the project (1.83) with
  the worst conditioning. Collapse scaled with decoder capacity: frozen 2.86% → LoRA r16 1.16%
  → r64 0.32% top-1. Final neighbour gate 0.0083; disconnecting both channels cost ~0.001
  nats; domain agreement fell below the majority-class baseline (0.324 vs 0.414). **And the
  premise was false:** an untrained mean of the reference embeddings retrieves the target's
  own embedding at **64.34%** top-1 of 945 — the information was there; it was not being read.
- **Step 7 — ranking loss.** `L = L_LM + λ·L_rank`, the rank term asking whether the right
  abstract outscores M = 8 wrong ones under this paper's prefix, half the negatives being the
  paper's own references. Stage A pretrained the GNN with InfoNCE (no LLM): 67% top-1 vs
  64.34% for the untrained mean (+2.7 points). The rank term broke from chance once (step
  ~2,100: rank_hit 0.11 → 0.30–0.40) but a selection bug (validation nodes missing from the
  negative pool) silently fell back to LM loss and picked a pre-breakthrough checkpoint. Most
  collapsed outputs of the project (34 distinct).
- **Run 8 — whitening, VICReg, cleaned targets, a generation canary.** Collapsed to 59
  distinct strings, 199 distinct word types across all 945 outputs (golds: 17,344), loops
  (coherence 5.1). Graph contribution vs its gate-0 control +0.224 (t = 10.2): the graph was
  connected for the first time and carried almost nothing. **Root cause:** VICReg, InfoNCE and
  the output-similarity gate were all computed on `head(h)`, but the training script loaded
  only the encoder and fed the adapter `h`, which was ~0.85-collinear across papers (the
  adapter passes input collinearity through ~1:1). **Second cause:** with half the negatives
  being the paper's own references, the ranking task had a ceiling of 46.1% at M = 8 even in
  clean embedding space (97.2% with random negatives) — the rank loss sat at chance and carried
  no gradient. Selection then picked a checkpoint whose canary read 1 distinct of 16.

### The lessons carried into the final design

1. **The information is in the references; reading the vector is the problem.** A no-LLM
   retrieval floor — return the nearest *training* abstract to the reference mean — gives
   **27.5%** TF-IDF top-1 (32.0% in whitened space), 6–7× the best end-to-end graph run
   (`src/analysis/retrieval_baseline.py`).
2. **The vector → text mapping was learned from too little data on a template-friendly loss.**
   Wherever the vector carried little, more decoder capacity bought a template.
3. **LM loss is the wrong selection criterion**; select on generated text.
4. **Change one thing at a time**; three diagnoses in a row (steps 3 → 4 → 6) reversed and
   misdirected the next build.
5. **Assert properties on the exact tensor the next stage consumes** (run 8).
6. **Stop rules belong in code**, not prose (run 8 ran 1,250 steps past a written stop rule).

## 5. The reader/estimator split (7–10 Oct)

### Run 9 (job 14529557) — stopped at epoch 1

Train only the reader: own (whitened) embedding → abstract, on **all 90,461** training papers,
with synthetic noise (each vector rotated to a cosine drawn from U[0.3, 1] with the truth), gate
initialised open (0.2). The canary stop rule fired at the end of epoch 1 (937 steps, 1.91 h):
**1 distinct text for 16 validation papers**. `valid/delta` 0.0003 (step 4 at the same step:
0.022); gate ×1.2 (step 4: ×13). Three input changes at once vs step 4 (whitening, gate 0.2,
noise from step 0); whitening is the leading suspect, not isolated.

### Run 10a (job 14531703) — step 4's recipe on 90k papers, trained on real estimates

Raw features, gate from 0, input = own embedding or with p = 0.5 the reference mean; stop
rules from epoch 2. ~12 h training.

| arm (run 10a reader) | judge acc | ≥ 3 | coherence | TF-IDF top-1 |
|---|---|---|---|---|
| refs-mean | 2.818 | 52.1% | 7.19 | 15.2% |
| oracle (ceiling) | 3.407 | 70.5% | 6.99 | 32.6% |
| text baseline | 3.592 | 79.6% | 7.90 | 42.9% |

- Given a perfect vector the reader is within 0.18 of the text baseline; oracle − refs-mean =
  +0.589 (SE 0.051), so most of what is left is in the **estimate**.
- Judge score followed the estimate's cosine to the true embedding: ~+0.6 per +0.1 cosine
  (quintiles 2.33 → 3.21), while the oracle arm stayed flat (3.30–3.57) on the same papers.
- Loops: 17.6% of oracle outputs (13.7% of refs-mean) repeated a sentence (coherence 3.8 vs
  7.7).

### Run 11 (job 14541449) — the final model

Reader (run 10a's recipe, retrained), a GNN grid trained to imitate the target's embedding,
a joint fine-tune, and the 6-gram loop fix — all in one job (`03_final_model.md`). Results in
`06_results.md`.

## 6. Why the final model is not trained fully end-to-end (the argument, with evidence)

- **End-to-end from scratch was tried six times** (steps 1, 2, 5, 6, 7, run 8). With a frozen
  LLM the graph runs never exceeded 4.2% top-1; with LoRA they collapsed (453 → 342 → 34–195 →
  59 distinct outputs) as the LLM learned to ignore a weak, noisy conditioning signal.
- **The gradient path is long and starts closed**: LM loss → 40 layers → prefix → gate (0 at
  init) → adapter → GNN. At gate 0 the GNN gets no gradient; when the gate opens, the signal
  to the GNN is a few nats on a handful of content words among ~230 generic tokens.
- **The staged design fixes the order of learning**: the reader first learns, from clean
  vectors, that the vector carries the content (step 4 showed this works), then the GNN learns
  an objective (cosine to the target's embedding) that is directly measurable and correlates
  with the judge.
- **The joint phase of run 11 is the end-to-end step**, but at GNN lr 1e-5 the GNN did not
  move; it changed nothing measurable (−0.03, SE 0.04). A real end-to-end fine-tune (GNN lr
  ~1e-4, possibly from the InfoNCE GNN; ~13 h) is the open experiment (`09`).

## 7. Experiment folder ↔ run map

| run | folder(s) under `experiments/` |
|---|---|
| Vicuna 495-node pair | `2026-08-04_adapter-ep2`, `2026-08-04_vicuna-baseline` |
| Vicuna reference result | `2026-08-15-14207605_adapter`, `2026-08-15-14207605_vicuna-baseline` |
| prompt eras / variants | `_archive/2026-08-20-*`, `_archive/2026-08-21-*`, `2026-08-23-*`, `2026-08-24-*`, `2026-08-25-*` |
| Qwen text pair | `2026-09-10-14364743_qwen14b-baseline`, `2026-09-10-14364743_adapter-qwen14b-gnn2` |
| step 1 / 2 | `2026-09-16-14393649_adapter-qwen14b-graphonly-gnn2`, `2026-09-17-14395431_adapter-qwen14b-graphonly-kv192-gnn2` |
| step 3 / 4 | `2026-09-21-14411691_oracle-qwen14b-kv192`, `2026-09-22-14422070_oracle-qwen14b-kv192-lora` |
| step 5 | `2026-09-24-14437531_graphonly-qwen14b-kv192-lora` |
| step 6 | `2026-09-25-14450409_ntok64-gatv2-lora64` |
| step 7 | `2026-09-29-14462189_rank-gatv2-lora16` (+ `-gate0`), `2026-09-29-14463510_rank-gatv2-lora16` |
| run 8 | `2026-10-05-14482476_run8-whitened` (+ `-gate0`) |
| retrieval floor | `2026-10-07_retrieval-baseline` (generations only, not judged) |
| run 10a | `2026-10-08-14531703_run10a-decoder-{oracle,refs-mean,zero}` (zero not judged) |
| run 11 | `2026-10-09-14541449_run11-{reader-zero,reader-gnn,reader-refs-mean,reader-oracle,joint,looptest-ngram0,looptest-ngram6}` |
| studies | `prompt_sensitivity_gpt`, `prompt_sensitivity_qwen`, `rubric_sensitivity`, `sentiment_analysis` |
