# Chapter 5 — Method: the final model

Approved 2026-11-02. File: `content/05-method.tex`, label `chap:method`. Target ≈ 8–9 pages.
Describes run 11 only (`thesis_context/03_final_model.md`, `src/adapter/model.py`,
`hpc/komondor/jobs/run11_all.sh`).

## Decisions taken (2026-11-02)

- **Term:** "reference mean" (glossary), not "reference average". §4.6.1 changed to match.
- **No earlier models in this chapter.** Every design choice is explained by its reason in
  the final model; earlier runs are not cited as evidence here (see "Parked evidence" below).
- **Phase 3 is presented as the end-to-end training step** (GNN, adapter and LoRA trained
  together). What it did (the GNN did not move at lr 1e-5) is shown later, in Ch. 7–8.
- **Figures:** one global pipeline figure at the start, plus one smaller, more detailed figure
  for each part of the model (GNN estimator, prefix adapter, reader). New draw.io diagrams,
  made with the `thesis-figure` skill, shown to the author before insertion.
- **Multi-head attention** is named with the correct term here (GATv2 with 4 attention heads,
  concatenated); Ch. 2 introduced the single-head form.
- **No TF-IDF definition here.** Checkpoint selection is described in words (generated
  abstracts on fixed validation papers); the instrument is defined in Ch. 6.
- **Decoding stays in this chapter.**
- **Engineering details of prefix injection are not in this chapter for now** (list below).

## Skeleton

### Chapter introduction (1 paragraph) + Figure 5.1
- The model writes a paper's abstract from one input: the embeddings of its references; it
  sees no text of the paper or its references.
- Two halves: the **estimator** (GNN: where in embedding space is this paper?) and the
  **reader** (prefix adapter + LoRA + frozen Qwen3-14B: what abstract does this vector
  describe?). Each trainable and measurable on its own.
- **Fig. 5.1 (new):** references' embeddings + `[MASK]` → GATv2 + head → one 1024-d vector →
  prefix adapter → key/value prefixes at 40 layers → Qwen3-14B + LoRA, with the fixed prompt
  → abstract. Trainable red tint, frozen grey (skill colours).

### 5.1 The GNN estimator (+ Fig. 5.2, detailed)
- P1 **Input and masking.** Target + all its recorded references (no sampling); edges
  cited → citing (§4.3). The target's row is replaced by one learned `[MASK]` vector shared by
  all targets (initialised to unit norm), so it cannot carry paper-specific information; it
  marks the unknown node. (`model.py` `GraphEncoder.forward`)
- P2 **One GATv2 layer**, notation of Eqs. (2.x) `eq:gatv2-score`–`eq:gatv2-output`.
  Multi-head attention: 4 heads × 256 channels, concatenated to 1024. PyG adds self-loops by
  default, so the softmax runs over the references and the `[MASK]` state, consistent with
  Eq. `eq:gatv2-softmax`. No activation on the output layer. One layer → only direct
  references. (`model.py` `_conv`: `GATv2Conv(in, out//heads, heads=4, concat=True,
  dropout=0)`; ≈ 3.15M params with head, `03` §2)
- P3 **Head, normalisation, objective.** Linear 1024 × 1024, then L2 normalisation → output in
  the space and at the norm of a real Voyage embedding, which is what the reader is trained
  on. Loss 1 − cos(estimate, target's own embedding). Reason (no history): the reader already
  reads embeddings, so the GNN's job is to produce one; cosine is directly measurable.
- P4 **Reference mean** = normalise(mean of reference embeddings): no parameters, no
  attention; the structure-agnostic estimator the GNN is compared with; links to §4.6.1.
- **Fig. 5.2 (new):** the target node with `[MASK]`, its references, attention weights α,
  4 heads concatenated, head, normalise → estimate; cosine to the own embedding during training.

### 5.2 The prefix adapter (+ Fig. 5.3, detailed) — revised 2026-11-02 (author)

Order: the required output shape → the obvious design and why it fails (parameter count vs the
LLM) → the factorised design inspired by multi-head attention → base + gate → positions.

- P1 **The output shape the adapter must produce.** Qwen3-14B has 40 layers; with GQA (§2.2.3)
  each layer stores keys and values for 8 KV heads of dimension 128 (40 query heads share them).
  A prefix of 32 positions therefore needs, in every layer, one key tensor and one value tensor
  of shape 8 × 32 × 128 = 32,768 numbers each; in total 40 × 2 × 32,768 = 2,621,440 numbers per
  paper, all computed from one 1024-d GNN estimate. (Qwen3-14B `config.json`, quoted in
  `docs/model_swap_qwen.md` §4: 40 layers, 40 query heads, 8 KV heads, head_dim 128, hidden
  5120; cite the Qwen3 report `yang2025qwen3` — needs a bib entry.) Reference §2.3.3 / Eq.
  `eq:prefix` for where the prefix enters, and §2.2.3 for the KV cache.
  - The 32 prefix tokens are stated as a setting, without the Vicuna history (style rule).
- P2 **The obvious design: a separate mini adapter for each layer's keys and values.** 80
  independent maps (40 layers × key/value), each a linear map 1024 → 32,768:
  each 1024 × 32,768 + 32,768 = 33,587,200 parameters; × 80 = **2,686,976,000 ≈ 2.69 B** —
  identical to one direct linear map 1024 → 2,621,440. Compared with the LLM: Qwen3-14B has
  **14,768,307,200 ≈ 14.8 B** parameters (13.2 B without the embedding and output matrices), so
  the adapter alone would be **≈ 18.2%** of the LLM's size (≈ 20% of its non-embedding part),
  and far larger than anything trainable alongside it on one 40 GB GPU (the bf16 LLM weights
  take 29.6 GB, `03` §7). The cost is in the output dimension, not the input.
  - Recounted 2026-11-02 for Qwen3-14B (not Vicuna): LLM count from `config.json` (embeddings
    151,936 × 5,120 and an untied output matrix of the same size; per layer q 5120², k and v
    5120 × 1024, o 5120², MLP 3 × 5120 × 17,408, q/k-norm 2 × 128, two RMSNorms 2 × 5120);
    matches the published 14.8 B. Prefix volume 2,621,440 is the same as in the Vicuna design,
    so the 2.69 B direct-map figure in `03` §2.2 was correct for Qwen too.
  - Decided 2026-11-02: the text uses the 80 linear 1024 → 32,768 adapters (2.69 B, ≈ 18%, ≈ 84× the factorised adapter); NOT the bottleneck variant. Dropped: 80 separate bottleneck adapters 1024 → 192 → 32,768
    (no sharing): 80 × 6,521,024 = **521,681,920 ≈ 0.52 B ≈ 3.5%** of the LLM — shows that
    sharing the decoders, not only the bottleneck, is what removes most parameters.
- P3 **The factorised design, inspired by multi-head attention.** Multi-head attention computes
  all heads with one projection and slices the result into heads (§5.1 / §2.2). Likewise:
  1. a shared **trunk** (Linear 1024 → 1024, GELU, LayerNorm);
  2. **one** linear projection 1024 → 15,360, **sliced** into 80 codes of 192 dims, one per
     (layer, key/value) — each layer gets its own view of the paper;
  3. **two decoders**, one for keys and one for values, each Linear 192 → 32,768 and shared by
     all 40 layers of its role (separate because keys decide where attention goes, values carry
     what is read); the output is reshaped to 8 heads × 32 tokens × 128.
  Parameter counts (printed by the code for run 11, Qwen dims; recomputed 2026-11-02):
  trunk 1,051,648; codes 15,744,000; decoders 2 × 6,324,224 = 12,648,448; base + gate 2,621,520;
  **total 32,065,616 ≈ 32.1 M ≈ 0.22% of the LLM**, ≈ 84× smaller than the per-layer design.
  - Wording: "inspired by" is the author's reasoning; describe it as the design rationale, not
    as a claim from the literature.
  - Possible small table: design | parameters | share of Qwen3-14B (direct / per-layer linear
    2.69 B, 18.2%; [optional] per-layer bottleneck 0.52 B, 3.5%; factorised 32.1 M, 0.22%).
- P4 **Base + gate:** prefix = base + gate · delta. Base: learned, same for every paper (plain
  prefix tuning, init std 0.02). Gate: 80 scalars, init 0, unbounded. Reasons: training starts
  as plain prefix tuning (a known start); gate = 0 at evaluation gives "graph off" on the same
  weights; per-layer gate values show where the graph is used. Cost: at gate = 0 nothing behind
  the gate (trunk, codes, decoders, GNN) gets gradient; the gate gets gradient ⟨∂L/∂prefix,
  delta⟩ and must open itself. Caveat: gate = 0 is "graph off", not "no prefix".
  (Do NOT use the "17× larger than base" number: it was measured on the original design.)
- P5 **Positions:** prefix at positions 0–31, prompt from 32, in training and generation; keys
  injected already in RoPE space (§2.2.4).
- **Fig. 5.3 (new):** the target shape on the right (per layer: a key block and a value block of
  8 heads × 32 tokens × 128, for 40 layers); left to right: GNN estimate → trunk → one
  projection sliced into 80 codes → key decoder / value decoder (shared across layers) → delta;
  base + gate · delta → prefix in each layer's cache. The naive per-layer design is not drawn
  (numbers in the text or table).

### 5.3 The reader (no figure) — revised 2026-11-04 (author)

Decisions (2026-11-04): no Fig. 5.4 (Fig. 5.1 already shows the reader); no parameter totals
here (they go in §5.6); the prompt is summarised with its important parts, the full prompt goes
in App. B; the q/v choice is kept as a design justification. **Main goal (author):** the reader
must understand how the prefix and LoRA complement each other, why both are necessary and what
each one is responsible for.

- P1 **Qwen3-14B**, frozen, bf16: 40 layers, hidden 5120, 40 query / 8 KV heads, head dim 128
  (`yang2025qwen3`, exists). Frozen: its weights stay as published.
- P2 **Division of labour: prefix vs LoRA.** The prefix is the only per-paper input: it is
  computed from the vector and differs for every paper. LoRA is a set of weight updates shared
  by all papers, so by construction it carries no information about any single paper. Prefix =
  *what* to write about (the paper's content); LoRA = *how* to use the prefix: the frozen model
  was never trained to read keys and values produced by an adapter, and LoRA adapts it to read
  them and to write an abstract from them. Neither suffices alone: without the prefix the model
  has nothing paper-specific; without LoRA the frozen model is not adapted to the prefix.
  Evidence for "both necessary" is NOT given here (style rule: no earlier runs in the method
  chapter; instruments not yet defined): forward reference to Ch. 8 (zero input, gate = 0,
  frozen vs LoRA reader), as a TODO until Ch. 8 exists.
- P3 **LoRA settings and the q/v choice.** r 16, α 32, dropout 0.05, on q_proj and v_proj of all
  40 layers (Eq. `eq:lora`, §2). Why q and v: the prefix arrives as finished keys and values, so
  the key projection never acts on it; the query projection decides how strongly each token
  attends to the prefix, and the value projection adapts what the prompt and abstract tokens
  carry. Shapes under GQA (q 5120 → 5120, v 5120 → 1024) only if needed; the count goes to §5.6.
- P4 **Fixed prompt**, identical for every paper (ChatML, thinking disabled, 241 tokens), so the
  only thing that differs between papers is the prefix. Summary of the important parts: the
  context encodes the reference list of a paper whose abstract is missing, and that paper is a
  newer work citing them; the abstract should cover, in order, the problem, what the cited work
  established, what the paper does (described by mechanism, not by name) and what it evaluates;
  where the context leaves the contribution undetermined, a correct general statement is
  preferred to a specific guess; written in the authors' voice without mentioning the cited
  works; 150–180 words of plain prose. User turn: "Write the abstract." Full prompt: App. B
  (TODO until App. B exists).
- P5 **Loss:** next-token cross-entropy on the abstract tokens only, prompt masked
  (Eq. `eq:lm-loss`), against the cleaned targets (§4.7).

### 5.4 Decoding — revised 2026-11-04 (author): keep it short

Decisions (2026-11-04): no generation cap, no repetition penalty (not used), no "run 11" (the
reader has no context for it); no TF-IDF and no mean length here, only the loop rate; no P3 —
only one clause that the decision was made on evaluation papers with a rule fixed in advance;
n-gram blocking described in words, no citation.

- P1 **Greedy decoding** (`sec:decoding`), the same for every input variant; deterministic, so
  reruns reproduce and differences between input variants come from the input, not sampling.
- P2 **Loops and the n-gram ban.** Greedy decoding can fall into repeating a sentence. Real
  example (no ban), shown as a short, shortened quote with the repetition underlined: node
  31565 in `experiments/2026-10-09-14541449_run11-looptest-ngram0/generations.csv` — "We also
  show that the block-wise approach can be used to reduce the latency of SC decoding by a
  factor of O(N)." repeated three times at the end of a polar-codes abstract. The ban: no
  sequence of 6 tokens may occur twice in one output (Hugging Face `no_repeat_ngram_size = 6`,
  counts tokens, `src/adapter/generate.py:447`; the run notes say "words", which is inexact).
  Decided on 200 evaluation papers with a rule fixed in advance (keep if it lowers the loop
  rate at a small, pre-set cost); loop rate 14.0% → 0% (28 of 200 outputs without the ban;
  loop = an output repeating a sentence of ≥ 5 words, `05` §2.6). Kept for every input variant.
- Revised by review 2026-11-04 (author-pasted): the 320-token limit IS stated (overrides the
  earlier "no cap"), assessed after the fact against the cleaned training abstracts (Qwen3
  tokenizer, 90,461 papers, `training_sets_run9.json`, `text.clean_abstract`): median 191,
  p95 339, mean 199.3, max 2,008 tokens; 6,566 (7.26%) exceed 320. The statistics did not
  choose the limit. Example shows two repetitions; the test input is "the target's own
  embedding"; the rule is "identification-accuracy reduction within the predefined tolerance"
  with a pointer to Ch. 6.
- Parked for Ch. 7: how often generated outputs reach 320 tokens without an end-of-sequence
  token (truncation).
- Parked for Ch. 9 (evaluation limitations): the decoding constraint was decided on 200
  evaluation papers.
- Parked for Ch. 6: whether the text baseline was decoded with the ban (`05` §1 says "graph
  arms"); the TF-IDF and length side of the loop test (`06` §5).

### 5.5 Training in three phases — WRITTEN 2026-11-04, review applied 2026-11-04

Decision (author): §5.5 shows the training settings and objectives only; the training curves
go to Ch. 7 "Training dynamics" (GNN grid curves: Ch. 7 or Ch. 8, undecided). No figure in §5.5.

Review applied 2026-11-04 (author's pasted review):
- selection motivation: validation loss vs independently generated text (no "generic language"
  claim); MRR explained step by step with the 96-token budget = cheaper validation, final
  evaluation uses §5.4 settings;
- input mix justified as direct text embedding vs imperfect neighbourhood-derived representation,
  "not identical to GNN estimates";
- InfoNCE: motivation → operation → implementation (256 batch + 4,096 sampled from the joint
  graph, own-id duplicates excluded — verified in `src/adapter/pretrain_encoder.py`, τ = 0.05,
  weight 0.05 "to compensate for the explicit 1/τ scaling");
- joint phase: two purposes, effects assessed in Ch. 7–8;
- "fixed-size training epochs"; table headers "parameters updated", "checkpoint selection".

Moved out of §5.5 (must land elsewhere):
- 937 optimiser steps per epoch (5,622 total) and 2,811 joint steps → App. C (hyperparameters);
- "epoch 5 of 6 selected" (reader) and the selected joint epoch → Ch. 7 / checkpoint summary;
- stop rules / canary and hardware → App. A.
- Added 2026-11-04 (author): resource cost as a reason for fixed-size epochs; phase 1 = 11.5 h
  on one A100 40 GB (`03` §7, `08`). Full hardware description still → App. A.

Original plan below (kept for reference):

General rules carried over: no "run 11", no earlier runs as evidence, describe instruments in
words before Ch. 6 defines them (forward reference). Keep the logical order (reader, GNN,
joint); the actual run order (GNN grid first) is not mentioned. Stop rules / canary and the
hardware sentence: not discussed yet (proposed: App. A).

- P1 **Why staged** (agreed): each half is first trained on a clean, separately measurable
  objective, then all parts together. Forward reference to Ch. 8 (end-to-end from scratch).

- P2 **Phase 1, the reader** (agreed, with additions):
  - Trains the prefix adapter + LoRA; Qwen frozen; GNN unused. Learns to turn a vector into the
    abstract it describes.
  - **Objective (must be explained):** the next-token cross-entropy of §5.3 on the abstract
    tokens, with the prefix computed from the input vector (`03` §4.1).
  - **Input mix (must be justified):** for each paper, with probability 0.5 its reference mean,
    otherwise its own embedding (a paper without sampled references keeps its own embedding).
    Justification, design terms only: the own embedding is the cleanest description of the
    target, so it teaches that the vector carries the content; the reference mean is an
    imperfect estimate of the kind the reader receives at test time (from the GNN), so
    training on it exposes the reader to real estimation error rather than only perfect
    inputs. Source `03` §4.1, §6 ("trains on the real estimation error"). Do NOT cite run 9's
    synthetic-noise failure (earlier-run rule).
  - Data: 6 epochs × 15,000 papers, each epoch a slice of one seeded permutation of the 90,461
    training papers (so consecutive epochs see different papers); effective batch 16
    (micro-batch 1 × 16 accumulation), 937 steps/epoch, 5,622 total.
  - Optimiser: AdamW lr 1e-4, 100 warm-up steps then constant, gradient clip 1.0. **Weight
    decay is 0 → do not mention it** (author).
  - **Generation-based score (must be explained):** at every epoch end the reader writes an
    abstract (96 tokens) for 128 fixed validation papers; each generated text is compared with
    the 128 true abstracts by TF-IDF similarity (`identify.py`: TfidfVectorizer, sublinear tf,
    English stop words, fitted on the gold abstracts only; ties count against the generation),
    and the rank of the paper's own abstract gives the mean reciprocal rank (MRR; chance top-1
    1/128 = 0.8%). It measures whether the generated text identifies its paper among the
    others. Why generation, not LM loss: LM loss is dominated by generic scientific English
    (state as design reason, `03` §6; not the earlier-run evidence). Full definition: Ch. 6.
  - Selection: the epoch with the best MRR on the reference-mean input (the realistic input) →
    epoch 5 of 6.
  - **Figure (author: "I need the diagram of training here")** — OPEN: which diagram?
    (a) a new schematic of phase-1 training (input vector: own embedding / reference mean with
    p = 0.5 → prefix adapter → frozen Qwen + LoRA → LM loss on the abstract; red = trained);
    (b) an existing training-curve figure, e.g. `diagrams/data/run11-reader-14541449/figures/
    reader_identification_mrr.pdf` (validation MRR per epoch, own embedding vs reference mean)
    — but results/curves were planned for Ch. 7.

- P3 **Phase 2, the GNN estimator** (agreed, with additions):
  - No LLM; 83,259 training papers with ≥ 1 reference; 30 epochs, batch 256, lr 1e-3.
  - Objective: 1 − cos(estimate, own embedding) (§5.1), optionally + 0.05 · InfoNCE.
  - **InfoNCE (must be explained):** a contrastive loss — each estimate (anchor) must pick its
    own paper's true embedding among the 256 in the batch plus 4,096 randomly sampled
    embeddings; similarities are cosines divided by a temperature τ = 0.05, followed by a
    softmax cross-entropy; false negatives masked. It rewards estimates that are distinctive
    (closer to their own paper than to others), whereas the cosine term only rewards closeness.
    Weight 0.05 = τ so that the two terms pull about equally (1/τ = 20 scales InfoNCE's
    gradient). Needs a bib entry: van den Oord et al. 2018 (verify the record). (`03` §4.2)
  - Grid: {GATv2, GraphSAGE} × {1, 2 layers} × {cosine, cosine + InfoNCE}; selection on
    validation mean cosine → 1-layer GATv2, cosine. Grid results → Ch. 8.
  - **Figure (author: "here we should insert the diagram of training")** — OPEN: a new
    schematic of GNN training, or an existing curve, e.g.
    `diagrams/figures/gnn_comparison/gnn_cosine_validation.pdf` (validation cosine of the
    eight variants)? (Grid results were planned for Ch. 8.)

- P4 **Phase 3, joint fine-tune** (agreed, with additions):
  - Starts from the phase-1 reader and the phase-2 winner; the reader now receives the GNN
    estimate. Loss LM + 1 · (1 − cos); lr GNN 1e-5, reader 1e-4 (GNN arrives trained);
    3 × 15,000 papers with references, 2,811 optimiser steps; selection on the MRR with the
    GNN input.
  - **Why joint, what was expected (must be explained):** (i) the reader was trained on own
    embeddings and reference means, never on the GNN estimate, so it should adapt to the
    actual input it receives at test time; (ii) the LM loss can pass gradients into the GNN,
    so the estimator can learn what helps the generated abstract, not only what is close in
    cosine; the cosine term keeps the estimate in the space the reader understands, so the LM
    loss cannot drag it elsewhere. The low GNN learning rate protects the trained estimator.
  - Outcome NOT stated here (the GNN did not move; reader specialised) → Ch. 7–8.

- **Table 5.x:** phase | trains | data | objective | selection. Full hyperparameters: App. C.
- Resolves the §5.1 TODO ("reference to the training section").

Open for the discussion:
1. Which "diagram of training" in P2 and P3: new schematics, existing curves, or one schematic
   of all three phases?
2. Stop rules / canary and the hardware/run-time sentence: leave out or App. A?
3. Table 5.x: keep, or prose only?
4. How much of the generation-based score here vs Ch. 6 (above: a short definition here).

### 5.4 Model size — WRITTEN 2026-11-04 (moved before Decoding/Training; author)
- Table `tab:model-size` (millions, phases updated); exact calculation in App. A.1
  `app:parameters` (trainable components; Qwen3-14B from config.json, bib `qwen2025config`;
  direct mapping 2,686,976,000). GNN count 3,151,872 verified by instantiating the 1-layer
  GATv2 estimator on CPU. Resolves the §5.2 appendix TODO.

#### (earlier plan) 5.6 Parameter summary (added 2026-11-04, author)
- A short closing section of the chapter that sums up the parameter counts: GNN estimator,
  prefix adapter (by component), LoRA (10,485,760), reader total 42.6 M, joint total 45.7 M,
  against ≈ 14.8 B frozen Qwen3-14B (`03` §1 table). Likely a small table. Plan in detail when
  §5.5 is done.

## Parked: engineering details of prefix injection (not in Ch. 5 for now; App. A candidate)

Keep for later — a wrong offset produces fluent text and a falling loss, not an error.
- Read-only training cache (`PrefixCache`): a mutating cache grows under gradient-checkpoint
  recomputation.
- Explicit `cache_position` when `use_cache=False` (gradient checkpointing forces it);
  otherwise silent logit error.
- Patched `generate()` / `prepare_inputs_for_generation`, which overwrites `cache_position`;
  under transformers 4.51.3 the library otherwise silently drops the first 32 body tokens.
- Verification by splitting a real sequence into prefix + body on a small random model
  (logit error ≈ 1e-7 with the fix; most tokens wrong without it).
- Sliced loss: logits only at supervised positions, numerically identical (4.8e-07), saves
  ≈ 2.4 GB.
- LoRA weights held in fp32 (AdamW on bf16 drops small updates).
- Non-reentrant gradient checkpointing.
Sources: `04` §2.1, §3; `03` §2.2–2.3; `08`.

## Parked from §5.2 (removed in the 2026-11-02 review; App. A or later chapters)

- **TODO (author, 2026-11-02): an appendix shows the exact calculation** behind §5.2's rounded
  numbers: 80 × (1024 × 32,768 + 32,768) = 2,686,976,000; Qwen3-14B = 14,768,307,200 from
  `config.json` (layer-by-layer); the factorised adapter by component (below). A red TODO in
  §5.2 marks where the reference goes.

- Prefix keys/values bypass the model's k/v projections; the keys skip Qwen3's per-head key
  normalisation and the RoPE rotation (they are learned in cache form, "RoPE space").
- Component parameter counts: trunk 1,051,648; code projection 15,744,000; decoders
  12,648,448; base + gates 2,621,520; total 32,065,616 (≈ 0.22% of Qwen3-14B; ≈ 84× fewer than
  the 2.69 B direct map). Use in a table only if a later discussion needs it.
- Gate gradient: dL/dΔ = g · dL/dP, gate gets ⟨dL/dP, Δ⟩. Gate magnitudes alone are NOT a
  reliable measure of layer importance (a small gate can multiply a large Δ).

## Parked: evidence from earlier runs (not in Ch. 5; possible use in Ch. 8)

- Frozen Qwen cannot read the prefix: oracle frozen 0.3% vs oracle + LoRA 12.0% TF-IDF top-1
  (steps 3–4).
- Judge score follows the estimate's cosine: ≈ +0.6 judge points per +0.1 cosine (run 10a,
  old judge).
- Untrained delta ≈ 17× base at init (original design).
- LM loss misled earlier runs; generation-based selection adopted.
- Stop rules in code because a prose rule was overrun (run 8).
- Raw features and gate from 0 were the only configurations that opened the channel.
- Keep the GNN head (run 8 failed by discarding it).
- One prefix vector, not per-reference tokens (step 6).

## Citations needed

- `brody2022attentive` (exists), `li2021prefix`, `hu2022lora`, `ainslie2023gqa`,
  `su2024roformer` (exist, Ch. 2).
- Qwen3 technical report — needs a bib entry (`01_literature.md`).
- InfoNCE (van den Oord et al. 2018) — needs a bib entry if InfoNCE is named in §5.5.
- PyG (if the self-loop default is attributed to the library) — needs a bib entry, or say it
  in words.

## Open items for writing

- How to report the decoding numbers (TF-IDF) before Ch. 6 defines the instrument.
- §5.2: whether to show the optional per-layer bottleneck design (0.52 B) and whether the
  parameter comparison is a table or prose.
- The TODO in §4.3 ("reference to the method chapter") becomes `\autoref{sec:...}` of §5.1
  once written.
