# The run-11 model: components and training

Written 2026-10-10 from the code (`src/adapter/`), `docs/run9_embedding_decoder.md` §8–9, the
job script `hpc/komondor/jobs/run11_all.sh` and the W&B project `dytag-adapter` (job
14541449). This is the model behind the run-11 rows of `chosen_evaluations.md`.

## What it does

Given one paper, the model has to write that paper's abstract without seeing it. The only
input is the paper's place in the citation graph: the papers it cites, as Voyage embeddings
(voyage-4-large, 1024-d, one vector per abstract). The model never sees any reference text.

```
references' embeddings ──► GNN + head ──► one 1024-d vector ──► prefix adapter ──► 32 key/value
(the paper's own row is    (estimate of the   (unit norm)        (32M params)       tokens at each of
 masked)                    paper's own                                              Qwen3-14B's 40 layers
                            embedding)                                                   │
                                     fixed prompt: "Write the abstract." ──► Qwen3-14B + LoRA ──► abstract
```

The design idea: **split the problem in two.**
- **The GNN** answers "where in embedding space is this paper?" It turns the references into
  an estimate of the paper's own embedding.
- **The reader** (prefix adapter + LoRA + Qwen) answers "what abstract does this embedding
  describe?" It turns a vector into text.

Each half can be trained and measured on its own. If the reader is given the paper's true
embedding (the "oracle"), it shows the ceiling of the reader. If it is given the GNN estimate,
it shows the full model.

## The components

| component | trainable parameters | what it does |
|---|---|---|
| Voyage embeddings | 0 (fixed input) | one 1024-d unit vector per paper, from its abstract |
| GNN (1-layer GATv2) + linear head | ~3.2M | references → estimate of the paper's own embedding |
| Prefix adapter | ~32.1M | one vector → key/value prefixes for all 40 layers |
| LoRA (rank 16 on q and v projections) | ~10.5M | adapts Qwen to read the prefix and write in the abstract register |
| Qwen3-14B | 0 (frozen, bf16) | writes the text |

### 1. The GNN: an estimate of the paper's embedding

- **Input:** the paper and its references, each with its Voyage embedding. All references are
  used (no sampling cap).
- **Masking:** the paper's own embedding *is* the answer, so its row is replaced with one
  learned `[MASK]` vector before message passing. A test in `model.py` checks that changing the
  paper's own features does not change the output.
- **One GATv2 layer** (4 attention heads, 1024 wide). Each reference gets a learned attention
  weight, and the output is a weighted combination of the references. A plain mean would give
  every reference the same weight. One layer means it reads only the direct references, not
  the references of references.
- **Linear head + normalisation:** a 1024 × 1024 linear layer, then scaled to unit length. This
  puts the output in the same space as a real Voyage embedding, which is what the reader was
  trained on (`model.EmbeddingEstimator`).

The baseline it has to beat is the **plain reference mean**: average the references'
embeddings and normalise. This has no learned parameters and already reaches cosine 0.816 to
the true embedding.

### 2. The prefix adapter: a vector into Qwen's attention

The vector does not enter Qwen as a word. It becomes **32 extra key/value tokens at each of the
40 layers**. Every real token can attend to them, as if 32 tokens had come before the prompt
that only exist inside the attention (prefix tuning).

That is 40 layers × 2 (key, value) × 8 heads × 32 tokens × 128 = 2.6M numbers per paper. A
direct linear map from 1024 numbers would need 2.7B parameters, so the adapter is factorised
(`model.PrefixAdapter`):

1. **Trunk:** linear 1024 → 1024, GELU, LayerNorm.
2. **Codes:** one 192-d code for each (layer, key/value) pair: 80 codes.
3. **Two decoders**, one for keys and one for values, each shared across all 40 layers. They
   expand a code to that layer's 32 × 8 × 128 prefix.
4. **Base + gate:** `prefix = base + gate × delta`.
   - `base` is a learned prefix that is the same for every paper.
   - `delta` is the paper-specific part from steps 1–3.
   - `gate` is one learned number per (layer, key/value), started at 0. So training starts as
     plain prefix tuning with the graph disconnected.
   - Setting the gate to 0 at evaluation time gives the "no graph" ablation on the same
     weights. This is what `valid/delta` measures.

Keys and values have separate decoders because they do different jobs: keys decide where
attention goes, values carry what is read.

### 3. LoRA and the frozen Qwen3-14B

- **Qwen3-14B** (40 layers, 40 query heads, 8 key/value heads) stays frozen.
- **LoRA rank 16** (alpha 32, dropout 0.05) on the query and value projections of every layer.
  It lets Qwen learn to use the prefix and to write in the style of an abstract.
- **The prompt is the same for every paper:** a fixed system prompt that explains the context
  encodes a paper's reference list and asks for a 150–180 word abstract, then the user message
  "Write the abstract." (`text.GRAPH_ONLY_SYSTEM`; ChatML, thinking off). Everything
  paper-specific comes through the prefix.
- **Positions:** the prefix takes positions 0–31, and the prompt starts at 32.
- **Loss:** next-token cross-entropy on the target abstract tokens only. The prompt is not
  scored.

### 4. Decoding

- Greedy, up to 320 new tokens, no repetition penalty.
- **`no_repeat_ngram_size = 6`:** no 6-word sequence may repeat. This was decided by the job
  itself on 200 papers: the loop rate fell from 14.0% to 0%, and TF-IDF top-1 went from 52.0%
  to 51.5%, within the allowed 2 points. Every run-11 arm uses it.

## The training process

One Slurm job on Komondor (A100 40 GB) ran all three phases: about 18 hours of training, plus
the evaluation generations. Each phase can be resumed separately.

| phase | trains | data | objective | time |
|---|---|---|---|---|
| 1. Reader | prefix adapter + LoRA | 6 epochs × 15,000 of the 90,461 training papers | LM loss; input is the paper's own embedding, or with p = 0.5 its reference mean | 11.5 h |
| 2. GNN grid (run first) | GNN + head (no LLM) | 83,259 training papers with ≥1 reference | 1 − cosine to the paper's own embedding | ~31 min for all 8 |
| 3. Joint | GNN + head + adapter + LoRA | 3 epochs × 15,000 papers with references | LM loss + 1 × (1 − cosine to own embedding) | 5.7 h |

The splits: 90,461 training, 806 validation and 945 evaluation papers. The evaluation set is
read once, at the end.

The phase numbers follow `run9_embedding_decoder.md` §8. The job runs the GNN grid first
because it takes minutes and the two phases do not depend on each other.

### Phase 1: the reader

- **What it learns:** to turn a vector into the abstract it describes.
- **Training input:** each paper's own embedding (the oracle), or, with probability 0.5, the
  mean of its references' embeddings (`--refs-mix 0.5`).
  - The oracle half teaches the reader that the vector carries the content.
  - The reference half shows it the kind of imperfect estimate it will get at test time.
  - The GNN is not used here.
- **Settings:**
  - micro-batch 1 × 16 gradient accumulation = effective batch 16, so 937 optimizer steps per
    epoch and 5,622 in total;
  - learning rate 1e-4 with 100 warmup steps, then constant;
  - gradient clipping 1.0, no weight decay;
  - gate started at 0.
  - 42.6M trainable parameters.
- **Validation:** every 250 steps (LM loss) and at every epoch end (generation).
  - `valid/delta`: LM loss with the gate forced to 0, minus LM loss with the graph. It shows
    how much the vector helps the model predict the abstract.
  - `val_id`: the model writes 96 tokens for 128 validation papers from each input. Each text
    is ranked by TF-IDF against those papers' real abstracts, giving MRR and top-1 (chance:
    1/128 = 0.8%). This is measured on generated text, because LM loss alone has misled
    earlier runs.
  - **Selection:** the epoch with the best `val_id` MRR on the **reference-mean** input, since
    that is the realistic input.
  - Stop rules (canary diversity, oracle top-1 ≥ 10× chance from epoch 2) did not fire.

**What happened** (W&B `run11-reader-14541449`):

| epoch end | valid/delta (nats) | gate mean abs. | oracle MRR / top-1 | refs-mean MRR / top-1 |
|---|---|---|---|---|
| 1 | 0.024 | 0.019 | 0.162 / 7.8% | 0.126 / 4.7% |
| 2 | 0.052 | 0.045 | 0.350 / 21.1% | 0.328 / 21.1% |
| 3 | 0.071 | 0.066 | 0.503 / 37.5% | 0.428 / 30.5% |
| 4 | 0.078 | 0.080 | 0.604 / 47.7% | 0.479 / 33.6% |
| **5 (selected)** | 0.082 | 0.092 | 0.625 / 48.4% | **0.530** / 41.4% |
| 6 | 0.085 | 0.102 | 0.704 / 59.4% | 0.528 / 39.8% |

- **The graph channel opened from the start.** `valid/delta` rose from 0 to 0.085 nats. The
  gate grew steadily from 0 to 0.10 and never levelled off.
  - The graph's share of the prefix norm, measured as ‖gate·delta‖ / ‖base‖, went from 0 to
    about 12.
  - The training LM loss fell from 3.1 to about 2.15 within the first 500 steps, then slowly
    to about 2.13.
- **The reader was still improving when it stopped.** Oracle MRR kept rising (0.62 → 0.70)
  while reference-mean MRR was flat, and the learning rate never decayed. More epochs would
  probably help the oracle input more than the realistic one.

### Phase 2: the GNN grid

Eight variants: {GATv2, GraphSAGE} × {1, 2 layers} × {cosine, cosine + 0.05 × InfoNCE}.
- 30 epochs, batch 256, learning rate 1e-3.
- The epoch is selected on validation cosine.
- The winner is the variant with the best validation cosine (`select_encoder.py` →
  `winner.pt`).

Cosine is the main loss because, in run 10a, the judge score followed the estimate's cosine to
the true embedding (+0.6 judge points per +0.1 cosine). InfoNCE is optional and weighted by
0.05. Its gradient is scaled by 1/0.05, so at full weight it would take over.

| variant | val cosine | val top-1 | eval cosine | eval top-1 |
|---|---|---|---|---|
| plain reference mean (no training) | 0.816 | 63.5% | 0.823 | — |
| **GATv2 1-layer, cosine (winner)** | **0.828** | 70.5% | 0.833 | 70.8% |
| GATv2 1-layer, cosine + NCE | 0.820 | 72.3% | 0.826 | 75.2% |
| SAGE 1-layer, cosine | 0.820 | 67.2% | 0.825 | 68.4% |
| GATv2 2-layer, cosine | 0.822 | 64.4% | 0.822 | 63.2% |

The other four variants are in `run9_embedding_decoder.md` §9. Top-1 means retrieving the
right paper among all validation (806) or evaluation (945) papers by cosine.

**The winner's training curve** (validation): cosine 0.807 after epoch 1 → 0.818 (3) → 0.825
(15) → 0.828 (29). It is still rising slowly at the end, with most of the gain in the first few
epochs. Its outputs are less alike from paper to paper than the mean's: average pairwise
cosine is 0.53 against 0.65.

**Reading:** all eight variants are within 0.02 cosine of each other and of the plain mean.
Two layers are worse than one. InfoNCE trades cosine for top-1. The references limit how well
the paper can be estimated; the architecture does not.

#### The InfoNCE loss

**The idea.** The cosine loss asks: "is the estimate close to the right paper?" InfoNCE asks:
"is the estimate closer to the right paper than to other papers?" It turns training into a
multiple-choice question. For each paper, the GNN's estimate has to pick the paper's real
embedding out of a set of candidates.

**How one step works** (`pretrain_encoder.info_nce`):

1. **Anchors:** a batch of 256 papers. For each, the GNN + head produce an estimate `z`. This is
   the anchor.
2. **Candidates:** 256 + 4,096 embeddings.
   - The batch's own 256 true embeddings. For paper *i*, entry *i* is the right answer, and the
     other 255 are wrong answers that come for free.
   - 4,096 embeddings drawn at random from all 384k papers. These are only rows of the
     embedding matrix, with no GNN pass, so thousands of them cost almost nothing.
3. **Scores:** everything is normalised to unit length, so a dot product is a cosine. Each
   anchor gets a cosine with every candidate, divided by the temperature τ = 0.05. The result is
   a 256 × 4,352 table of scores.
4. **Loss:** softmax over each row, then cross-entropy with the right answer (column *i* for
   row *i*). This is the standard classification loss, where the "classes" are the candidate
   papers. The loss is low when the right paper's score is much higher than all others.
5. **False negatives are masked.** If a random negative happens to be one of the batch's own
   papers, it would push the anchor away from its own target. Such entries are set to −∞
   before the softmax, so they don't count, except the one true answer.

The function also returns the share of anchors whose highest score is the right paper. This is
logged as `pretrain/batch_accuracy`.

**The temperature.** Dividing by τ = 0.05 multiplies every cosine difference by 20. A right
answer at cosine 0.80 and a wrong one at 0.75 become scores 16 and 15, so the softmax separates
them clearly. Without it, cosines between 0 and 1 would give an almost flat softmax and almost
no gradient.

**Why it is weighted by 0.05.** Because of the 1/τ factor, InfoNCE's gradient on a cosine is
about 20 times larger than the cosine loss's. At weight 1 it would decide the training alone.
At weight 0.05 = τ the two terms pull with roughly equal strength. The total loss is
`(1 − cosine) + 0.05 × InfoNCE`. The VICReg terms in the same function were off in run 11.

**What it changed** (GATv2 1-layer, with and without InfoNCE):

| | cosine only | + InfoNCE |
|---|---|---|
| validation cosine | **0.828** | 0.820 |
| validation top-1 (806 papers) | 70.5% | **72.3%** |
| eval top-1 (945 papers) | 70.8% | **75.2%** |
| batch accuracy (256 + 4,096 candidates, training) | 0.65 | 0.70 |
| average cosine between different papers' estimates | 0.53 | **0.45** |

- **It trades closeness for distinctiveness.** The estimate moves a little away from the true
  embedding (−0.008 cosine), but more towards what makes the paper different from its
  neighbours. So it retrieves the right paper more often, and different papers' estimates are
  less alike.
- **The winner was chosen by cosine,** because run 10a's judge score followed cosine. So the
  joint stage started from the cosine-only GNN. The InfoNCE GNN was never put through the LLM.
  Whether its better retrieval gives better abstracts is untested. It is the obvious next GNN to
  try, since the detailed evaluation found the GNN's gain is in getting the specific topic
  right.

#### The W&B metrics of the GNN pretraining

Each of the eight variants is its own W&B run (`run11-gnn-<conv>-l<layers>-<cos|cosnce>-14541449`,
project `dytag-adapter`). Everything is logged once per epoch, so each curve has 30 points. The
example numbers below are the winner's (GATv2 1-layer, cosine), first epoch → last epoch.

**Training metrics:** averaged over all training batches of the epoch, while the weights are
still changing.

| metric | what it is | winner |
|---|---|---|
| `pretrain/loss` | The **InfoNCE** value (above), not the total loss. It is computed in every variant, including the cosine-only ones, where it is only watched and not trained on. The total loss is not logged. | 3.88 → 2.37 |
| `pretrain/train_cosine` | Cosine between the estimate and the paper's true embedding on training papers. The term the cosine loss pushes up (`1 − train_cosine` is that loss). | 0.731 → 0.794 |
| `pretrain/batch_accuracy` | The share of anchors whose top-scoring candidate, out of 256 + 4,096, is the right paper. | 0.42 → 0.65 |
| `pretrain/variance` | The VICReg variance term: per output dimension, how far its spread across the batch is below 1, averaged (0 = every dimension spread enough). | 0.95 → 0.80 |
| `pretrain/covariance` | The VICReg covariance term: how strongly different output dimensions move together. | 0.0007 → 0.048 |

- **The training cosine is below the validation cosine** (0.79 against 0.83). The winner has
  one layer, so dropout is not the cause; this has not been investigated.
- **Ignore variance and covariance in run 11.** They had weight 0, and they are computed before
  normalisation, on an output whose length the cosine loss leaves free. So they mostly track how
  long the vectors grow, not whether they collapse. This is why the 2-layer variants show
  covariance rising to 5–17. `output_cosine` is the collapse check to read instead.

**Validation metrics:** measured at the end of each epoch on the 806 validation papers
(`retrieval()`). The model is in eval mode, and each estimate is compared with the true
embeddings of all 806 papers.

| metric | what it is | winner |
|---|---|---|
| `pretrain/mean_cosine` | The average cosine between each estimate and its own paper's embedding. **This is the selection metric:** the epoch with the highest value is saved. | 0.807 → 0.828 |
| `pretrain/top1` | The share of papers whose own embedding is the closest of the 806. | 63.7% → 69.9% |
| `pretrain/top5` | The share where it is in the closest 5. | 89.5% → 92.2% |
| `pretrain/median_rank` | The median rank of the paper's own embedding. It is 1 as soon as top-1 is over 50%, so in run 11 it is always 1 and says nothing. | 1 → 1 |
| `pretrain/output_cosine` | The average cosine between *different* papers' estimates. Lower means the estimates are more distinct. It catches the collapse that top-1 cannot see: estimates that rank correctly but are almost all the same vector (step 7's failure). | 0.61 → 0.55 |
| `pretrain/epoch` | The epoch, counted from 0. | 0 → 29 |

- **The cosine gate.** An epoch whose `output_cosine` is above `--cosine-gate` cannot be
  selected. Run 11 set the gate to 1, so it was off, since the refs-mean input at 0.65 still
  gave distinct abstracts in run 10a.

**The baseline:** logged once, before training, as `pretrain/refs_mean_<metric>`. These are the
same five validation metrics for the untrained mean of the references (cosine 0.816, top-1
63.5%, top-5 89.5%, output cosine 0.65). Every curve is read against these: a GNN that does not
beat them is not extracting anything the mean doesn't already have.

**The summary (`best/…`):** written once at the end, for the selected epoch.
- `best/valid_<metric>`: the validation metrics at that epoch.
- `best/eval_<metric>`: the same metrics on the 945 evaluation papers. They are measured only
  once, for the selected epoch, and are never used to select.
- `best/epoch`: counted from 1, unlike `pretrain/epoch`. The winner's `best/epoch` is 28,
  which is `pretrain/epoch` 27.

### Phase 3: the joint fine-tune

- **What it learns:** everything together. It starts from the phase-1 reader and the phase-2
  GNN winner, and now the reader is fed the **GNN's estimate**.
- **Loss:** LM loss + 1 × (1 − cosine of the GNN estimate to the paper's own embedding). The
  cosine term keeps the estimate in the space the reader understands, so the LM loss cannot
  pull it somewhere the reader was never trained on.
- **Learning rates:** GNN 1e-5, reader 1e-4. The GNN is 10× slower because it arrives trained.
- **Selection:** the epoch with the best `val_id` MRR on the GNN input.
- 45.7M trainable parameters, 2,811 optimizer steps.

**What happened** (W&B `run11-joint-14541449`):

| epoch end | GNN-input MRR / top-1 | refs-mean MRR / top-1 | valid/delta |
|---|---|---|---|
| 1 | 0.592 / 44.5% | 0.525 / 39.8% | 0.062 |
| 2 | 0.616 / 49.2% | 0.475 / 33.6% | 0.060 |
| **3 (selected)** | **0.643** / 53.1% | 0.484 / 34.4% | 0.061 |

- **The GNN did not move.** `train/cos_loss` stayed at 0.20 and the pairwise similarity of the
  GNN outputs stayed at 0.54 for all three epochs. At lr 1e-5 the GNN was effectively frozen.
  This phase therefore only adapted the reader to the GNN's output, and is not true end-to-end
  training.
- **The reader specialised.** GNN-input MRR rose (0.59 → 0.64) while reference-mean MRR fell
  (0.53 → 0.48). These are 128 papers each, so the SE is about 0.04.
- **`valid/delta` here** is measured on the GNN input. In phase 1 it was measured on the
  oracle input, so the two numbers are not comparable.

## Results on the 945 evaluation papers

Old judge rubric (accuracy 1–10, gpt-5.4-mini), and TF-IDF identifiability:

| arm | input to the reader | judge accuracy | TF-IDF top-1 |
|---|---|---|---|
| Qwen3-14B text baseline | reference abstracts as text | 3.59 | 42.9% |
| reader ← oracle (ceiling) | the paper's own embedding | not judged | 27.8% |
| **joint (full model)** | GNN estimate, jointly fine-tuned | 2.94 | 18.8% |
| reader ← GNN | GNN estimate, pieces trained separately | 2.98 | 18.5% |
| reader ← refs-mean | plain reference mean | 2.86 | 16.5% |
| reader ← zero | zero vector | not judged | 0.1% |

- **GNN against the plain mean, same reader:** +0.113 judge points (SE 0.037). With the
  detailed evaluator (`detailed_evaluation.md`), the subfield, problem and task scores are each
  about +0.15 on a 0–3 scale (z ≈ 5).
- **The joint phase adds nothing measurable:** −0.03 (SE 0.04) against reader ← GNN.
- **The zero arm shows that the content comes from the vector,** not from LoRA having memorised
  a generic abstract.

## Limitations and open questions

- **No true end-to-end training yet.** The GNN did not move in phase 3. A real run would use a
  GNN learning rate of about 1e-4 and possibly start from the InfoNCE GNN (best top-1). It
  would take about 13 h.
- **The estimate is the bottleneck, not the GNN architecture.** All eight GNNs are within 0.02
  cosine of the plain mean. The gap that matters is the estimate against the oracle (18.5% vs
  27.8% top-1). The worst cases are papers whose references do not describe them.
- **Training and evaluation papers differ.** Training papers have a median of 5 references,
  validation and evaluation papers 45–52. 44% of core papers lack a full bibliography (S2
  coverage).
- **The reader was undertrained for the oracle input:** oracle MRR was still rising at
  epoch 6, and the learning rate never decayed.
- **This reader is one sample of its recipe.** Run 10a's reader, trained the same way, had a
  stronger oracle arm (32.6% against 27.8% top-1).
- **The joint loss weights are reasoned, not tuned.** The cosine weight is 1 and the GNN
  learning rate is 1e-5.
