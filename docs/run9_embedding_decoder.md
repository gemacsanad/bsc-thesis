# Run 9 — the embedding → abstract decoder

Branch `feature/embedding-decoder`, from `4605c9a` (run 8). Written 2026-10-07. Built, verified
on CPU, **not yet run**.

---

## 1. Why: what eight soft-prompt runs add up to

All rows rescored with one instrument. Top-1 means a generation's closest gold abstract by
TF-IDF, among the 945 eval papers, is its own; chance is 0.11%. Domain means the generated
abstract is classified into the same arXiv category as the gold paper.

| run | judge acc | top-1 | distinct | domain |
|---|---|---|---|---|
| text baseline (references in the prompt) | **3.592** | 42.9% | 945 | 74.9% |
| **step 1 — frozen LLM, small adapter** | **1.954** | 4.2% | 879 | 69.1% |
| step 2 — bigger adapter | 1.737 | 2.4% | 809 | 61.3% |
| step 4 — oracle + LoRA (a ceiling, it sees the answer) | 2.712 | 16.5% | 944 | 69.5% |
| step 5 — graph + LoRA r16 | 1.490 | 1.1% | 453 | 60.6% |
| step 6 — neighbour tokens + GATv2 + r64 | 1.314 | 0.5% | 342 | 31.5% |
| step 7 — ranking loss | 1.377 | 0.2% | 195 | 32.3% |
| run 8 — whitened + VICReg | 1.519 | 0.2% | 59 | 30.8% |
| gate=0 controls (one string for all 945) | 1.29–1.35 | chance | 1 | 23.9% |

The best graph run is still step 1, the simplest one. Every later change lowered the judge
score, and steps 6–8 also lost the domain.

Four findings decide what run 9 is:

1. **The information was never missing.** With no model at all, averaging a paper's reference
   embeddings and returning the nearest *training-split* abstract gives **27.5% top-1**, or
   **32.0%** in whitened space (`src/analysis/retrieval_baseline.py`; 27.3% with
   near-duplicates removed). That is 6–7× the best graph run, and above the oracle+LoRA
   ceiling's 16.5%.
2. **Reading the vector is the bottleneck, and it was learned from too little data.** Every
   run learned the vector → text mapping from 7,750 papers, on a loss dominated by generic
   scientific wording. Wherever the vector carried little, more capacity bought a template:
   frozen 2.9% → LoRA r16 1.2% → r64 0.3% top-1. Step 4 shows the same channel *can* carry
   identity once the vector is informative.
3. **Domain is already solved; the judge gap is at the problem/task level.** Step 1 matches
   the text arm on arXiv category (69.1% vs 74.9%, against a 72.4% instrument ceiling), yet a
   step-1 output in the right domain scores 2.02 while a text-arm output scores 3.63
   (`src/analysis/domain_probe.py`). The defensible claim is: **domain fully, problem/task
   partially, methodology not**, because a paper's contribution is by definition what its
   references don't contain.
4. **Most run-to-run changes were several at once, and three diagnoses reversed in turn**
   (steps 3 → 4 → 6), each misdirecting the next build. Run 9 moves one idea.

## 2. The plan

Split the task in two, and train only the half that needs the LLM.

```
Skill A (trained here):   paper's own embedding ──► adapter ──► KV prefix ──► Qwen3 + LoRA ──► its abstract
                          90,461 papers, no graph

Skill B (no training):    references ──► mean of their embeddings  ≈  the paper's embedding

Evaluation:               references ──► Skill B ──► Skill A ──► abstract
```

Skill A is step 4's oracle, with two changes:

- **Much more variety.** 90,461 papers seen about once each, instead of 7,750 seen ten times,
  at similar compute.
- **Corrupted input.** At evaluation the decoder is handed an estimate of the embedding, not
  the embedding itself, so it is trained on estimates.

Its training signal cannot be satisfied by a template: the vector always describes the target.

## 3. What changed, and why

| change | where | why |
|---|---|---|
| train on all train-split core papers (90,941 → 90,461 after removing 477 stubs and 3 near-duplicates of eval papers) | `prepare_run9_sets.py` → `data/frozen/training_sets_run9.json` | the 7,750 cap came from the text arm's context budget, which does not apply here. Validation keeps the same 806 nodes |
| `--oracle-noise 0.3`: each training vector rotated to a cosine drawn uniformly from [0.3, 1] with the truth | `model.corrupt_to_cosine`, `train.py` | in whitened space the reference mean sits at cosine 0.31 / 0.49 / 0.62 (p10/50/90) from its target. Without noise that whole range is unseen. Rotation is exact and keeps unit norm. Whitened space is near-isotropic (participation ratio 948/1024), so an isotropic error direction is a fair stand-in |
| `--epoch-size 15000`: an epoch is a slice of one seeded permutation | `train.epoch_nodes` | a full pass is ~5,600 optimizer steps, too long between checkpoints for 48 h. Slicing walks every node before any repeat and is reproducible on resume |
| oracle training loads seeds only, no neighbourhood | `graph.make_loader(hops=0)` | nothing reads them, and some train papers have 2-hop neighbourhoods in the tens of thousands |
| selection = MRR of text generated from the **reference mean** on 128 fixed validation papers; the clean-oracle score is logged beside it | `train.identifiability`, `identify.py`, `selection_score` | LM loss is the number that misled steps 2, 5 and 6. This is the graph arm the thesis reports, measured on generated text, never on eval |
| stop rules in code: exit if the canary writes < 9/16 distinct texts at any epoch end, or if the clean oracle is < 10× chance from epoch 2 | `train.py` | run 8 ran 1,250 steps past a stop rule written only in its plan |
| LoRA targets q/k/v → **q/v** | `config.py`, `lora.py` | matches step 4; `k_proj` was only for the neighbour tokens. `generate.py` now reads the targets off the checkpoint, so steps 6–8 still load |
| `--input {oracle, refs-mean, stage-a, zero}` in generation | `generate.py` | one checkpoint, four arms (§4). `stage-a` loads run 8's encoder **with its head**: the head output is the tensor Stage A decorrelated, and discarding it was run 8's bug |
| scikit-learn added to the container | `adapter-qwen-gpu.def` | `identify.py` uses scikit-learn's `TfidfVectorizer`, so training and the analysis scripts score with the same instrument. **The `.sif` must be rebuilt** |
| `verify_decoder_inputs.py` (new, runs in both jobs) | | 20 synthetic CPU checks, with negative controls. Covers the noise cosine, the reference mean reading exactly the cited papers and never the paper itself, hops=0, the epoch slicing, the stage-a head, empty text ranking last, and LoRA targets read off a checkpoint |

Unchanged from step 4: Qwen3-14B, the kv192 adapter, 32 prefix tokens, the graph-only prompt,
LoRA r16, AdamW 1e-4, batch 1×16, greedy decoding, the 945 eval papers and the judge. Kept from
run 8: whitened features, cleaned targets, gate-init 0.2. Off: ranking loss and neighbour tokens.

**Why branch from run 8 and not step 1.** Step 1 can't be reproduced at HEAD: its adapter
shape, raw features and raw targets changed without flags. More importantly, step 1 is the
number to beat, not a base to build on. HEAD has the step-4 oracle path, the canary and every
checkpoint guard.

## 4. How to run, and how to read it

```
cd src/adapter && python prepare_run9_sets.py         # local, done: 90,461 / 806
sbatch hpc/komondor/jobs/run_train_decoder.sh          # resubmit on timeout
sbatch hpc/komondor/jobs/run_generate_decoder.sh       # RUN_TAG=<first tag> to resume
```

**One job for both (recommended while the queue is busy):** `sbatch
hpc/komondor/jobs/run_decoder_all.sh` trains, then evaluates `best.pt` on all four arms. The
whole run is estimated at ~45 h against a 72 h walltime, and if it does run over, **resubmitting the same file is the
intended way to continue**:
- training resumes from its newest epoch, or exits in seconds once finished;
- evaluation resumes into the same folders, via a tag saved in `$CKPT_DIR/eval_tag` on the
  first start.

The exception is a stop rule. `train.py` exits with code 3 only for those, and the job then
writes `$CKPT_DIR/STOPPED` and refuses every later submission until the file is deliberately
deleted, so a failed run cannot be continued by accident. Each resubmit still reloads the
14B model before deciding (a few minutes).

**Container rebuild** (scikit-learn was added), on a login node, building to `/tmp` because
Lustre cannot host a fakeroot build:

```
module load singularity
cd /project/c_gnn42/csanad_thesis/baseline
singularity build --fakeroot /tmp/adapter-qwen-gpu.sif hpc/komondor/containers/adapter-qwen-gpu.def
cp /tmp/adapter-qwen-gpu.sif adapter-qwen-gpu.sif && rm /tmp/adapter-qwen-gpu.sif
```

Score each arm as before (R1, plus R3 for the domain/problem/task claim), then run
`domain_probe.py` and `probe.py`. Also judge the floor already written to
`experiments/2026-10-07_retrieval-baseline/` (32.0% top-1, no LLM).

| arm | compare to | reading |
|---|---|---|
| zero | gate=0 controls, 1.29–1.35 | the floor |
| **refs-mean** | step 1 (1.954 / 4.2%), retrieval (32.0%) | **the result**: the graph through a trained decoder, no graph training |
| stage-a | refs-mean | is a GNN worth more than averaging? Stage A alone added +2.7–4 points of retrieval |
| oracle | step 4 (2.712 / 16.5%) | did variety teach Qwen to read the vector? A ceiling, always labelled as one |

| outcome | next |
|---|---|
| oracle well above 16.5%, refs-mean above step 1 | fine-tune the reference-mean estimator and decoder jointly at a low learning rate, then per-reference tokens through the same adapter |
| oracle up, refs-mean flat | the estimate falls outside what the decoder learned to read: heavier noise, or train on blends of the true and estimated vectors |
| oracle flat | data was not the bottleneck; move from the KV prefix to input-embedding tokens with matched norms |

In W&B, watch `val_id/oracle_top1` and `val_id/refs_top1` (chance 0.008) at every epoch end,
`canary/distinct`, and `train/tokens_per_sec` in the first hour to size the walltime.

## 5. Limitations and open questions

- **Three things move against step 4** (data, noise, selection), plus run 8's whitening and
  cleaning. They are argued as one hypothesis; a positive result cannot be split among them
  without ablations.
- **Whitening is kept, not proven necessary.** Step 4 worked on raw features. Whitening adds
  +4.5 points to the reference-mean lookup and makes the isotropic noise model sensible, but
  most of its retrieval gain is a better similarity measure, which a learned adapter could
  learn by itself.
- **Isotropic noise is an assumption** about the estimate's error. If refs-mean lags the
  oracle, matching the real error (blending with the reference mean) is the fallback.
- **Stage A picked its epoch on the eval set** (`pretrain_encoder.py`). That is a small leak
  for the stage-a arm only; it should be refit on validation before that arm is reported.
- **Walltime is unmeasured** (~15k × ~500 tokens per epoch).
- **Correction to `soft_prompt_only.md` (steps 6–7):** the text baseline does **not** see the
  target's own title. `text.py` shows only the references' titles and abstracts.
- Methodology stays out of reach by design. Per-reference tokens are the only planned lever on it.

---

## 6. Run 9 result: stopped at epoch 1, the vector unused

Job 14529557, 2026-10-07. The canary stop rule fired at the end of epoch 1 (937 optimizer
steps, 1.91 h): **1 distinct text for 16 validation papers**. The job wrote `STOPPED` and
skipped evaluation, as designed. Throughput was 1,386 tok/s, 2.3× the estimate in §4.

It was not a premature stop. Against step 4 at the same step, from W&B:

| at ~940 steps | step 4 (oracle + LoRA) | run 9 |
|---|---|---|
| `valid/delta` (loss gain from the vector) | **0.022**, rising to 0.104 by step 3,388 | **0.0003**, flat since step 250 |
| `gate/mean_abs` | 0.0012 → 0.016 (×13) | 0.013 → 0.016 (×1.2) |
| validation oracle top-1 (128 papers) | not logged | 0.008, chance |

In step 4 training pushed the gate open; in run 9 it never started, and Qwen learned the
register while ignoring even the clean embedding. Run 9 changed three things about the
adapter's input at once relative to step 4: **whitened features**, **gate-init 0.2** (both
from run 8) and **synthetic noise from step 0**, plus each paper seen once. The run cannot say
which one did it. Whitening is the leading suspect: in raw space most variance sits in a few
topic directions, which even an untrained adapter passes through and which gives the gate an
early reason to open; whitening spreads that evenly over 1,024 dimensions. Run 8, the only
other whitened run, also collapsed.

## 7. Run 10a: step 4's recipe, trained on the real estimate

Implemented 2026-10-08, CPU-verified, not yet run.

**The decision.** Return to the one configuration that opened the channel (step 4), keep the
one change with a clear reason (90k papers instead of 7,750), and replace the synthetic noise
with the input the graph arm actually feeds at evaluation.

| | run 9 | run 10a | why |
|---|---|---|---|
| node features | whitened | **raw** (`--features raw`) | step 4's; whitening is the leading suspect above |
| gate at step 0 | 0.2 | **0** (`--gate-init 0`) | step 4's; its gate opened ×13 in ~1,000 steps |
| training input | own embedding, rotated to cosine U[0.3, 1] | **own embedding, or with probability 0.5 the mean of its references** (`--refs-mix 0.5`) | the real estimation error instead of an assumed isotropic one, which also removes the noise model's dependence on whitened space |
| stop rules | canary from epoch 1 | **both rules from epoch 2** | step 4 needed ~1,000 steps to open; epoch 1 is 937 |
| eval arms | zero, refs-mean, stage-a, oracle | zero, **refs-mean**, oracle | run 8's Stage-A encoder reads whitened features |

Unchanged from run 9: the 90,461 / 806 node sets, 15k-paper epochs, Qwen3-14B, kv192, LoRA r16
on q/v, cleaned targets, selection on validation reference-mean MRR, the identifiability and
canary instruments.

**Code.** `model.mix_in_refs` (a paper with no sampled references keeps its own embedding);
`train.py --refs-mix` (training loader takes one hop when set) and `--features`, both recorded
in the checkpoint and checked on resume; `generate.py --features`, asserted against the
checkpoint, and refused for `--input stage-a` on raw features; `config.FEATURES`,
`config.REFS_MIX`. `verify_decoder_inputs.py` gained four checks: a mixed vector is exactly one
of the two inputs, the mean is drawn about half the time, a reference-less paper never gets a
zero vector, and fraction 0 is the plain oracle. All verify scripts pass.

**Run.** `sbatch hpc/komondor/jobs/run_decoder10a_all.sh`, own checkpoint directory
(`qwen14b-run10a-decoder`), so run 9's `STOPPED` does not block it. Same resubmit semantics as
§4. About 12 h of training (measured) plus ~14 h for the two long evaluation arms; 48 h
walltime. No container rebuild beyond the one run 9 needed.

**Reading it.** In W&B compare `valid/delta` and `gate/mean_abs` against step 4's curve
(0.022 and ×13 at ~940 steps). At each epoch end, `val_id/oracle_top1` says whether Qwen reads
the vector, `val_id/refs_top1` whether that carries over to the graph estimate.

| outcome | meaning | next |
|---|---|---|
| gate opens, refs-mean arm above step 1 (1.954) | the reader works on the graph's estimate | run 10b: one prefix per reference beside the aggregate prefix, from this adapter |
| gate opens, refs-mean flat | reading works, the estimate is too far | more weight on the reference half, or a trained aggregator |
| gate stays shut with step 4's own recipe | the one-pass-per-paper regime is the difference from step 4 | repeat papers (smaller pool, more epochs) |

**Limitations.** A stop at epoch 2 can still be a false positive if this run opens later than
step 4 (half its examples carry the weaker reference signal); deleting `STOPPED` and
resubmitting continues from the saved epoch. Cleaned targets are the one run-8 setting kept.

## 8. Run 11: reader, GNN imitation, joint fine-tune

**Why.** Run 10a's evaluation (945 papers, LLM judge, accuracy 0–5) separated the two halves
of the pipeline:

| arm (run 10a reader) | judge accuracy | ≥3 | coherence | TF-IDF top-1 |
|---|---|---|---|---|
| refs-mean (the graph arm, no training) | 2.82 | 52% | 7.19 | 15.2% |
| oracle (own embedding, a ceiling) | 3.41 | 71% | 6.99 | 32.6% |
| Qwen text baseline (reads the neighbour abstracts) | 3.59 | 80% | 7.90 | 42.9% |

- **The reader is close to good enough.** Given a perfect vector it is within 0.18 of the text
  baseline. The gap between refs-mean and oracle is 0.59 (paired SE 0.05), so most of what is
  left is in the estimate.
- **Score follows the estimate's cosine to the paper's own embedding**, about +0.6 judge per
  +0.1 cosine on the refs-mean arm (quintiles 2.33 → 3.21). The oracle arm stays flat
  (3.30–3.57) on the same papers, so this is not the papers being harder or easier.
- **Loops:** 17.6% of oracle outputs (13.7% of refs-mean) repeat a sentence. They score
  coherence 3.8 against 7.7 for the rest.

Run 11 therefore trains a GNN to produce a better estimate than the plain mean, and trains the
whole pipeline as one standalone run so the thesis can describe it without earlier
checkpoints.

**The pipeline.**

| phase | trains | on | objective | selection |
|---|---|---|---|---|
| 1. reader | prefix adapter + LoRA r16 (q/v), Qwen3-14B frozen | 6 × 15k of the 90,461 run-9 papers | LM loss; input is the own raw embedding, or with p = 0.5 the reference mean (run 10a's recipe, unchanged) | validation refs-mean MRR |
| 2. GNN grid | GNN + linear head, no LLM | the 83,259 training papers with ≥1 reference, 30 epochs, batch 256 | 1 − cosine(head output, own raw embedding), optionally + 0.05 × InfoNCE | validation mean cosine (top-1 breaks ties) |
| 3. joint | GNN + head + adapter + LoRA, from phases 1 and 2 | 3 × 15k papers with references | LM loss + 1 × (1 − cosine to own embedding); GNN lr 1e-5, reader lr 1e-4 | validation MRR of text written from the GNN estimate |

The GNN grid has eight variants: {GATv2, GraphSAGE} × {1, 2 layers} × {cosine, cosine +
InfoNCE}. Each takes minutes on the GPU; one epoch's neighbourhood sampling takes ~6 s even on
a laptop CPU. All eight are reported by cosine and retrieval. Only the winner goes through the
LLM, since an evaluation arm costs ~7 GPU-hours.

**Decisions and their reasons.**

- **Cosine is the main GNN loss** because cosine is what the judge tracks.
- **InfoNCE is weighted by tau = 0.05.** Its gradient on a cosine is scaled by 1/tau, so at
  weight 1 it would take over the objective.
- **VICReg and the 0.40 contrast gate are off.** The targets are real embeddings, which cannot
  collapse. Also, the refs-mean input the reader decodes well sits at output cosine 0.66,
  above the old gate.
- **Papers without references are dropped from GNN training** (8% of the training set). Their
  estimate would be the [MASK] row alone, the same vector for every one of them.
- **Selection is on validation, and the eval set is read once at the end.**
  `pretrain_encoder.py` used to select on eval-set top-1, which read the test set. Run 8's
  Stage-A numbers were selected that way.
- **The GNN keeps the head**, so its output is normalised and in the raw embedding space, as
  the reader expects (`model.EmbeddingEstimator`).
- **The joint stage keeps the cosine term.** Without it the LM loss could pull the estimate
  out of the space the reader was trained on, as the encoder drifted in runs 7–8. The GNN
  moves at a tenth of the reader's rate because it arrives trained.
- **No per-neighbour prefixes yet.** That keeps the joint stage's gain attributable to joint
  training alone.
- **Loop fix:** `generate.py --no-repeat-ngram 6` bans any repeated 6-word sequence. That is
  long enough to leave ordinary phrasing alone and short enough to stop a repeated sentence.
  The job tests it on 200 papers against no fix and keeps it only if it lowers the loop rate
  and costs at most 2 points of top-1. Every run-11 evaluation arm then uses one setting, so
  the arms stay comparable with each other but not with run 10a's numbers.
- **One job, not two.** The queue wait on Komondor can be days. The only reason to split was a
  human check between phases, and both checks are in code: the loop rule above, and the GNN
  winner by validation cosine. The joint stage runs even if no GNN beats the plain mean, which
  costs ~13 h in that case; the GNN table in the log says whether it happened.

**Code.**

- `model.EmbeddingEstimator`: GNN + head, unit-norm output, same call signature as the encoder.
- `graph.with_references`.
- `pretrain_encoder.py`:
  - `--sets run9`, `--features`, `--lambda-cos`, `--lambda-nce`;
  - validation selection;
  - the refs-mean baseline scored on the same papers;
  - the eval set scored once for the chosen epoch.
- `train.py`:
  - `--reader` (phase 3: adapter and LoRA from a reader checkpoint, GNN + head from
    `--pretrained-encoder`), `--lambda-cos`, `--gnn-lr`;
  - identifiability on the GNN and refs-mean arms, with selection on the GNN arm;
  - `joint` recorded in the checkpoint and checked on resume.
- `generate.py`:
  - loads joint checkpoints;
  - `--no-repeat-ngram`, `--subset N`;
  - `--input stage-a` checks the GNN's recorded features instead of requiring whitened ones.
- New scripts: `select_encoder.py` (the GNN table and `winner.pt`) and `loop_check.py` (loop
  rate, top-1 and length per CSV; `--choose 6` makes the loop decision).
- Config: `JOINT_LAMBDA_COS = 1.0`, `JOINT_GNN_LR = 1e-5`, `NO_REPEAT_NGRAM = 0` (off unless
  asked for, so earlier runs reproduce).

**Verified on CPU.** All five verify scripts pass. The full flow ran end to end with a tiny
random Qwen3 of the 14B's attention shape:

- the GNN grid (two variants) and `select_encoder.py`;
- a 1-epoch reader;
- a joint fine-tune from it, then a resume;
- generation from the reader with the GNN, and from the joint checkpoint, with the loop fix.

This checks the mechanics, not the numbers.

**How to run.** `sbatch hpc/komondor/jobs/run11_all.sh` (~50 h estimated, 72 h walltime). In
order, it runs:

1. the GNN grid and `select_encoder.py`;
2. the reader;
3. the loop test, with its choice saved to `checkpoints/run11/ngram`;
4. the reader evaluated from zero, the GNN winner and refs-mean;
5. the joint fine-tune and its evaluation;
6. the oracle ceiling.

Resubmitting continues where the last job stopped. Finished GNN variants are skipped, both
trainings resume, the loop choice is read back, and evaluations append only missing rows. A
reader stop rule writes `checkpoints/run11/reader/STOPPED` and blocks resubmission.

Checkpoints go to `checkpoints/run11/{gnn,reader,joint}`. Experiments go to
`experiments/<tag>_run11-{reader-zero,reader-gnn,reader-refs-mean,joint,reader-oracle}` and
`_run11-looptest-ngram{0,6}`.

**What to check in the log:**

- **The GNN table:** does the winner beat the plain mean's validation cosine?
- **The loop decision line:** loop rate and top-1, with and without the fix.

**Reading it: the thesis table.**

| row | what it shows |
|---|---|
| reader ← refs-mean | the graph with no training |
| reader ← GNN (phases 1 + 2, separate) | what imitation adds over averaging |
| joint (phase 3) | what training the pieces together adds |
| reader ← oracle | the ceiling of a single-vector reader |

**Expected range.** These are estimates from run 10a's cosine–judge relation, not
measurements. Phase 2 GNN: ~+0.1–0.2 over refs-mean, if it gains cosine the way earlier
pretraining gained retrieval. Joint: ~3.1–3.3. Loop fix: up to +0.1 accuracy and +0.5–0.7
coherence on every arm. The single-vector ceiling is ~3.5; differences under ~0.1 are within
noise.

**Limitations and open questions.**

- **The training papers are not like the eval papers.** Training papers have a median of 5
  references, validation and eval papers 45–52. The GNN trains mostly on small
  neighbourhoods. Selection on validation guards against choosing badly, not against the
  shift. 44% of core papers lack a real bibliography (S2 coverage).
- **The GNN may not beat the plain mean.** A plain mean already reaches cosine ~0.82. If no
  variant beats it, the next step is a GNN that predicts a correction to the mean rather than
  the whole vector.
- **The joint weights are untuned.** `JOINT_LAMBDA_COS` and `JOINT_GNN_LR` are reasoned, not
  swept.
- **The reader is retrained from scratch**, so run 11's reader is a second sample of run 10a's
  recipe. The two oracle arms show how much that recipe varies between runs.

## 9. Run 11 result (job 14541449): the loop fix worked, the estimate barely moved

Trained 2026-10-09/10 in one job. Only the joint arm has been judged so far. The other arms
were scored with TF-IDF identifiability (`loop_check.py`) alone.

**GNN grid** (validation cosine to own embedding; plain reference mean = 0.816, top-1 63.5%):

| variant | val cos | val top-1 | eval cos | eval top-1 |
|---|---|---|---|---|
| **GATv2 1-layer cos (winner)** | **0.828** | 70.5% | 0.833 | 70.8% |
| GATv2 2-layer cos | 0.822 | 64.4% | 0.822 | 63.2% |
| GATv2 1-layer cos+NCE | 0.820 | 72.3% | 0.826 | 75.2% |
| SAGE 1-layer cos | 0.820 | 67.2% | 0.825 | 68.4% |
| SAGE 2-layer cos | 0.819 | 66.0% | 0.821 | 65.2% |
| SAGE 1-layer cos+NCE | 0.818 | 72.0% | 0.822 | 72.8% |
| GATv2 2-layer cos+NCE | 0.814 | 71.0% | 0.814 | 69.4% |
| SAGE 2-layer cos+NCE | 0.810 | 70.6% | 0.812 | 71.0% |

All eight variants fall within 0.02 of each other and of the plain mean. Second-hop layers
are worse. InfoNCE trades cosine for top-1. On eval, the winner beats the mean by +0.010
cosine (0.833 against 0.823). By run 10a's slope of ~0.6 judge per 0.1 cosine, that is worth
about +0.06. Architecture is not the limit; what the references say about a paper is.

**Loop test** (200 papers): `--no-repeat-ngram 6` kept. Loop rate 14.0% → 0%, top-1
52.0% → 51.5%, 123 → 107 words.

**Training.**
- **Reader:** the gate opened from the first eval (`valid/delta` 0 → 0.085). Selected at
  epoch 5 on refs-mean MRR, which peaked at 0.530 and was flat at epoch 6. Oracle MRR was
  still rising (0.62 → 0.70), and the learning rate never decayed.
- **Joint:** `train/cos_loss` stayed at 0.20 and `encoder/output_cosine` at 0.54 for all three
  epochs, so at lr 1e-5 the GNN did not move and the stage only adapted the reader to the GNN
  output. Validation GNN MRR rose 0.59 → 0.64 (128 papers, SE ~0.04).

**Eval (945 papers).**

| arm | judge acc | ≥3 | coherence | TF-IDF top-1 | MRR | loops |
|---|---|---|---|---|---|---|
| Qwen text baseline | 3.59 | 80% | 7.90 | 42.9% | 0.557 | 0% |
| 10a oracle | 3.41 | 70% | 6.99 | 32.6% | 0.449 | 17.6% |
| 10a refs-mean | 2.82 | 52% | 7.19 | 15.2% | 0.256 | 13.7% |
| **11 joint** | **2.94** | 60% | **7.86** | 18.8% | 0.301 | 0% |
| 11 reader ← oracle | – | – | – | 27.8% | 0.408 | 0% |
| 11 reader ← GNN | – | – | – | 18.5% | 0.305 | 0% |
| 11 reader ← refs-mean | – | – | – | 16.5% | 0.271 | 0% |
| 11 reader ← zero | – | – | – | 0.1% | 0.008 | 0% |

- **Joint − 10a refs-mean = +0.125 (paired SE 0.039)**, but it comes almost entirely from the
  loop fix. On the 129 papers where 10a looped the gain is +0.61. On the other 816 it is +0.05
  (SE 0.04), which is not significant. Coherence rose by 0.67, as expected.
- **The joint stage added nothing measurable.** Joint and reader ← GNN have the same TF-IDF
  top-1 and MRR (18.8% / 0.301 against 18.5% / 0.305).
- **GNN against plain mean, same reader:** +2.0 points of top-1 and +0.034 MRR. That is
  small, and in line with the cosine gain.
- **This reader's oracle arm is weaker than 10a's** (27.8% against 32.6% top-1). It is not
  the n-gram ban, which cost 0.5 points in the loop test. Retraining the same recipe varies by
  this much, and the reader was selected for its refs-mean arm.
- **Where the gap to the oracle sits:** on the fifth of papers whose references are least
  like them (refs-mean cosine < 0.79), the joint arm scores 2.52 against the oracle's 3.34. On
  the most similar fifth it scores 3.32 against 3.57. The estimate fails where the references
  do not describe the paper.

**Against the expected range:** the loop fix was as predicted for coherence, and larger for
accuracy because looped outputs scored low. The GNN gain was below the +0.1–0.2 guess, and
the joint stage came in under 3.1–3.3.
