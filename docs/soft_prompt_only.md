# Soft-prompt-only — the graph prefix as the only channel

Branch `feature/soft-prompt-only`. Append a section per step.

---

## Step 1 — removing the references from the prompt

**Set up 2026-09-16. Run pending.**

### Why

The Qwen3-14B runs of 2026-09-10 (945 capped eval nodes, era-1 prompt, ~13 reference abstracts
in the prompt) reversed the Vicuna result:

| | `llm_accuracy` |
|---|---|
| frozen Qwen3-14B, abstracts in prompt | 3.592 |
| + GraphSAGE prefix adapter | 3.520 |
| **gap** | **−0.072 ± 0.101** (95% CI, Wilcoxon p = 0.07) |

Against +0.473 under Vicuna-7B. The gate opened during training — further than it did under
Vicuna — so the prefix was used; it just did not help once a stronger model could read the
abstracts itself. The question therefore changes from *does the graph add to the text?* to
*can the graph replace the text?* The prompt now carries no references, and the prefix is the
only thing that tells the model which paper it is writing.

### Decisions

| decision | reasoning |
|---|---|
| system message + a fixed user turn `Write the abstract.` | ending on a bare system message is off-distribution for a chat-tuned model |
| system message = era-4 instruction, re-pointed at "the context you have been given" | era 4 is the best of the four under Qwen3-14B (48 nodes: era 1 3.875, era 2 4.438, era 3 5.292, era 4 5.333), the same ordering as GPT |
| compare against the **existing** 09-10 text-path baseline (3.592), no new baseline | a no-prefix run of this prompt has no information about the paper and writes one identical abstract for all 945 nodes under greedy decoding — a floor, not a control |
| architecture and node sets held fixed (2-layer GraphSAGE, T = 32, trunk 768, code 96; 7,750 / 806 / 945 nodes) | so this run measures removing the text and nothing else. A richer aggregator/adapter is the planned second run |
| schedule unchanged (batch 1 × 16, 48 h, 10 epochs, patience 2) | fewest moving parts; the run is simply cheaper |
| text path replaced in `train.py` / `generate.py`, not flagged | the text runs are reproduced from commit `45dd942`, the way the Vicuna runs are |
| text-only helpers left in place, unused | `neighbour_cap.py`, `freeze_prompt_refs.py`, `prepare_data_capped.py`, `verify_capped_prompts.py` and the capped branch of `text.py` still run, so the historical freeze stays re-derivable without a checkout |

### What changed

- `text.py` — `GRAPH_ONLY_SYSTEM`, `GRAPH_ONLY_USER`, `build_graph_only_prompt`, and
  `graph_only_example(tokenizer, corpus, node_id)`, the one builder both scripts call.
  `PROMPT = "graph-only"`.
- `train.py`, `generate.py` — use `graph_only_example`. `--sets` now selects nodes only. Every
  checkpoint records `prompt`; resume and generation refuse a checkpoint without it. This guard
  is the one nothing else provides: a text-path Qwen checkpoint has exactly the same shapes and
  would otherwise load cleanly.
- `generate.py` — the `--max-new-tokens` reserve guard and the capped overflow assertion are
  gone; both protected the frozen reference budget, which no longer exists.
- `verify_graph_only_prompt.py` (new) — replaces `verify_capped_prompts.py` in both job scripts.
- Job scripts — `CKPT_DIR=checkpoints/qwen14b-graphonly`, run/arm name
  `adapter-qwen14b-graphonly-gnn2`, `RUN_BASELINE=0`.

### How to run

```
sbatch hpc/komondor/jobs/run_train_adapter.sh     # resubmit on timeout; resumes by itself
sbatch hpc/komondor/jobs/run_generate_eval.sh     # after training; adapter arm only
```

Then score as before and compare to `experiments/2026-09-10-14364743_qwen14b-baseline`.

### Verified (locally, `DyTAG-qwen` env, transformers 4.51.3)

| check | result |
|---|---|
| `verify_graph_only_prompt.py` | exact ChatML string matches; **241 prompt tokens**, identical for every node; all 7,750 train / 806 valid / 945 eval nodes have a non-empty target and labels masking exactly the prompt; longest sequence **1,104 tokens** incl. prefix (was 3,425 prompt tokens alone) |
| `verify_sliced_loss.py` | unchanged, max error 4.8e-07 |
| `verify_generation.py` | unchanged, max logit error 2.7e-07, prefix survives `generate()` |
| prompt guard | `generate.load_adapter` on a checkpoint with no `prompt` key raises before loading weights |

The local `DyTAG` env (transformers 4.39.3) cannot read Qwen3's tokenizer; use `DyTAG-qwen`.

### Measured for the next step — what a plain centroid already knows

For each of the 945 eval nodes, the mean Voyage vector of its references, ranked against the
945 target embeddings by cosine (scratch probe, not in the repo):

| input | correct paper top-1 | top-10 |
|---|---|---|
| 1-hop mean, 1,024-d | 64.3% | 96.0% |
| 1-hop mean, first 512 dims | 64.6% | 95.3% |
| 1-hop mean, first 256 dims | 59.0% | 93.0% |
| uniform mean over 1- and 2-hop nodes | 33.2% | 76.4% |
| best single reference (oracle, uses the target) | 84.1% | 99.5% |

Read as: embedding width is not the bottleneck (512 = 1,024); how neighbours are combined is.
And structurally, the adapter's shared linear decoder puts every layer's graph delta in the same
96-dimensional subspace — widening the trunk would not lift that. Both point the planned second
run at a non-pooling aggregator (learned queries attending over neighbour states, one per prefix
token) rather than a larger embedding model.

### Known limitations

- **The comparison differs in the wording as well as the text.** The 3.592 baseline ran under
  era 1; this run's system message is era 4, which scored 1.46 higher than era 1 on the 48-node
  Qwen probe. A gain cannot be attributed to the graph alone. Not regenerated, by decision.
- **No per-run floor.** `valid/gate0` is now "base prefix, no graph, no text" — the closest thing
  to a floor, logged every validation pass.
- **One 1,024-d vector carries the whole neighbourhood.** Names, numbers and specific methods of
  the references cannot survive a mean; expect topic-level anchoring at best.
- **Epoch cost is unmeasured again.** Sequences average ~460 tokens (241 prompt + ~219 target)
  against ~3,075 before, so
  the ~10-11 h/epoch estimate is now far too high; the first epoch measures it. Faster epochs
  also mean early stopping fires sooner in wall-clock time.

---

## Step 1 — result

**Run 2026-09-16, job 14393649.** `experiments/2026-09-16-14393649_adapter-qwen14b-graphonly-gnn2/`,
945 rows, 0 errors, 879 distinct generations.

| | `llm_accuracy` | `llm_coherence` | `llm_language` |
|---|---|---|---|
| graph-only adapter | **1.954** | 7.526 | 7.566 |
| text baseline (09-10) | 3.592 | 7.899 | 8.056 |

Paired against the text baseline: **−1.637 ± 0.097** (Wilcoxon p = 2e-115); against the text
adapter: −1.565 ± 0.108. Accuracy distribution: 321 papers score 1, 441 score 2, 25 score 5 or more.

The prefix alone does not replace ~13 reference abstracts: the text stays fluent and coherent
(coherence 7.5) but the judge rarely recognises it as the right paper. The generations have not
been read yet. The per-node outputs are not collapsed — 879 of 945
are distinct, so the prefix does steer generation — but not toward the right content. Reading which
topics survive is open.

The wording confound noted in step 1 works *against* this result, not for it: era 4 outscored era 1
under Qwen, so the drop is not a prompt artefact.

---

## Step 2 — a larger adapter with separate key and value decoders

**Set up 2026-09-17. Run pending.** Architecture only; prompt, node sets, encoder and schedule are
unchanged from step 1.

### Why

The run showed no overfitting (user's reading of the W&B curves), so capacity is a candidate.
The old adapter had one structural limit that more parameters in the wrong place would not lift:
a single decoder, shared by all 80 (layer, role) pairs, so every key and every value prefix was
drawn from the same 96-dimensional subspace. Keys and values do different jobs in different spaces
— a real Qwen3 key is RMS-normalised per head (`k_norm`) and rotated by RoPE before it is cached,
and it decides *where* attention goes; a value carries *what* is written back. Forcing both
through one linear map is the least defensible sharing in the design.

### Decisions

| | was | now | reasoning |
|---|---|---|---|
| decoders | 1, shared by keys and values | **2** (`decoders[0]` keys, `decoders[1]` values), each shared across the 40 layers of its role | the key/value split above; fully per-layer decoders would be ~250M parameters |
| `CODE_DIM` | 96 | **192** | doubles the directions each (layer, role) can use |
| `TRUNK_DIM` | 768 | **1024** | removes the only point narrower than the encoder output. Measured cost of 768 was small — PCA of the 7,750 train nodes' reference centroids: 768 components keep 99.83% of variance (512: 99.07%, 256: 95.74%) — so this closes a question rather than fixing a measured loss |

### Parameters

| part | old | new |
|---|---|---|
| trunk Linear + LayerNorm | 788,736 | 1,051,648 |
| codes Linear (→ 40 × 2 × code) | 5,905,920 | 15,744,000 |
| decoder(s) (→ 32 × 8 × 128) | 3,178,496 | 12,648,448 |
| `base` + `gate` | 2,621,520 | 2,621,520 |
| **adapter** | **12,494,672** | **32,065,616** |
| **+ encoder (4,197,376)** | **16,692,048** | **36,262,992** (2.17×) |

Optimizer state and gradients add ~580 MB of GPU memory against 29.6 GB of frozen Qwen; epoch
checkpoints grow to ~435 MB. The prefix itself is unchanged at 2,621,440 values per node. At
`gate = 0` the pathway behind the gate — 33.6M parameters — still gets zero gradient; watch
`gate/mean_abs` in the first ~100 steps.

### What changed

- `config.py` — `TRUNK_DIM = 1024`, `CODE_DIM = 192`.
- `model.py` — `self.decoder` → `self.decoders` (a 2-element `ModuleList`); `forward` decodes each
  role with its own decoder and stacks on the role axis. New smoke-test check: perturbing one
  decoder moves only its own role.
- Job scripts — `CKPT_DIR=checkpoints/qwen14b-graphonly-kv192`, run/arm name
  `adapter-qwen14b-graphonly-kv192-gnn2`.

Old checkpoints do not load into the new adapter (no `decoders.*` keys, different shapes) — loudly.
Step 1's run is reproduced from commit `b34ab0e`.

### Verified (locally, `DyTAG-qwen`)

| check | result |
|---|---|
| `model.py` smoke test | adapter 32,065,616 parameters; prefix `(8, 40, 2, 8, 32, 128)`; mask, gate=0 and gradient checks pass; key decoder moves only keys, value decoder only values |
| negative control for the new check | stacking on the wrong axis makes the value decoder move keys too — the check catches it |
| `verify_graph_only_prompt.py`, `verify_sliced_loss.py`, `verify_generation.py` | all exit 0, unchanged results |

The new check's first version was itself wrong: restoring the bias with `add_(1)` then `sub_(1)`
does not round-trip in float, so it reported keys moving. Restored from a saved copy instead.

### Known limitations

- **Step 1's result suggests capacity may not be the main constraint.** The prefix already produces
  879 distinct outputs; the failure is content, not variety. A larger adapter is a reasonable
  test, not an expected fix.
- Both decoders are still shared across all 40 layers.
- Walltime estimates in the job scripts are unchanged and still unmeasured for graph-only sequences.

---

## Step 2 — result, and what the failure actually is

**Run 2026-09-17, job 14395431.** `experiments/2026-09-17-14395431_adapter-qwen14b-graphonly-kv192-gnn2/`,
945 rows, 0 errors, 809 distinct generations. Training loss improved over step 1 (W&B); the
evaluation did not.

| | `llm_accuracy` | `llm_coherence` | `llm_language` | rouge1 | bertscore f1 |
|---|---|---|---|---|---|
| step 2, kv192 adapter | **1.737** | 7.423 | 7.544 | 0.288 | 0.834 |
| step 1, original adapter | 1.954 | 7.526 | 7.566 | 0.295 | 0.837 |
| text adapter (09-10) | 3.520 | 7.873 | 7.923 | 0.341 | 0.845 |
| text baseline (09-10) | 3.592 | 7.900 | 8.056 | 0.336 | 0.846 |

Paired over the same 945 nodes: **−0.218 ± 0.065** against step 1 (Wilcoxon p = 2.8e-10),
−1.855 ± 0.092 against the text baseline. Improved on 178 nodes, worsened on 305, tied on 462.

So this is not a second collapse. The collapse happened at step 1; a 2.17x larger adapter moved
it a further fifth of a point *down*. The prediction recorded in step 2's limitations — that
capacity was not the constraint — held, and the sign is informative.

### The larger adapter made the output more generic, not more specific

| | step 1 | step 2 | real abstracts |
|---|---|---|---|
| distinct word types across all 945 outputs | 4,817 | **4,194** | — |
| type/token ratio | 0.0429 | **0.0377** | — |
| rows sharing the single most common 8-word opening | 60 | **232** | — |
| rows whose opening is one of the top 2 | 108 | **423** | — |
| mean pairwise TF-IDF similarity between the 945 outputs | 0.0629 | **0.0670** | 0.0375 |
| accuracy ≥ 5 | 25 | **2** | — |

The 65 nodes that scored ≥ 4 in step 1 average **2.20** in step 2 (from 4.46). The extra capacity
was spent on the part of the loss that is shared across all papers: "We propose a novel approach
to the problem of …". Lower training loss and lower accuracy are consistent, not contradictory —
cross-entropy over a ~219-token abstract is dominated by generic scientific English, and the
tokens that identify *which* paper it is are a small minority of it. More parameters buy more of
the majority.

### The graph's identifying information never reaches the output

Two probes, both on the 945 eval nodes, neither using the judge.

**How identifiable is the generated text?** Rank each generation's own gold abstract among all
945 golds by TF-IDF cosine (chance top-1 = 0.1%):

| run | top-1 | top-10 | median rank |
|---|---|---|---|
| step 2, kv192 | 2.9% | 14.2% | 142 |
| step 1 | 4.3% | 21.6% | 71 |
| text adapter | 36.5% | 73.7% | 3 |
| text baseline | 43.5% | 82.1% | 2 |

**Does a better graph signal produce a better abstract?** The plain reference centroid retrieves
the correct paper for 64.3% of these nodes. Split the nodes on that:

| | n | step 2 accuracy | step 1 accuracy |
|---|---|---|---|
| centroid retrieves the right paper | 608 | 1.743 | 1.949 |
| centroid does not | 337 | 1.724 | 1.964 |

**Flat.** The identifying content of the graph vector — the 64.3% — has no measurable effect on
the generated abstract. What does get through is coarse and topical: accuracy rises weakly with
reference count (1.53 at ≤ 28 references to 1.86 at > 75), and read side by side the outputs do
land in the right field. Node 161266's gold is RL graph-to-sequence question generation and the
model writes seq2seq question generation on SQuAD; node 1121's gold is formal languages for
ecological modelling and the model writes model checking for concurrent programs. Right
neighbourhood, wrong paper, and usually a more prototypical and older-sounding paper than the
real one.

That is the finding: **the prefix carries topic and not identity**, and step 2 made it carry
topic more fluently.

### The implementation was audited and is not at fault

| check | result |
|---|---|
| `python model.py` | 32,065,616 parameters, prefix `(8, 40, 2, 8, 32, 128)`, mask / gate=0 / gate=1 / gradient checks pass |
| role separation + its negative control | pass; the key decoder moves only keys |
| `stack(dim=2)` → `view` → `transpose(3,4)` | `stack` returns a contiguous tensor, so the view is well defined and the layout still matches `base` |
| delta magnitude relative to `base` | 16.90 (step 1 config) → 16.79 (step 2 config) — the trunk's LayerNorm makes the scale independent of both widths, so there is no init-scale regression |
| optimizer coverage | `list(adapter.parameters())` includes the `ModuleList`; every parameter got a gradient |
| checkpoint loading | `load_state_dict` is strict in both `train.py` and `generate.py`; a stale checkpoint fails loudly |

The step-2 diff is four hunks and none of them is wrong. The result is real.

### Known limitations of this analysis

- **No gate=0 arm for this run.** `RUN_GATE0=0`, so what the base prefix writes on its own at
  eval time is still unmeasured. Given the flat centroid table it is likely close to 1.7, but
  that is the control the claim deserves and it has not been run.
- **Two runs, one bundle.** Step 2 changed three things at once (one decoder → two, code 96 →
  192, trunk 768 → 1024). The −0.218 cannot be attributed to any one of them, and is not worth
  another run to separate.
- **TF-IDF retrieval is a lexical proxy.** It understates a paraphrase. It is used only to
  compare runs against each other, where that bias is shared.

---

## Step 3 — the oracle ceiling run

**Set up 2026-09-21. Run pending.** Branch `feature/oracle-prefix`.

### Why

Step 2 established that the prefix carries topic and not identity, and that the graph's
identifying content has no measurable effect on the output (1.743 against 1.724, split on
whether the reference centroid retrieves the right paper). That leaves two candidate causes,
which none of the runs so far can tell apart:

1. **The graph vector is the problem** — a mean over references does not describe the citing
   paper well enough, and a better aggregator, encoder objective or embedding model would fix it.
2. **The prefix → LLM channel is the problem** — a frozen Qwen3-14B cannot be steered into
   writing a specific paper by a KV prefix decoded from one dense vector, whatever that vector
   contains.

The oracle separates them in one run. The encoder is bypassed and the adapter is fed the
**target abstract's own voyage embedding** — a perfect 1,024-d description of the paper it is
being asked to write. Everything downstream is unchanged.

- If the oracle writes recognisable abstracts, the channel works and cause 1 is the one to fix.
  The contrastive head, a non-pooling aggregator and a better embedding model all become
  worth their cost.
- If the oracle still writes generic abstracts, cause 2 holds and none of those three can help,
  because they all improve the vector that the oracle already supplies perfectly.

**Its score is a ceiling, never a result.** This arm sees the answer. It is not comparable to
any arm that does not, and must not be reported beside one without that label.

### Decisions

| decision | reasoning |
|---|---|
| oracle vector = the seed node's own feature | that feature *is* the voyage embedding of the target abstract, which is why `GraphSAGEEncoder` masks it. The oracle skips the mask and the encoder both |
| kv192 adapter kept, not reverted to step 1's | user's decision; keeps the ceiling directly comparable to step 2 (1.737), the most recent arm |
| `--epochs 2`, not 10 | the ceiling only has to show whether the channel can carry identity at all. Graph-only sequences average ~460 tokens, so two epochs are cheap |
| encoder excluded from the optimizer in oracle mode | it is bypassed, so its parameters would sit in AdamW collecting no gradient and inflate the reported trainable count |
| prompt, node sets, decoding and schedule unchanged | one variable moves |
| no gate=0 arm | still `RUN_GATE0=0`. It remains the missing control from step 2 and is not added here |

### What changed

- `model.py` — `oracle_vector(batch)`, the one function both scripts call, returning the seed
  rows of `batch.x`. Smoke test asserts it equals the seeds' stored embeddings, is not the
  encoder output, and has the right shape.
- `train.py` — `--oracle` and `--epochs`. `step_loss` and `validate` take the flag; the
  checkpoint records `oracle`; resume refuses a mismatch. `--epochs` replaces `config.EPOCHS`
  in the loop, the exhaustion check and the W&B config.
- `generate.py` — `--oracle`, threaded into `iter_prefixes`; `load_adapter` asserts the flag
  matches the checkpoint. Both scripts print a banner naming the arm as a ceiling.
- Job scripts — `CKPT_DIR=checkpoints/qwen14b-oracle-kv192`, run/arm name
  `oracle-qwen14b-kv192`, `--oracle --epochs 2`.

An oracle checkpoint has exactly the shapes of a graph one, so the recorded `oracle` flag is
the only thing preventing the two from continuing or scoring each other. It is asserted on
resume and on load, like `prompt` and `gnn_layers` before it.

### How to run

```
sbatch hpc/komondor/jobs/run_train_adapter.sh     # --oracle --epochs 2
sbatch hpc/komondor/jobs/run_generate_eval.sh     # --oracle, adapter arm only
```

### Verified (locally, `DyTAG-qwen`)

| check | result |
|---|---|
| `model.py` smoke test | oracle shape `(8, 1024)`, equals the seeds' stored embeddings, cosine 0.020 to the (untrained) encoder output; all previous checks still pass |
| `verify_graph_only_prompt.py`, `verify_sliced_loss.py`, `verify_generation.py` | exit 0, unchanged |
| both job scripts | `bash -n` clean |
| CLI | `--oracle` and `--epochs` present in both scripts' `--help` |

### Known limitations

- **The oracle is an upper bound on the channel, not on the method.** It says what this
  prefix shape can carry at best; a different injection point (input embeddings, cross
  attention) is not covered.
- **Two epochs is short.** If the oracle scores poorly, "not enough training" is a live
  alternative explanation, and the validation curve is what has to rule it out.
- **Still no gate=0 arm**, so the floor of this prompt with a node-independent prefix remains
  unmeasured.

---

## Step 3 — result: the channel, not the encoder

**Run 2026-09-21, job 14411691.** `experiments/2026-09-21-14411691_oracle-qwen14b-kv192/`,
945 rows, 5 judge errors, **198 distinct generations**.

Scored with the same TF-IDF identifiability probe step 2 used — rank each generation's own gold
among all 945 golds, chance top-1 = 0.11%:

| arm | top-1 | top-10 | median rank | `llm_accuracy` | distinct | word types |
|---|---|---|---|---|---|---|
| text baseline (09-10, era 1) | 40.5% | 80.3% | 2 | 3.592 | 945 | 10,701 |
| graph-only step 1 | 3.9% | 21.6% | 71 | 1.954 | 879 | 4,817 |
| graph-only step 2 (kv192) | 2.6% | 13.7% | 141 | 1.737 | 809 | 4,194 |
| **oracle** | **0.3%** | **2.0%** | **405** | **1.249** | **198** | **884** |

(The probe reproduces step 2's published figures to within 0.4 points, so it is the same
instrument.)

**The oracle is at chance.** Handed a perfect 1,024-d embedding of the very abstract it is asked
to write, the model produces text no more identifiable than a random draw — median rank 405
against a chance median of 473. 736 of 945 rows open with one of two phrases. One identical
600-token abstract about image classification was emitted 75 times, including for a
self-supervised representation-learning paper (node 2629) and an Arabic NLP paper (node 5182).

Read against the two candidate causes step 3 was set up to separate, this is cause 2: the
prefix → LLM channel is the bottleneck. A better aggregator, encoder objective or embedding
model all improve the vector the oracle already supplies perfectly, so none of them can help.

**Specifying one of 945 papers takes about 10 bits.** The oracle supplies a 1,024-d float
vector. Whatever fails here, it is not the channel's width — which is also why "the pooled
vector loses information" does not survive as an explanation for steps 1 and 2.

### Two caveats, one of them serious

- **The oracle ran 2 epochs; steps 1 and 2 ran 10 with early stopping.** A collapsed, generic
  output is also what an undertrained prefix looks like, and one short run cannot tell the two
  apart. This is what step 4 exists to close, and until it does the conclusion above is
  *strongly indicated, not established*.
- **The 5 judge errors** (`ERROR: max retries exceeded`) are excluded from the mean rather than
  imputed; at 5/945 the choice cannot move the result.

### A correction to step 1's headline gap

Step 1 reports graph-only at −1.637 against the 3.592 text baseline and notes a wording
confound without quantifying it. It is now quantified, and it is large. The 945-node baseline is
**era 1** — verified here, not assumed: its generations are byte-identical to
`experiments/prompt_sensitivity_qwen/01_era1/` on 48/48 shared nodes, and 0/48 for eras 2, 3
and 4. The graph-only runs use the **era-4** system message. On the 48 nodes where all arms
exist:

| same 48 nodes | `llm_accuracy` |
|---|---|
| gpt-5.6-luna, text, era 4 | 5.812 |
| Qwen3-14B, text, era 4 | 5.333 |
| Qwen3-14B, text, era 1 (= the 945 baseline) | 3.875 |
| graph-only step 1 | 1.896 |
| graph-only step 2 | 1.833 |
| oracle | 1.438 |

Like for like at era 4, the graph-only gap is about **−3.44**, not −1.637. The confound runs
against the result, as step 1 said — but by twice as much as the published number implies. An
era-4 text baseline over all 945 nodes does not exist and is owed.

It is also worth recording what the same table says about the model gap: gpt-5.6-luna leads
Qwen3-14B by **0.48** on identical references and instructions. Era 1 → era 4 is worth +1.46 to
Qwen. The prompt, not the model, is most of that distance. (These 48 were stratified on era-2/3
score movement, so absolute means are not estimates of the 945; the paired comparisons are.)

---

## Step 4 — the oracle with LoRA, at full epoch budget

**Set up 2026-09-22. Run pending.** Branch `feature/oracle-prefix`.

### Why

Step 3's result has two holes, and one run closes both.

**Every arm of this project has held the LLM frozen.** `lora.py` has been written, documented
and verified by `verify_lora_prefix.py` since Stage 6d, and `LORA=0` in every job script that
has ever run. So step 3's null says "a *frozen* Qwen3-14B cannot be steered by this prefix",
which is weaker than the claim the project needs in order to abandon soft prompts. LoRA is the
only untested fix for the cause step 3 diagnosed.

**The oracle is where LoRA can be read cleanly.** The standing worry about LoRA — recorded in
`project_overview.md` — is that it may learn the task faster than the gate learns the graph, and
so absorb the effect being measured. That worry does not apply here: **LoRA has no per-paper
information.** It is 80 modules of weights shared across all 7,750 training papers. If the
outputs become paper-specific, the identity can only have arrived through the prefix. Nothing
else in the run knows which paper it is.

**And 10 epochs, not 2**, so that "undertrained" stops being an alternative explanation.

The decision rule is set before the run, not after it:

- **Identifiability rises well above chance** → the channel works once the LLM can adapt to it.
  Steps 1–3 then measured a frozen model, not a graph, and the soft-prompt track reopens — at
  which point a non-pooling aggregator (learned queries over neighbour states, one or two
  prefix tokens per neighbour) becomes the right next build, since pooling ~599 2-hop papers
  into one vector is a real loss once the channel can carry more than topic.
- **Identifiability stays at chance** → the KV-prefix-from-a-pooled-vector shape cannot carry
  identity, with a matched-budget oracle ceiling to prove it was never the encoder's fault.
  That is a complete negative result and the justification for changing task.

### Decisions

| decision | reasoning |
|---|---|
| `--lora` on, r=16, q_proj/v_proj | unchanged from `config.py`; the prefix arrives as extra keys and values, so those two projections decide how much of it is read (`lora.py`) |
| `--epochs` dropped, so `config.EPOCHS = 10` | matches steps 1 and 2. Passing `--epochs 10` would restate a constant, which this job script deliberately does not do |
| its own `CKPT_DIR`, `qwen14b-oracle-kv192-lora` | `train.py` refuses to resume across a change of `--lora`, but `best.pt` is a fixed name — sharing step 3's directory would abort at best and overwrite the comparison arm at worst |
| **`RUN_GATE0=1`** | the missing control of this entire stage. This run differs from the 09-10 baseline by two things at once (LoRA and the oracle vector); gate=0 keeps LoRA and disconnects only the vector, so adapter − gate0 is the oracle's contribution alone |
| judged by identifiability first, `llm_accuracy` second | the judge cannot distinguish "generic but fluent" from "specific but wrong" as sharply as the retrieval probe does, and step 3 is the evidence |
| architecture, prompt, node sets, decoding unchanged | one variable moves — two, counting the epoch budget, and that one only restores a matched setting |

### What changed

Job scripts only. `--lora`, `--oracle` and the epoch budget already compose in `train.py`, and
`generate.py` injects LoRA from the checkpoint automatically with no flag to keep in sync.

- `run_train_adapter.sh` — `LORA=1`, `CKPT_DIR=…-lora`, `--epochs 2` removed, W&B run renamed.
- `run_generate_eval.sh` — `CKPT_DIR`/`ARM` follow, `RUN_GATE0=1`, and `--oracle` added to the
  gate0 invocation (`load_adapter` asserts the recorded flag for *every* non-baseline mode, so
  omitting it there would abort the arm).

### How to run

```
sbatch hpc/komondor/jobs/run_train_adapter.sh     # resubmit on timeout; resumes by itself
sbatch hpc/komondor/jobs/run_generate_eval.sh     # after training; adapter + gate0 arms
```

### What to watch in the first ~100 steps

`gate/mean_abs` and `prefix/graph_fraction`. At `gate = 0` the pathway behind the gate gets zero
gradient and has to bootstrap; if LoRA drives the loss down while the gate stays shut, the run
will end with a good validation loss and a disconnected prefix, which reads as a null but is
not one. That is a distinguishable failure and it is already logged.

### Known limitations

- **Checkpoints grow to ~560 MB** (LoRA tensors plus their AdamW moments). Ten epochs is ~5.6 GB
  if every epoch improves.
- **Walltime is untested at this budget.** Step 3 ran 2 epochs; the 10-11 h/epoch estimate in
  the job script predates graph-only sequences and is far too high, but has still never been
  measured. Expect to resubmit; the resume path handles it.
- **A positive result here is a ceiling, not a method.** It would say the channel can carry
  identity when handed a perfect vector — not that a GNN can produce one.
- **Still no era-4 945-node text baseline**, so the absolute comparison for any graph arm
  remains against a prompt known to be 1.46 points weak.

## Step 4 — result: LoRA opens the channel

`experiments/2026-09-22-14422070_oracle-qwen14b-kv192-lora/`, 945 nodes, greedy, era-4
graph-only prompt. Identifiability is the TF-IDF cosine rank of a generation's own gold among
all 945 golds; chance is 0.11% top-1, median rank 473.

| arm | LLM | epochs | top-1 | med rank | distinct | judge acc |
|---|---|---|---|---|---|---|
| oracle (step 3) | frozen | 2 | 0.4% | 404 | 198 | 1.25 |
| **oracle + LoRA (step 4)** | **LoRA** | **10** | **12.0%** | **17** | **944** | **2.71** |
| graph-only kv192 (step 2) | frozen | 10 | 1.7% | 152 | 809 | 1.74 |
| text baseline (era 1) | frozen | — | 39.4% | 2 | 945 | 3.59 |

**Step 3's conclusion was wrong.** It read the oracle's collapse as proof that the prefix→LLM
channel could not carry paper identity, and concluded that no better graph, aggregator or
embedding could help. Channel width never changed between the two oracle runs — same 32 tokens,
same 40 layers, same single 1024-d vector in — and top-1 identifiability moved from 3.6x chance
to 109x chance. The prefix was always carrying identity. A *frozen* Qwen could not read it.

That re-partitions the remaining gap into two measurable pieces:

- **1.7% → 12%** (graph → oracle) is the encoder's: pooling, aggregation, how much of the
  neighbourhood reaches the vector.
- **12% → 39%** (oracle → text) is the channel's: adapter capacity, prefix width, LoRA rank.

Neither is yet sized honestly, because every graph arm in steps 1-2 was measured under the
frozen LLM now known to suppress identifiability ~7x on the same prefix.

### Known limitations

- **The `gate=0` arm did not run.** No `-gate0` experiment directory exists, so LoRA's own
  contribution is still not separated from the oracle vector's. Part of 1.25 → 2.71 may be
  LoRA alone writing better generic abstracts.
- **Generations average 121 words against the golds' 174**, so ROUGE is partly penalising
  length rather than content.
- **The text comparison is cross-era.** The baseline is era 1; the oracle uses era 4. An era-4
  text baseline would be higher still (+1.46 on the 48-node stratified sample), making the
  oracle's deficit larger, not smaller.
- **This is a ceiling, not a method.** It says the channel can carry identity when handed a
  perfect vector, not that a GNN can produce one.

## Step 5 — the graph arm, rerun with LoRA

### Why

Step 4 established that a frozen Qwen3-14B cannot read this prefix: the same 32-token,
40-layer channel carrying the same single 1024-d vector went from 0.4% to 12.0% top-1
identifiability when LoRA was added. Every graph arm in steps 1 and 2 was measured under that
frozen model. Their nulls are therefore measurements of a crippled apparatus, not of the graph,
and the honest headline number for this whole stage does not yet exist.

### Decisions

- **Exactly one change against step 2's graph run** (`checkpoints/qwen14b-graphonly-kv192`):
  `--lora`. Same 2-layer encoder, same kv192 adapter, same graph-only prompt, same 10 epochs,
  same node sets. Step 4 changed two things at once and consequently cannot split its
  1.25 → 2.71 between LoRA and the oracle vector; this run is built not to repeat that.
- **`gate=0` runs this time.** It keeps the LoRA weights and disconnects only the graph, so
  `adapter − gate0` is the graph's contribution and nothing else. Steps 1-4 all shipped
  without it.
- **`gate=0` is a floor, not a paired control.** `text.build_graph_only_prompt` takes no node
  argument, so at gate=0 both prompt and prefix are node-independent and greedy decoding emits
  one identical abstract for all 945 nodes. The scores still vary, because each row is scored
  against a different gold — the 945-row mean is what LoRA plus a generic prefix earns knowing
  nothing about the paper. Per-node comparisons against it would be measuring the golds.
- **Branched from the oracle work, not from before it.** The `--oracle` additions to
  `train.py`/`generate.py` are inert when the flag is not passed, and keeping them means the
  HPC copy of `src/adapter/` does not change at all between step 4 and step 5.

### What changed

`hpc/komondor/jobs/run_train_adapter.sh` and `hpc/komondor/jobs/run_generate_eval.sh` only:
`--oracle` dropped from all three invocations, `CKPT_DIR` → `checkpoints/qwen14b-graphonly-kv192-lora`,
`ARM` → `graphonly-qwen14b-kv192-lora`, W&B run renamed. No Python changed.

### How to read the result

Against step 4's oracle (12.0% top-1, judge 2.71), which is the ceiling for this channel, and
step 2's frozen graph arm (1.7%, 1.74), which is the same encoder on the broken apparatus.

- **Lands near 12%** — the encoder is already saturating the channel. Mean-pooled GraphSAGE
  extracts what a perfect vector would, and GATv2 or per-neighbour tokens have no headroom to
  recover. Spend the effort on the channel instead.
- **Lands near 4-6%** — a real encoder gap exists between pooled neighbours and a perfect
  vector. That is the number that justifies per-neighbour tokens.
- **Stays near 1.7%** — LoRA helps the oracle and not the graph, which would say the
  neighbours' embeddings do not determine the target at all. The Voyage probe (below) is the
  cheap way to check that before building anything.

### Known limitations

- **Still no era-4 945-node text baseline**, so the absolute comparison remains against a
  prompt known to be ~1.46 points weak on the 48-node stratified sample.
- **The `gate=0` arm spends ~7 h to produce one distinct string.** Left on because the
  walltime fits and the control has been lost once already.
- **Walltime is untested with the encoder training.** Step 4 bypassed the encoder under
  `--oracle`; this run adds 2-hop neighbourhood sampling and an encoder backward pass per step.

---

## Step 6 — one token per neighbour, GATv2, and r=64

### What step 5 actually returned

The graph+LoRA rerun finished (job 14437531) and it is a **negative result**, not the
improvement the step was designed to produce.

| arm | LLM | judge acc | ROUGE-1 | distinct | top-1 ID | med rank |
|---|---|---|---|---|---|---|
| graph, kv192 (step 2) | frozen | 1.737 | 0.2881 | 809 | 1.69% | 152 |
| **graph + LoRA (step 5)** | LoRA r=16 | **1.490** | 0.2829 | **453** | **0.95%** | 314 |
| oracle + LoRA (step 4) | LoRA r=16 | 2.712 | 0.3090 | 944 | 11.96% | 17 |
| text baseline (era 1) | frozen | 3.592 | 0.3364 | 945 | 39.4% | 2 |

LoRA made the graph arm worse on every metric. The failure mode identifies itself:

| | step 2 (frozen) | step 5 (LoRA) |
|---|---|---|
| distinct abstracts | 809 / 945 | 453 / 945 |
| singletons | 742 | 324 |
| largest identical cluster | 11 rows | 61 rows |
| rows scoring 3+ | 136 | 47 |

That is **conditioning collapse**. Given a prefix that carries little about the paper, a
decoder with the capacity to adapt minimises loss by ignoring it and emitting a good generic
abstract. LoRA did not fail to open the channel — it opened it, found little coming through,
and learned a template.

The `gate=0` arm confirmed the floor: one identical abstract ×945, judge accuracy at the
minimum. So within the LoRA arm the graph is worth about **+0.5 judge points over nothing**,
while LoRA itself cost about −0.25 against the frozen run. Both are true at once.

### What this rules in and out

Against step 4, the partition is now measured rather than argued. Same 32 tokens, same 40
layers, same LoRA, same everything except what the adapter is fed:

- fed the **true** target embedding → 11.96% top-1
- fed the **GraphSAGE pooled** embedding → 0.95% top-1

**The channel works. The single pooled 1024-d vector is what does not carry the paper.** So
the fix has to add *information*, not capacity — which is why a bigger LoRA alone would have
been the wrong next move, and why it is not run alone here.

### What was built

Three changes at once, against `checkpoints/qwen14b-graphonly-kv192-lora`.

**1. One input token per 1-hop neighbour** — the change that matters. Every run up to step 5
compressed the whole reference list into one vector before the LLM saw any of it. Now each
reference arrives as its own token.

They enter as **input embeddings**, not as KV prefix. The distinction is the point: a
KV-prefix token is something attention can look at, an input-embedding token is something the
40 layers *compute on* — and comparing references against each other is the operation the
whole idea rests on. It is also 16× cheaper per token (hidden 5,120 against a KV token's
40 × 2 × 8 × 128 = 81,920 values), so the new path is one 5.3M-parameter projection rather
than a second 17M adapter.

The sequence the model sees:

```
positions 0 .. P-1          KV prefix        (pooled seed vector, all 40 layers)
positions P .. P+k-1        neighbour tokens (one per reference, input embeddings)
positions P+k .. P+k+L-1    prompt and target
```

- `NUM_NEIGHBOUR_TOKENS = 64`, a ceiling not a target: a node contributes `min(out_degree, 64)`.
  Out-degree is 5–100 (median 52), so 64 covers 64% of eval nodes whole.
- Which 64 is decided by `neighbour_cap`'s centroid ranking — the **same score the text prompt
  trims by**, so the graph arm and the text arm select the same references and the comparison
  between them is about the encoding, not the sample.
- Ordered closest-first, so RoPE position encodes relevance. This is a choice, not a neutral
  default: a set is being given an order it does not have. Recorded as a limitation.
- `NeighbourTokenAdapter` mirrors `PrefixAdapter`: `token = scale · LayerNorm(base + gate · proj(h))`,
  weights shared across neighbours, `gate` initialised to zero. So training begins as ordinary
  soft-prompt tuning, and `gate=0` at eval keeps the k slots while removing their content.
- `scale` is set from the mean row norm of Qwen3's own embedding matrix before training. An
  unscaled projection of GNN outputs is out of distribution for the embedding layer, and that
  degrades quietly rather than raising.

**2. GATv2 replaces GraphSAGE.** `GATv2Conv(1024, 256, heads=4)`, concatenated back to 1024, so
swapping the aggregation does not also change the width. `--gnn-conv sage` reproduces every
earlier run. Encoder parameters 4,203,520.

**3. LoRA r 16 → 64, and `k_proj` added.** The rank is raised only now because rank is worth
paying for once the prefix carries something to read. `k_proj` reverses the rule the earlier
runs followed, for a specific reason: the KV prefix supplies its keys directly, so `k_proj`
never touches it — but a neighbour token is an ordinary input embedding whose key is
*computed* by `k_proj` at every layer. With it frozen, the model has no way to make the new
tokens attendable. `o_proj` and the MLP stay frozen, for the original reason.

### Why three at once

A deliberate trade against the single-change discipline of steps 1–5, taken because GPU time
is short. It costs attribution: a gain cannot be split between the three from this run alone.
What buys most of it back is cheap, generation-only, and already in the eval job:

| arm | what it isolates |
|---|---|
| `adapter` | the result |
| `ngate0` | k slots kept, content removed — **the step-5 architecture on step-6 weights** |
| `gate0` | the graph disconnected from *both* channels — the floor |

`ngate0` is the important one: it is the pooled-only arm measured on the same weights, so
`adapter − ngate0` is what the neighbour tokens added.

### Correctness

The thing that could break silently is the two-pass prefill. Training feeds
`[neighbour tokens ++ prompt]` in one forward as `inputs_embeds`; generation cannot, because
`generate()` in transformers 4.51.3 only forwards `inputs_embeds` when `cache_position[0] == 0`,
and with a prefix in the cache it never is — the embeddings would be dropped for an empty
`input_ids`. So generation prefills the neighbour tokens into the cache first
(`prefix_generation.prefill_neighbours`) and decodes on the `input_ids` path.

Splitting there is exact, because attention is causal: a neighbour token's keys and values
cannot depend on a prompt token that follows it. `verify_neighbour_tokens.py` holds that
numerically rather than by argument — **max drift 9.5e-07** over the prompt positions, with two
negative controls (dropping the prefill moves the output by 1.7; changing the tokens' content
moves it) so the check cannot pass vacuously. It also checks that the 1-hop rows are exactly
the seed's references and not its citers, and that the torch centroid score matches
`neighbour_cap`'s numpy one to 6e-08.

All five verify scripts pass, and the job scripts run `verify_neighbour_tokens.py` before
spending GPU hours.

### How to read the result

Against step 5's 1.490 / 0.95% and step 4's ceiling of 2.712 / 11.96%:

- **Near the oracle** — per-neighbour tokens were the whole bottleneck; the references do
  determine the abstract, and pooling was throwing it away.
- **Clearly above step 5 but well short** — the compression was part of the problem and the
  remaining gap is the encoder or the task itself. `ngate0` says which of the three changes paid.
- **Still near 1.5, with `adapter ≈ ngate0`** — the neighbour tokens added nothing the pooled
  vector did not already carry, which would be strong evidence that a paper's references do not
  determine its own abstract. That is a real finding for the thesis, not a failed run.

### Known limitations

- **Three changes at once**, so this run alone cannot attribute a gain. `ngate0` recovers the
  neighbour-token split; separating GATv2 from r=64 needs another run.
- **Voyage is an input representation, not ground truth.** Everything here encodes neighbours
  through embeddings we inherited; a limit of that encoding is indistinguishable from a limit
  of the graph. This is why the ridge-regression probe was dropped — it would have measured
  predictability of *Voyage*, not of the abstract.
- **RoPE imposes an order on an unordered set.** Mitigated by ordering on relevance, not fixed.
- **No era-4 945-node text baseline**, so the absolute comparison is still against a prompt
  known to be ~1.46 points weak on the 48-node stratified sample.
- **The text baseline gets the paper's own title; the graph arms get nothing.**
  `build_graph_only_prompt` takes no node argument. So part of the 12% → 39% gap is the title,
  not the channel, and a fair soft-prompt-vs-text comparison needs a no-title text arm.
- **Degree confounds judge accuracy.** The oracle arm, which sees no graph at all, still rises
  2.00 → 2.88 with degree. Any "more neighbours helps" claim from this run must be stratified
  by degree or it is measuring gold length.
- **Walltime is not measured.** The forward is ~25% longer and the eval job now has three arms.
  Expect ~21 h+ against the 24 h limit; a resubmit is free, since each arm appends only the
  node_ids missing from its own CSV.
