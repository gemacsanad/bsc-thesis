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
