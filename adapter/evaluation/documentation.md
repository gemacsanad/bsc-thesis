# Stage 5 — generating abstracts for the frozen evaluation set

**Date:** 2026-08-04

Continues `adapter/documentation.md`, which covers Stages 1–4.

## Goal

Turn the trained checkpoint into text that can actually be scored. Training measured only
teacher-forced cross-entropy on the validation nodes (`valid/graph` vs `valid/gate0`); it never
generated a token and never touched the frozen 495-node evaluation set. This stage produces the
two CSVs the comparison rests on:

| condition | file | what it is |
|---|---|---|
| baseline | `eval_baseline.csv` | frozen Vicuna-7B, no prefix, nothing else |
| adapter | `eval_adapter.csv` | GraphSAGE + prefix adapter + the same frozen Vicuna-7B |

Same 495 nodes, same prompts, same order, same decoding. The only difference between the two
files is whether a graph-conditioned KV prefix is present.

## What controls the comparison

Everything except the prefix is held fixed, and each of these was a live choice:

- **Prompts come from `text.build_prompt` via `text.Corpus`** — the exact code training used, not
  a reimplementation. The target paper's own title is absent; the model sees only the titles and
  abstracts of the papers it cites.
- **Greedy decoding** (`do_sample=False`). Sampling noise would sit on top of the effect being
  measured and reruns would not reproduce. This is why the frozen-LLM condition is *regenerated*
  here rather than reusing `test_data/vicuna_7b`, which sampled at temperature 0.35 and is
  therefore not a drop-in comparison.
- **bfloat16 in both conditions.** `scripts/generate_test_csv.py` loaded the old baseline in
  8-bit, but the adapter was trained against a bf16 model and its prefix is calibrated to those
  numerics. Running the baseline in 8-bit would let quantisation error masquerade as an adapter
  effect.
- **Batch size 1.** Prompts run from ~500 to ~5,600 tokens, and padded batches combined with a
  pre-filled prefix cache were never verified (Stage 4 step 1 flagged exactly this gap). One node
  at a time removes the question rather than answering it.

## Context overflow: 63 nodes written as errors

With `max_new_tokens=256` and the 10 prefix positions charged against the 4,096-token context,
**63 of 495 prompts (12.7%)** do not fit. Measured with the real Vicuna tokenizer:

| | prompt tokens |
|---|---|
| mean | 2,684 |
| median | 2,630 |
| p90 | 3,947 |
| max | 5,582 |

Following `scripts/generate_test_csv.py`, those rows are still written with
`generated_abstract = "ERROR: prompt too long (N tokens > 3830 limit)"`;
`scripts/evaluate_results.py` already skips such rows. **The 10 prefix positions are charged in
both modes**, even though the baseline has no prefix, so the two CSVs carry identical ERROR rows
and the comparison runs on the same 432 nodes.

This resolves the question `text.py` left open, and it is not a neutral resolution: Stage 4c showed
the overflowing nodes concentrate at high out-degree — at degree 15 more than half of all
candidates overflow — which is precisely where the graph has the most to contribute. **The scored
set is biased against the adapter's best case, and results must be reported as 432/495 nodes, not
495.** Truncating neighbour abstracts to fit was the alternative; it was rejected because the LLM
would then read less than the GNN sees, which flatters the adapter in the opposite direction.

## The `generate()` bug, and the fix

Stage 4 step 1 established that `generate()` cannot be used unmodified. That was re-derived
against the installed `transformers==4.39.3` source, because the notebook recording it is no
longer in the repository.

Every search method in `generation/utils.py` opens with
`model_kwargs["cache_position"] = torch.arange(cur_len)`, where `cur_len` is the length of
`input_ids` alone — the P prefix positions already in the cache are not counted. That value
survives `LlamaForCausalLM.prepare_inputs_for_generation` (`modeling_llama.py:1309`,
`cache_position = cache_position[-input_length:]`), and `cache_position` is what sizes the causal
mask. Every row shifts by P, so each token attends mostly to prefix positions instead of to the
tokens before it. Passing `cache_position=` into `generate()` does nothing — it is overwritten
before the loop starts.

`prefix_generation.patch_for_prefix` corrects it on entry to `prepare_inputs_for_generation`, then
delegates to the stock method. Three details are load-bearing:

- **The correction is `+P` on every step, not only the first.**
  `_update_model_kwargs_for_generation` increments the dict `generate` holds, which never sees
  what the override returns, so the arriving value is always short by exactly P.
- **Correcting on entry, not exit**, also repairs `past_length` — the stock method derives it from
  `cache_position[0]` and uses it to decide how much of `input_ids` is still unprocessed.
- **`PrefixCache` from `train.py` cannot be reused.** It is deliberately read-only so gradient
  checkpointing can recompute a layer without the cache growing. Decoding needs the opposite, so
  the prefix goes into a real `DynamicCache`, which then grows P → P+L → P+L+1 on its own.

Positions need no correction: the stock `position_ids` path (cumsum over the P+L mask) yields
`P .. P+L-1` for the body, which is the convention `train.py` trains under.

## Verification

`verify_generation.py` — 4-layer random Llama, CPU, ~10 s. It compares two runs of the same model,
so the weights are irrelevant. A token sequence is split into `prefix ++ body`; the *real* key/values
of `prefix` are injected and `body` is generated from, then compared against plain generation over
the whole sequence. Real KV of real tokens is used rather than a zero or random prefix, because
only then is the correct answer already known.

| check | result |
|---|---|
| unpatched vs reference (negative control) | **19/20** generated tokens differ |
| patched vs reference | **20/20** tokens match exactly |
| prefilled forward pass vs plain forward | max logit error **1.9e-07** |
| `transformers` version asserted | 4.39.3 |

The negative control is the point: without it, a passing patched run would say nothing about the
patch. It is also the measure of the hazard — get this wrong and 19 of 20 tokens are wrong, with no
error raised.

Separately checked against the real data (not automated, run once):

- `iter_prefixes` emits node_ids in exactly `eval_nodes.json` order in both modes, so a row can
  never be paired with another node's neighbourhood.
- Prefix shape from a real `NeighborLoader` batch is `(1, 32, 2, 32, 10, 128)` — the cache layout.
- `mag_id` is present for all 495 eval nodes; no nulls.

## Inputs / outputs

| | |
|---|---|
| In | `adapter/eval_nodes.json`, `adapter/checkpoints/epoch2.pt`, `joint_dataset/{nodes.csv, edges.csv, node_embeddings.npy}`, `lmsys/vicuna-7b-v1.5` |
| Out | `adapter/evaluation/eval_baseline.csv`, `adapter/evaluation/eval_adapter.csv` — columns `node_id, mag_id, original_abstract, generated_abstract`, matching `scripts/evaluate_results.py` |
| Run | `python adapter/evaluation/generate.py --mode baseline`<br>`python adapter/evaluation/generate.py --mode adapter --checkpoint ../checkpoints/epoch2.pt` |

Both modes are resumable: rerunning appends only the node_ids missing from the output CSV, so a
walltime kill costs at most one node. Baseline mode never loads the graph or the 1.5 GB embedding
matrix.

Scoring is then `scripts/evaluate_results.py` unchanged, once per CSV — ROUGE-1/2/L, BERTScore, and
the GPT judge (accuracy / coherence / language).

## The job (`Komondor_files/run_generate_eval.sh`)

Modelled on `run_train_adapter.sh`: same account, `gpu` partition, GPU request, `$BASE_DIR:/app`
bind, offline HuggingFace environment, and the same `sort -V` idiom for finding the newest
checkpoint. The `adapter-gpu.sif` image needs **no rebuild** — `generate.py` imports nothing
`train.py` did not. No `adapter/.env` is required either; nothing here logs to W&B.

Three deliberate differences:

**Both conditions run back to back in one job.** Same node, same GPU, same driver, so nothing but
the prefix can differ between the two CSVs. It also costs one queue wait instead of two.

**It refuses to start without a checkpoint** rather than discovering the problem after the GPU is
allocated and the 13 GB model is loaded.

**`verify_generation.py` runs first**, before either condition. It is ~10 s on CPU and it guards a
silent failure — a `cache_position` off by 10 produces fluent abstracts, not an error — so it
belongs before the GPU hours, not after.

Walltime is **8 h** against an estimated ~1.5 h per condition, absorbing a ~2.5x miss on an
estimate that has never been checked against a real A100. Both modes are resumable, so a
resubmit after a timeout costs at most one node, and an already-finished condition costs only its
model load. Rerunning the script is safe.

### Verified

The shell logic was exercised locally under `bash`: with no checkpoints present the script prints
its error and exits 1 with no stray output; with `epoch1/2/10.pt` present `sort -V` selects
**epoch10**, not epoch2. `SINGULARITYENV_CKPT` reaches the inner shell as `$CKPT`. Nothing else in
the script can be tested off Komondor.

## Limitations and open questions

- **Never run against the real Vicuna-7B.** The patch is verified on a 4-layer random model on CPU;
  whether bf16 numerics or the GPU attention implementation (`sdpa` vs `flash_attention_2`) change
  anything is the first Komondor run's job. The Stage 4 check had the same limitation.
- **Runtime is a guess.** ~1.5 h per condition is derived from an assumed ~35 tok/s for unbatched
  7B bf16 decoding at a ~2,700-token context; nothing has timed it. First job settles it.
- **Only two of the three conditions exist.** The `gate=0` ablation — same trained weights, graph
  disconnected — is what isolates the graph from plain prefix tuning, and it is not built here. It
  is a small addition (zero `adapter.gate` before generating) and belongs to Stage 6.
- **432 scorable nodes, not 495**, and the missing 63 are the high-out-degree end. Any headline
  number needs this stated alongside it.
- **`eval_baseline.csv` is not comparable with `test_data/vicuna_7b`** — different decoding, different
  precision, different node set. Do not mix them in one table.
