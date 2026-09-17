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
