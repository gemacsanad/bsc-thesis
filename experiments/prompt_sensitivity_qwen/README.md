# Prompt sensitivity under Qwen3-14B

The Qwen twin of [`../prompt_sensitivity_gpt/`](../prompt_sensitivity_gpt/). Four prompt eras,
one model, one node set, no graph prefix.

## Why

The four instructions moved the two models that have seen them in **opposite directions** on
the same 48 nodes:

| | era 1 | era 2 | era 3 | era 4 |
|---|---|---|---|---|
| Vicuna-7B | 4.292 | 3.771 | 3.667 | — |
| `gpt-5.6-luna` | 4.583 | 4.938 | 5.417 | **5.812** |

`+1.23` for GPT era 1 → era 4 (Wilcoxon paired, n=48, p=0.0002) against a *fall* for the 7B.
`../prompt_sensitivity_gpt/` reads that as prompting being exhausted for the 7B, not for the
task.

Qwen3-14B is the model this project now actually uses, and it sits between the two — open
weights at 14B, but modern and instruction-tuned. Whether it behaves like the 7B or like GPT
is not predictable from either, and the answer decides whether the prompt is still a live
variable for this thesis.

It is not idle. **The 945-node Qwen runs of 2026-09-10 were generated under era 1** —
`src/adapter/text.py`'s `INSTRUCTION` is the era-1 wording, unchanged by the Qwen migration
(`53fb068` kept the prompt text fixed deliberately), and so is its block framing. Era 1 is the
*worst* of the four under GPT by 1.23 points. If Qwen orders the eras the way GPT does, both
published Qwen arms are standing on the weakest prompt available.

## What is held fixed

Two comparisons have to survive at once, and they fix different things.

**Against GPT and Vicuna — the content.** The four `0N_eraN/test_set.csv` files are **copied
byte for byte** from `../prompt_sensitivity_gpt/` (md5-identical). Same 48 nodes, same
references in the same order, same instruction text, same centroid cap measured under the
*Vicuna* tokenizer against a 4,096-token context with 10 prefix tokens and a 320-token
reserve. None of those describe Qwen3-14B; they are kept so all three models read the same
bibliography. Nothing is rebuilt here — `generate.py` lifts the content out of the stored
`prompt` column, so the bytes reaching the tokenizer are provably the bytes in the CSV.

Copying rather than re-running `build_test_set.py` also sidesteps a live trap: that script
reads `BASE_MODEL` from `config.py`, which is now Qwen, so re-running it would re-cap the
prompts under Qwen's BPE and no longer reproduce the stored test sets
(`docs/qwen_migration.md`, step 4, known limitations).

**Against the 945-node Qwen baseline — the calling convention.** The Vicuna `SYSTEM … USER: …
ASSISTANT:` markers are discarded and the two messages are re-wrapped in Qwen3's own ChatML via
`apply_chat_template(..., enable_thinking=False)`, exactly as `src/adapter/text.build_prompt`
does. This is the keep-the-text-swap-the-delimiters split of `docs/qwen_migration.md` step 5,
and it is not cosmetic: a frozen Qwen3 handed non-ChatML input is off-distribution precisely on
instruction following, which is the faculty under test. `enable_thinking=False` does not remove
the reasoning block — it emits an empty, already-closed `<think>\n\n</think>`.

Decoding is imported from `src/adapter/config.py` rather than written down again, so this arm
cannot drift from the baseline: bfloat16, greedy (`do_sample=False`), `REPETITION_PENALTY`
1.00, `MAX_NEW_TOKENS` 320, batch size 1. There is no adapter and no checkpoint — all four arms
are the frozen-LLM condition.

## Era 1 is the baseline's prompt, not a fifth condition

Verified, not assumed:

- Era 1's reference lists are identical to `data/frozen/prompt_refs_capped.json` on **48/48**
  nodes.
- Era 1's user message, rebuilt through `src/adapter/text.user_message`, is identical to the
  stored one on **48/48** nodes.
- All **48/48** nodes are in `data/frozen/eval_nodes_capped.json`.

Same prompt, same model, same decoding: the era-1 generations should reproduce
`../2026-09-10-14364743_qwen14b-baseline/generations.csv` on those 48 rows. That is a free
end-to-end check on the whole pipeline, and the only thing tying this folder to the run it is
compared against. **Diff it first** — if era 1 does not reproduce the baseline, nothing else
here is interpretable.

## What is *not* held fixed

The token budget is shared between the instruction and the references, so a longer instruction
buys fewer references. Measured on these 48 nodes, against the frozen bibliography:

| era | instruction tokens | nodes with the frozen refs | refs shown (mean) | qwen tokens (mean / max) |
|---|---|---|---|---|
| 1 | 48 | **48/48** | 13.25 | 3,004 / 3,357 |
| 2 | 118 | 41/48 | 13.10 | 3,047 / 3,415 |
| 3 | 176 | 32/48 | 12.92 | 3,059 / 3,471 |
| 4 | 233 | 24/48 | 12.75 | 3,076 / 3,377 |

Half a reference between the shortest and longest instruction. This is not introduced here — it
is exactly how the 7B and GPT runs behaved, and reproducing it is the point. But it does mean
eras 2–4 are *not* the baseline's prompt, only era 1 is.

Nothing overflows: the budget is `4096 − 320 = 3,776` (no prefix in this condition) and the
largest prompt in any arm is 3,471, **305 to spare**. Qwen's BPE is ~10% more efficient than
Vicuna's SentencePiece on the same text (era 1: 3,004 against 3,372).

## Layout

```
prompt_sensitivity_qwen/
├── verify_prompts.py   CPU pre-flight, ~30 s — run before the GPU hours
├── generate.py         one arm (or all four) -> generations.csv
├── test_set.csv        the node selection: ids, out-degree, gold, group, Vicuna scores
└── 0N_eraN/
    ├── test_set.csv    copied byte for byte from ../prompt_sensitivity_gpt/
    └── generations.csv node_id, mag_id, original_abstract, generated_abstract
```

`0N_eraN/generations.csv` uses the four columns every run directory in this repository uses, so
`src/evaluation/evaluate_results.py` scores it unchanged.

## Running it

Upload two things to `$BASE_DIR` on Komondor:

```
experiments/prompt_sensitivity_qwen/
hpc/komondor/jobs/run_prompt_sensitivity_qwen.sh
```

then

```bash
sbatch hpc/komondor/jobs/run_prompt_sensitivity_qwen.sh
```

Needs `adapter-qwen-gpu.sif` and `Qwen/Qwen3-14B` in `$BASE_DIR/.hf_cache`; no checkpoint, no
`.env`, no W&B. All four arms run back to back in one job on one GPU, so nothing but the prompt
differs between the four files. Resumable — a rerun appends only the missing node_ids, so a
resubmit after a timeout costs at most one node.

Scoring, once the CSVs are back:

```bash
python src/evaluation/evaluate_results.py \
    --input  experiments/prompt_sensitivity_qwen/01_era1/generations.csv \
    --output experiments/prompt_sensitivity_qwen/01_era1/evaluated_result.csv
```

The evaluator's `--nodes` join adds `num_of_neighbors` from the old 200-node ogbn subset, which
none of these 48 are in, so that column comes back empty. Out-degree is already in
`test_set.csv`; use that.

## Verification

Everything checkable off Komondor has been run.

`verify_prompts.py`, locally under `DyTAG-qwen` (transformers 4.51.3): all five checks pass —
the Vicuna wrapper splits on all 192 prompts; era 1 rebuilds through `text.user_message` 48/48;
the ChatML wrapper opens `<|im_start|>system`, ends in the empty closed `<think>`, has no BOS
and ends assistant turns with `<|im_end|>`; no arm overflows; all 48 nodes are in the frozen
eval set.

`generate.py`'s loop was exercised against a fixture tree with a 2-layer random `Qwen3ForCausalLM`
on CPU — the idiom `src/adapter/verify_generation.py` uses. Generation writes the four columns
with the gold preserved; a rerun resumes and appends nothing; a forced-low budget produces
`ERROR: prompt too long (…)` rows rather than silent blanks.

`bash -n` passes on the job script.

## Open questions and limitations

- **Nothing has run on a GPU.** Walltime (6 h for 192 generations) is scaled from the 24 h job
  that produced 945 × 2 rows, i.e. ≤45 s/node — an upper bound off a walltime, not a
  measurement.
- **Eras 2–4 are not the baseline's prompt.** Only era 1 is (48/48). The other three show a
  trimmed bibliography on 7, 16 and 24 nodes respectively, so an era-4-vs-era-1 difference
  confounds the instruction with half a reference. Same confound as the GPT run, carried
  deliberately.
- **48 nodes, stratified not random.** They were drawn on how Vicuna's accuracy moved between
  era 2 and era 3, so a mean over them is not an estimate of the mean over the 945. What they
  buy is the per-node trace against Vicuna and GPT.
- **Judge affinity is untested here too.** The judge is `gpt-5.4-mini`. Against a Qwen
  generator that is at least *cross*-vendor, unlike the GPT run — which cuts the other way and
  makes GPT-vs-Qwen means not directly comparable on register grounds.
- **The reference cap is calibrated to a model that is not being used.** Kept on purpose:
  lifting it would change the model and the evidence at once. Worth measuring separately.
- **Era 4 contradicts its own reference block** — its voice clause bans mentioning how many
  cited works there are, while the block above opens `This paper cites 53 works; the 13 most
  representative are listed below`. Reproduced rather than repaired, as in the GPT run. If a
  Qwen arm starts describing the bibliography, look here first.
