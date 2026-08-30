# Prompt sensitivity under a stronger model

Four prompt eras, one model, one node set. The question is whether the four instructions that
made almost no difference to Vicuna-7B make a difference to `gpt-5.6-luna`.

## Why

Three rewrites of the generation prompt moved mean `llm_accuracy` by 0.145 in total across
the 945-node capped evaluation set:

| era | run | mean accuracy |
|---|---|---|
| 1 | 2026-08-15 | 3.288 |
| 2 | 2026-08-20 | 3.259 |
| 3 | 2026-08-21 | 3.214 |
| 4 | 2026-08-23 | 3.143 |

Every rewrite changed 70–90% of the generated text and none of them changed the score, and
some changes went the wrong way: era 3 banned naming a method absent from the cited works,
and the rate of coined method names went *up*, 33.1% → 35.2% (`docs/adapter.md`, Stage 6c).
`docs/adapter.md` reads that as prompting being exhausted for the mean.

That reading has a confound. Vicuna-7B may simply not be able to follow instructions of this
length and specificity, in which case what the four eras measured is the 7B's instruction
following, not the prompts. Holding everything else fixed and swapping in a model that does
follow instructions separates the two: if the four prompts still land on one another under
`gpt-5.6-luna`, the wording genuinely does not matter for this task; if they separate, the
earlier null result was a property of the 7B.

This run does **not** measure the adapter. There is no graph prefix here — `gpt-5.6-luna` sees
only the reference list, which is the no-graph condition.

## What is held fixed

Everything except the instruction:

- **Nodes** — the same 48 in all four arms.
- **References** — the same centroid cap, built with the *Vicuna* tokenizer against a
  4,096-token context, 10 prefix tokens and a 320-token reserve. None of those describe
  `gpt-5.6-luna`; they are kept so GPT reads the same bibliography the 7B read.
- **Chat wrapper** — the Vicuna `SYSTEM … USER: … ASSISTANT:` framing, split into a system
  and a user message, as `src/generation/generate_gpt.py` does for the archived gpt-5.4-mini
  baseline.
- **Decoding** — same model, same `reasoning_effort`, same `max_completion_tokens` in every
  arm. Temperature and top_p are not sent; the 5.6 models reject them.

One thing is *not* held fixed, and cannot be: the token budget is shared between the
instruction and the references, so a longer instruction buys fewer references. Measured on
these 48 nodes:

| era | instruction tokens | refs shown (mean) | prompt tokens (mean) |
|---|---|---|---|
| 1 | 48 | 13.25 | 3,372 |
| 2 | 118 | 13.10 | 3,414 |
| 3 | 176 | 12.92 | 3,424 |
| 4 | 233 | 12.75 | 3,438 |

Half a reference between the shortest and longest instruction — era 4's instruction costs
185 tokens more than era 1's, and pays for it out of the bibliography. This is not introduced
here; it is exactly how the 7B runs behaved, and reproducing it is the point.

## The node set

48 nodes copied from `../sentiment_analysis/03_prompt_fix/comparison_dataset.csv`, the sample
rounds 2 and 3 both judged. Out-degree 6–98, mean 53.2; all 48 are in
`data/frozen/eval_nodes_capped.json`.

They are **not a random draw**. They were stratified on how each node's accuracy moved between
era 2 and era 3 — cell A collapsed, B survived, C recovered, D stayed down — so a mean over
these 48 is not an estimate of the mean over the 945. What they buy is the per-node trace:
each already carries three Vicuna generations and three judge scores, so a GPT output can be
laid against the 7B's on the same paper under the same prompt. `test_set.csv` carries `group`,
`score_v1`, `score_v2` and `score_v3` for exactly that.

## Layout

```
prompt_sensitivity_gpt/
├── prompts.py         the four instructions verbatim + the reference block each wraps them in
├── build_test_set.py  48 nodes -> test_set.csv and the four arm test sets
├── generate.py        one arm -> generations.csv
├── test_set.csv       the node selection: ids, out-degree, gold, group, Vicuna scores
└── 0N_eraN/
    ├── test_set.csv     the 48 nodes + that era's finished prompt, refs kept, prompt tokens
    └── generations.csv   node_id, mag_id, original_abstract, generated_abstract
```

`0N_eraN/generations.csv` uses the four columns every run directory in this repository uses,
so `src/evaluation/evaluate_results.py` scores it unchanged.

## Running it

```bash
conda activate DyTAG
python experiments/prompt_sensitivity_gpt/build_test_set.py       # rebuild all four arm sets
python experiments/prompt_sensitivity_gpt/generate.py --era all   # 4 x 48 = 192 API calls

python src/evaluation/evaluate_results.py \
    --input  experiments/prompt_sensitivity_gpt/01_era1/generations.csv \
    --output experiments/prompt_sensitivity_gpt/01_era1/evaluated_result.csv
```

`generate.py` is resumable — a rerun appends only the node_ids missing from the output CSV.
Failures are written as `generated_abstract = "ERROR: ..."` and skipped by the evaluator.

The evaluator's `--nodes` join adds `num_of_neighbors` from the old 200-node ogbn subset,
which none of these 48 are in, so that column comes back empty. Out-degree is already in
`test_set.csv`; use that.

## Verification

- The 48 era-2 prompts were rebuilt through `src/adapter/text.capped_example` at commit
  `4105279` and compared string-for-string with `02_era2/test_set.csv`: **0 mismatches**, in
  the prompts and in the reference lists.
- The 48 era-1 prompts were rebuilt the same way through the working-tree `text.py`:
  **0 mismatches**.
- Era 3 has no reference implementation in the repository — no commit ever carried it — so it
  is transcribed from `../sentiment_analysis/03_prompt_fix/prompt.md` and can only be checked
  by eye.
- Era 4 was supplied by the author and is not in any commit either, so it cannot be diffed
  against a pipeline. What was checked is the transcription: the pasted text and
  `prompts.ERA4` are **identical once line wrapping is collapsed** — same em dashes, same
  typographic quotes around "the authors", same hyphen in `150-180`.
- One live call to `gpt-5.6-luna` on an era-3 prompt: 3,188 prompt tokens, 1,048 completion
  tokens of which 808 reasoning, `finish_reason=stop`, 169 words — inside the 150–180 the
  prompt asks for. `max_completion_tokens` is set to 2,048 to leave room for the reasoning
  tokens; a run that spent the whole budget thinking would return an empty string, which
  `generate.py` writes as an `ERROR:` row rather than as a silent blank.

## Results (2026-08-29)

All four arms generated with `gpt-5.6-luna`, `reasoning_effort=medium`, and scored with
`src/evaluation/evaluate_results.py` (judge `gpt-5.4-mini`), 48/48 rows in every arm.

| | era 1 | era 2 | era 3 | era 4 |
|---|---|---|---|---|
| gpt-5.6-luna, mean `llm_accuracy` | 4.583 | 4.938 | 5.417 | **5.812** |
| Vicuna-7B, same 48 nodes | 4.292 | 3.771 | 3.667 | — |

**The prompts separate under the stronger model, and in the opposite direction.** Era 1 → era 4
is **+1.23** (Wilcoxon paired, n=48, 31 wins / 10 ties / 7 losses, p=0.0002). The same four
rewrites moved Vicuna *downward* on the same nodes, 4.292 → 3.667. This is a crossover, and it
qualifies `docs/adapter.md` Stage 6c: prompting was exhausted for the 7B, not for the task. The
era-4 instruction, worst of the four under Vicuna at 3.143 over the 945, is the best here.

Individual steps do not carry that claim on their own — era1→2 p=0.46, era2→3 p=0.18,
era3→4 p=0.05. What is supported is that era 4 beats era 1, not that each rewrite helped.

Against the 7B: GPT era 4 vs Vicuna era 3 on the same 48 nodes is **+2.15**, p=9.3e-07.

### The scores are not recalled abstracts

`gpt-5.6-luna` may have these arXiv papers in its training data, so a high score could be
recall of the target rather than reconstruction from its bibliography. The score *distribution*
cannot settle this — it is bimodal, which is exactly the shape a mixture of "recognised" and
"inferred" papers would produce. Three lexical measurements settle it instead, over all 192
generations:

| | era 1 | era 2 | era 3 | era 4 |
|---|---|---|---|---|
| nodes leaking a gold-only name | 0/48 | 1/48 | 0/48 | 0/48 |
| bigram overlap with the gold | 0.028 | 0.038 | 0.032 | 0.027 |
| bigram overlap with the nearest cited reference | 0.067 | 0.061 | 0.079 | 0.055 |
| near-verbatim copies of one reference (>0.5) | 0/48 | 0/48 | 0/48 | 0/48 |

A *gold-only name* is an acronym, CamelCase name or alphanumeric identifier that occurs in the
target's own abstract and in **none** of the references the prompt showed — it cannot come from
the prompt, so it can only come from the model. A recalled abstract would be full of them. Era 4
produces none, and the largest gold overlap anywhere in the run is 0.115.

Decisive comparison: accuracy rose 2.15 points over Vicuna while overlap with the gold stayed at
**0.027**, Vicuna's own figure in Stage 6c, and the output stayed roughly twice as close to a
cited reference as to the target — the same ratio the 7B had. The copy rate went 10.4% (Vicuna,
945 nodes) to 0%. GPT scores far higher while sitting *further* from the gold text than the 7B
did. That is better inference, not retrieval.

### The bimodality is the sample, not the model

The two humps come from how the 48 were drawn. Stratified on Vicuna's era-2 → era-3 movement,
the cells order GPT's era-4 scores almost as well as they order Vicuna's:

| cell | n | Vicuna v3 | GPT era 4 |
|---|---|---|---|
| D_stuck_floor | 11 | 2.09 | 4.27 |
| A_collapsed_top | 15 | 3.53 | 5.93 |
| C_rescued_floor | 12 | 3.08 | 6.08 |
| B_survived_top | 10 | 6.30 | 7.00 |

Spearman(GPT era 4, Vicuna era 3) = **+0.36**, p=0.011: papers hard for the 7B are hard for GPT
too. Out-degree explains nothing (+0.05, p=0.74). The low hump is mostly the floor cells and the
high hump mostly the top cells — a property of the sampling design, which is why this folder's
means are not estimates for the 945.

## Open questions and limitations

- **Judge affinity is now the largest threat, ahead of memorisation.** The generator is
  `gpt-5.6-luna` and the judge is `gpt-5.4-mini` — same vendor. Part of +2.15 may be register
  preference rather than accuracy, and unlike memorisation nothing here measures it. A blind
  pairwise round in the style of `../sentiment_analysis/`, or a non-OpenAI judge, would.
- **The recall test is indirect.** It detects a model that reproduces the target's *names*; a
  model that recalled the target's content while renaming everything would pass it, though the
  0.027 gold overlap makes that unlikely. The direct test — hand the model the reference list
  and ask it to name the paper that cites all of them — is ~48 calls and has not been run.
- **Era 4 contradicts its own reference block.** Its voice clause says never mention the cited
  works as a collection or *how many there are*, while the block above it opens `This paper
  cites 53 works; the 13 most representative are listed below`. Era 3 carried the same tension.
  It is reproduced rather than repaired, because the 08-23 run had it; but if a GPT arm starts
  describing the bibliography, this is the first place to look, and removing the count line is
  a cleaner experiment than rewording the instruction.
- **48 nodes is a small, deliberately non-random sample.** It can show that the four prompts
  separate; it cannot estimate by how much over the full 945.
- **The judge is `gpt-5.4-mini` and the generator is `gpt-5.6-luna`** — same vendor, and the
  judge is the same one that scored the Vicuna runs. Keeping it fixed is what makes the
  GPT-vs-Vicuna comparison possible, but a same-family judge may not be neutral about the
  generator's register.
- **The reference cap is calibrated to a model that is not being used.** GPT could be given
  every reference; it is deliberately not, because then the model and the evidence would both
  have changed. A follow-up with the cap lifted would measure something different and worth
  measuring separately.
