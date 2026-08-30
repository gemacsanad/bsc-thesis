# Does the judge rubric decide how large the graph effect looks?

Every accuracy number in this project comes from one judge prompt, in
`src/evaluation/evaluate_results.py`. This folder holds the same two generation arms — the
no-graph baseline and the GraphSAGE-prefix adapter — scored under four rubrics, to see how
much of the measured graph effect is a property of the instrument.

The generations never change, and neither does the judge model. Only the rubric does.

## The honest framing, first

The stated goal was to maximise the gap between baseline and adapter. Tuning a measurement
until it produces a wanted result is how instruments stop being instruments, so the design
here is built to make that failure visible rather than to avoid the question:

- **R1 is the current rubric, verbatim**, scored by the project's own judge. It is the
  reference the other three are read against, not a competitor. On these 300 nodes it gives
  +0.433, against +0.473 over the full 945 — the subset is slightly less favourable to the
  adapter, which is worth knowing before any variant is called an improvement.
- **Every rubric is scored on a negative-control pair** — two adapter runs that differ only in
  repetition penalty. Nothing about the graph separates them. A rubric that widens the
  baseline-vs-adapter gap *and* this one is not detecting the prefix, it is spreading every
  comparison it touches.
- **Every rubric is scored on identical text twice**, giving its own test-retest noise. A gap
  smaller than a rubric's noise floor is not a measurement.
- **All four results get reported**, including the ones that shrink the gap.

A rubric earns the headline only if the graph gap grows while the other two hold still. There
is one defensible reason to change the rubric at all, and it is not gap size: R1 spends a third
of its accuracy definition on methodology, which `docs/adapter.md` records as unrecoverable
from a bibliography — 70% wrong for Vicuna, **95% wrong for gpt-5.4-mini**. Scoring a dimension
no arm can achieve adds variance that cannot discriminate. Removing it makes the instrument
measure what the task can deliver. That argument stands whether or not the gap widens, and it
is the argument the write-up should rest on.

## What the graph actually changes

From `analyse_arms.py`, over all 945 era-1 nodes, no API calls. This is what the rubrics were
written from.

**The adapter is more precise, not more complete.**

| | baseline | adapter |
|---|---|---|
| gold-term recall | 0.0876 | 0.0855 |
| gold-term precision | 0.0963 | **0.1183** |
| distinct terms used | 55.6 | 48.5 |
| bigram overlap with gold | 0.0596 | **0.0790** |

It says *less* and gets *more* of it right: 13% fewer distinct terms, 23% higher precision,
33% more bigram overlap with the gold, and slightly **lower** recall. Any rubric that rewards
completeness or coverage will therefore miss the adapter's actual advantage — which is one
concrete reason R1 may be understating it.

**The advantage is not extra neighbourhood information.** The centroid cap shows at most 15
references while the GNN reads the whole neighbourhood, so if the prefix were an information
channel the gap should grow with how many references the cap hid. It does not:
Spearman(gap, hidden references) = **−0.05, p=0.11**, and the largest gap of any bin (+0.71)
is at *zero* hidden references, where the GNN saw nothing the prompt did not. The prefix works
as a focusing signal on text the model already has, not as a second source of facts.

**The advantage lives in the upper tail.**

| | ≤2 | ≥5 | mean |
|---|---|---|---|
| baseline | 46.7% | 5.8% | 2.815 |
| adapter | 34.9% | **13.7%** | 3.288 |

Under R1 both arms are pinned near the floor — half the baseline's outputs score 1 or 2 — and
a compressed scale cannot express a difference. The adapter's win is that it produces 6s, 7s
and 8s (102 of 945) where the baseline produces almost none (31).

**The baseline's dominant failure is writing one cited paper instead of the citing paper.**
Share of outputs where a single cited reference accounts for more than 30% of the generated
bigrams: baseline **73.2%**, adapter **58.7%**. (The *mean* overlap runs the other way and is
confounded by generation length, so the share is the readable figure — this one is suggestive,
not settled.) The qualitative pattern is unambiguous, e.g. node 5182, out-degree 53:

- **gold** — AraBERT, a pre-trained language model for Arabic
- **baseline** (scored 2) — "a new application of pre-trained word representations… train
  high-quality word vector representations by using a combination of known tricks" — this is
  one cited paper, word2vec/fastText, written out as if it were the target
- **adapter** (scored 8) — "a new pre-trained language model for Arabic, called AraBERT"

Same at node 41817 (baseline writes Bitcoin-NG; gold and adapter are a blockchain security
survey) and node 68995 (baseline writes one GCN attack paper; gold and adapter are a survey of
graph adversarial attack and defence). The graph tells the model the *scope and topic the
bibliography converges on*. That is exactly the "problem and task" axis, and it is why the
proposal to stop scoring methodology is well founded rather than convenient.

## Why the judge is gpt-5.4-mini and not the cheaper model

`gpt-5.6-luna` scored R1 over the same 300 nodes before the judge was settled. Same rubric,
same generations, only the judge different — `judge_comparison/luna_R1_*.csv`:

| judge | baseline | adapter | gap | SD of paired diff | Cohen's *d* |
|---|---|---|---|---|---|
| gpt-5.4-mini | 2.850 | 3.283 | **+0.433** | 1.702 | **0.255** |
| gpt-5.6-luna | 2.103 | 2.383 | +0.280 | 1.667 | 0.168 |

The SD column is what decides it. A harsher judge that compressed the whole scale would shrink
the gap and the spread together and leave the effect size alone; luna's spread is
indistinguishable from mini's (pooled 1.33 vs 1.35) while its gap is a third smaller. It is
genuinely less able to separate the two arms — about 1.5x less per unit of variance — not
merely stricter. The two agree on which papers are good (per-node Spearman **+0.77**), so this
is sensitivity, not disagreement.

Cost, the reason luna was tried, does not hold up either: at `reasoning_effort=low` it emits
235 completion tokens per judge call (172 output + 63 reasoning) against mini's 169, so it
needs a substantially lower per-token price merely to break even.

**Choosing a judge because it shows a bigger difference is the same instrument-selection
problem as choosing a rubric that way**, so it is worth being precise about which argument is
doing the work. The one that does not depend on the outcome is reproducibility: the 5.6 models
reject `temperature=0.0` and run only at temperature 1, giving a measured per-item SD of
**0.44**, against roughly a point of disagreement in 1 node out of 6 for mini at
`temperature=0`. That argument would hold even if luna had shown the larger gap. The
sensitivity argument is supporting evidence, and it is only clean here because the spreads
matched — had they not, a larger gap would have proved nothing.

## The four rubrics

Only the `accuracy` definition changes; `coherence` and `language` are held identical
throughout, so the comparison has one variable.

| | what it does | why |
|---|---|---|
| **R1** control | current rubric, verbatim | the reference point the other three are read against; without it nothing else is interpretable |
| **R2** problem & task | scores research problem and specific task only, methods explicitly excluded | methodology is unrecoverable and pins both arms to the floor |
| **R3** decomposed | separate domain / problem / task / scope scores plus an overall | says *which* dimension carries the effect; `scope` is where the survey-vs-method failures live |
| **R4** target vs source | R2 plus: describing a cited work rather than the citing paper caps accuracy at 3 | names the baseline's dominant failure, which R1 has no language for |

**R4 is the one to distrust.** It encodes the specific failure the baseline was observed
making, so it is the most likely to widen the gap and the least likely to be measuring
something real. That is precisely what the negative control and the noise floor are for. If R4
widens the graph gap and leaves both controls flat, it is a sharper instrument; if it widens
all three, it is a thumb on the scale.

## Design

- **Pair**: era 1, `2026-08-15-14207605_{vicuna-baseline,adapter}` — same generation job, same
  prompt, same nodes, differing only in the prefix. The 2026-08-25 3-layer run is era 4 and its
  baseline was never judged; pairing those two would have put the four prompt rewrites inside
  the measured graph effect.
- **Nodes**: 300, stratified by out-degree over the 945 capped eval nodes (mean out-degree 51.4
  against the full set's 51.8). Selected on out-degree alone, never on a score.
- **Controls**: 150-node subset — the adapter re-scored under each rubric, and the two era-4
  adapter runs that differ only in repetition penalty.
- **Judge**: `gpt-5.4-mini` at `temperature=0.0` — the project's own evaluator, unchanged, with
  the same JSON response format, retry policy and user-message layout as
  `src/evaluation/evaluate_results.py`. Only the system prompt differs between cells, so
  results here sit beside every figure in `docs/adapter.md`.

  `gpt-5.6-luna` was tried first, for cost, and rejected. The measurement is in
  `judge_comparison/` and is summarised below; it is a result in its own right, not a false
  start.

## Layout and running it

```
rubric_sensitivity/
├── rubrics.py        the four judge prompts
├── runs.py           which generation runs are compared, and why those
├── analyse_arms.py   the no-API analysis above  -> arm_analysis.csv
├── build_subset.py   the frozen 300/150 node sets -> subsets.json
├── score.py          one rubric x one arm -> 0N_RN/<arm>.csv
├── report.py         graph gap vs noise floor vs control gap, and the R3 components
└── 0N_RN/            baseline.csv, adapter.csv, adapter_retest.csv, neg_a.csv, neg_b.csv
```

```bash
conda activate DyTAG
python experiments/rubric_sensitivity/analyse_arms.py
python experiments/rubric_sensitivity/build_subset.py
python experiments/rubric_sensitivity/score.py --all     # 4,200 judge calls, gpt-5.4-mini
python experiments/rubric_sensitivity/report.py
```

`score.py` is resumable and can be run cell by cell with `--rubric R2 --arm baseline`.

## Open questions and limitations

- **Not yet run.** Everything above the Design section is measured; the rubric comparison
  itself is 4,200 judge calls that have not been spent. No claim about R2/R3/R4 exists yet.
- **A wider gap under a new rubric does not make earlier numbers wrong.** Every figure in
  `docs/adapter.md` is an R1 figure produced by `gpt-5.4-mini`. Adopting a variant means
  rescoring the whole series — and because the judge changed here too, that rescoring has to
  fix *both* the rubric and the judge at once, or the prompt-era story, the rubric change and
  the judge change end up confounded together.
- **`temperature=0` is not exact determinism.** Rescoring 6 nodes under R1 against their
  stored `llm_accuracy` reproduced 5 exactly and moved 1 by a point. The judge is highly
  reproducible, not perfectly — which is why the retest control still has to be run rather
  than assumed to be zero, and why a single rubric's gap should not be read to three decimals.
- **The judge comparison is one rubric on one pair.** `judge_comparison/` shows luna
  separating these two arms less well under R1. It does not establish that luna is a worse
  judge in general, and a rubric that changes what is being judged could change that ordering.
- **R4's cap at 3 is a hard threshold on a judgement call.** "Reads as an account of a single
  earlier work" is not crisply decidable, and the judge is being asked to apply it
  categorically. Expect its noise floor to be the worst of the four; that is a reason to read
  the noise column before the gap column.
- **The single-source-capture measurement is not settled.** The share above 0.30 and the mean
  overlap disagree in direction, and both are confounded by generation length. The qualitative
  examples are stronger evidence than either number, which is a reason to treat this as the
  motivation for R4 rather than as a result.
- **The judge scores both arms with the gold abstract in front of it**, so it can reward
  surface similarity to the gold rather than correctness about the paper. All four rubrics
  inherit this; none of them addresses it.
