# Does the judge's rubric decide how big the graph effect looks?

Every accuracy score in this project comes from one judge prompt, in
`src/evaluation/evaluate_results.py`. Nobody had ever asked what that choice was worth. This
folder takes the **same two sets of generated abstracts** — the no-graph baseline and the
GraphSAGE-prefix adapter — and scores them **four times under four different judge prompts**.

The generations never change. The judge model never changes. Only the rubric changes. So any
difference in the measured graph effect is a property of the ruler, not the model.

## What we did

- **Two arms**, both from the 2026-08-15 era-1 run: frozen Vicuna-7B with no prefix, and the
  same model with the graph prefix. Same prompt, same nodes, differing only in the prefix.
- **150 papers**, picked by out-degree only — never by score, so no arm's result influenced
  which papers it was measured on.
- **Judge**: `gpt-5.4-mini` at temperature 0, exactly the project's own evaluator.
- **Two guards against fooling ourselves**, run under every rubric:
  - a **negative control** — two adapter runs that differ only in a repetition-penalty setting.
    Nothing about the graph separates them. A rubric that widens the real comparison *and* this
    one isn't detecting the graph, it's just spreading everything apart.
  - **all four results reported**, including the ones that make the effect look smaller.
- 1,800 judge calls. R1 cost nothing — see the note at the bottom.

### Why these four rubrics

First we measured what the graph actually changes, with no API calls (`analyse_arms.py`, all
945 papers). Three findings drove the design:

1. **The adapter is more precise, not more complete.** It uses 13% fewer distinct terms but
   gets 23% more of them right. Any rubric rewarding coverage will miss its advantage.
2. **Both arms are pinned near the floor.** Under the current rubric 76% of baseline outputs
   score ≤3. A squashed scale cannot show a difference that exists.
3. **The baseline's commonest failure is writing about the wrong paper** — describing one of
   the cited works instead of the paper citing them. E.g. paper 5182: the real abstract
   introduces AraBERT; the baseline confidently describes word2vec, one of its references. The
   adapter gets AraBERT. The current rubric has no language for this failure, so a fluent
   description of the wrong paper can still score mid-range.

## The four rubrics

Only the **accuracy** definition changes. Coherence and language are held word-for-word
identical throughout, so the comparison has exactly one variable.

| | what it scores | what it focuses on |
|---|---|---|
| **R1** *control* | the current rubric, verbatim: objective, problem domain **and methodology** | the reference point — not a candidate |
| **R2** *problem & task* | the research problem and the specific task **only**; methods, models, datasets and numbers explicitly excluded | "did it work out what this paper is trying to do?" |
| **R3** *decomposed* | R2, split into separate **domain / problem / task / scope** scores plus an overall | diagnostic: *which* dimension carries the effect |
| **R4** *target vs source* | R2 plus one override: if it reads as an account of a single cited work rather than the citing paper, accuracy is capped at 3 | names the baseline's commonest failure outright |

The reason to change the rubric at all — decided before any scores came in, and independent of
whether the gap grew — is that **methodology is not recoverable from a reference list**.
`docs/adapter.md` records methodology as wrong 70% of the time for Vicuna and 95% for
gpt-5.4-mini. R1 spends a third of its accuracy definition on a dimension no model can achieve.
That is noise that cannot discriminate. R2 removes it.

## Results

Graph gap = adapter − baseline. Control gap = the two runs that should *not* differ.

| rubric | baseline | adapter | **graph gap** | effect size | control gap | control significant? |
|---|---|---|---|---|---|---|
| R1 | 2.83 | 3.21 | +0.387 | 0.29 | 0.140 | no (p=0.18) |
| **R2** | 3.17 | 4.01 | **+0.833** | 0.45 | 0.360 | **yes (p=0.01)** |
| R3 | 2.98 | 3.45 | +0.473 | 0.32 | 0.227 | borderline (p=0.05) |
| R4 | 2.50 | 3.24 | +0.740 | 0.50 | 0.087 | no (p=0.64) |

Share of outputs scoring 5 or better — how much of the scale each rubric actually uses:

| | baseline | adapter |
|---|---|---|
| R1 | 6.7% | 12.0% |
| **R2** | 13.3% | **31.3%** |
| R3 | 6.0% | 21.3% |
| R4 | 2.7% | 16.0% |

**R3 — where the effect lives:**

| | baseline | adapter | gap | relative |
|---|---|---|---|---|
| domain | 7.83 | 8.19 | +0.37 | +4.7% |
| problem | 4.65 | 5.05 | +0.40 | +8.6% |
| task | 3.85 | 4.34 | +0.49 | +12.6% |
| **scope** | 2.64 | 3.16 | **+0.52** | **+19.7%** |

## What this means

**1. The rubric decides roughly half the measured effect.** The same models, the same outputs,
scored +0.387 or +0.833 depending only on how the judge was asked. That is the main result, and
it applies to every accuracy figure in the project.

**2. R2 doubles the gap, and the reason is the honest one.** Dropping methodology lifts both
arms, but the adapter more than twice as much (+0.79 vs +0.35), and adapter outputs scoring ≥5
go from 12% to 31%. This is finding 2 above confirmed: the advantage was real and the old scale
was too compressed to show it.

  **But R2 also spreads apart runs that should be identical** — it is the only rubric whose
  negative control became significant. R2 is a *more sensitive* instrument and a *less specific*
  one. The case for it is that it measures the right thing, not that it discriminates better.

**3. R4 looks best on the numbers and should still be rejected.** Biggest effect size, and the
only rubric whose control gap shrank. But its whole gain comes from pushing the baseline down
(−0.33) while leaving the adapter untouched (+0.03) — and the penalty does **not** land on the
papers where the baseline actually copied a cited work. It is flat across them
(correlation −0.09, not significant). So R4 isn't detecting the failure it names; it is simply
harsher on the baseline. Since R4 was written *from* the baseline's observed failure, that is
the thumb-on-the-scale outcome the design was built to catch. Worth noting the negative control
alone did *not* catch it — the extra check did.

**4. R3 confirms the original hypothesis: task and problem are recoverable, specifics are not.**
Scores fall steadily as the question gets more specific — domain 7.8, problem 4.7, task 3.9,
scope 2.6. Both arms identify the field; neither pins down the contribution. And the adapter's
advantage grows the harder the question gets: +4.7% on domain rising to +19.7% on scope. The
graph tells the model *what kind of paper this is and how broad its claim is* — a survey, not a
single method — which is exactly the failure in finding 3.

## Which rubric to use going forward

**R2.** Score the research problem and the specific task; do not score methodology.

The argument is that methodology is unrecoverable from a bibliography, so scoring it adds noise
that cannot separate any two models. That argument was made before the results and holds
regardless of them. The fact that it doubles the measured graph effect is a consequence, not
the reason — and it should be presented in that order, with the inflated negative control
disclosed alongside.

**Keep R3 as the explanatory figure** — it shows *why* the graph helps, which no single number
can. **Report R4 as checked and rejected**; a variant that was tested and thrown out is a
stronger methodological story than one that was adopted.

Adopting R2 means every accuracy number in `docs/adapter.md` would have to be rescored to match.
Nothing already published becomes wrong — it becomes an R1 number, and R1 and R2 cannot be
mixed in one table.

## Caveats

- **No noise floor was measured.** The same-text-scored-twice cell was dropped to save cost, so
  nothing here bounds the judge's own wobble. The negative control is a weaker substitute: it is
  a *real* effect, not noise — the repetition penalty genuinely scores worse, consistently, in
  all four rubrics — so it measures how much a rubric amplifies an irrelevant difference, not
  how much it invents one.
- **150 papers is thin, and it bites where it matters.** In R3, only domain and scope are
  statistically significant; **problem (p=0.06) and task (p=0.05) are not**. Read the R3 ranking
  as a pattern, not as four established facts.
- **R1's scores are cached** from the original evaluation run rather than re-judged, so R1-vs-
  others carries a small between-run difference the other comparisons don't.
- **The judge always sees the real abstract**, so it can reward looking similar to the gold
  rather than being right about the paper. All four rubrics inherit this; none fixes it.
- **One judge, one pair of runs, one prompt era.** A different generator could rank these
  rubrics differently.
- We also tried `gpt-5.6-luna` as a cheaper judge and rejected it: it separated the two arms
  about 1.5x less well, and cannot run at temperature 0, so it isn't reproducible. Measurement
  kept in `judge_comparison/`.

## Files

```
rubrics.py        the four judge prompts
runs.py           which generation runs are compared, and why those
analyse_arms.py   what the graph changes, no API calls  -> arm_analysis.csv
build_subset.py   the frozen 150-paper set              -> subsets.json
score.py          one rubric x one arm                  -> 0N_RN/<arm>.csv
report.py         the results tables above
```

```bash
conda activate DyTAG
python experiments/rubric_sensitivity/score.py --all     # 1,800 judge calls
python experiments/rubric_sensitivity/report.py
```

R1 costs nothing: it is the project's current rubric word-for-word, scored by the same judge at
the same temperature, so the runs' stored scores already *are* its scores. `score.py` copies
them rather than buying them again. `score.py` is resumable and runs cell by cell with
`--rubric R2 --arm baseline`.
