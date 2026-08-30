# What the new prompt did, and what to change next

Diagnosis of the 2026-08-15 adapter run (old prompt) against the 2026-08-20 adapter run (new
prompt + reference count + repetition penalty + larger reserve + retrain), on 48 nodes sampled
from four contrast cells. Judge: gpt-5.4-mini, blind to the accuracy scores and to which run is
which, told only what the four configuration changes were. Source data:
`comparison_dataset.csv`.

## Why this comparison was needed

Mean accuracy barely moved (3.288 → 3.259), but that number is the sum of two large opposite
effects:

| old tier | n | mean change | contribution to the overall mean |
|---|---|---|---|
| 1-2 (floor) | 330 | **+0.81** | +0.281 |
| 3-4 (middle) | 486 | −0.07 | −0.037 |
| 5-8 (top) | 129 | **−2.00** | −0.273 |

A judge re-run on identical text (`..._copy/`) gives per-node r = 0.933, so this is not
measurement noise: two runs of the same system agree almost perfectly, while these two agree at
r = 0.405. The change is real and it is redistributive — it raised the floor and destroyed the
ceiling by almost exactly the same amount.

## The finding

The blind judge reproduces the score movement without having seen it. On the 15 collapsed-top
nodes it picked Version 1 as closer to gold **15 times out of 15**.

| cell | n | V1 closer | V2 closer | genre error | invented specifics |
|---|---|---|---|---|---|
| A collapsed top (6.6 → 2.8) | 15 | **15** | 0 | 14 | 10 |
| B survived top (6.5 → 6.9) | 10 | 3 | 7 | 4 | 3 |
| C rescued floor (2.0 → 4.1) | 12 | 3 | 9 | 9 | 7 |
| D stuck floor (1.6 → 1.9) | 11 | **10** | 1 | 7 | 9 |

Cause (a), the instruction reframe, is cited in 14/15 of the collapses — and also in 11/12 of the
rescues. It is the lever in both directions. Causes (b) bibliography size and (c) repetition
penalty are cited rarely (3-5 and 1-4 per cell); neither is the culprit, which rules out two of
the three hypotheses we started with.

### Mechanism 1 — genre became a free variable, and it is guessed wrong in both directions

The new instruction says *"Infer from the references what kind of paper it is — a new method, an
application, an empirical study, or a survey — rather than assuming."* That converted a fixed
default into a decision the model now makes, and gets wrong:

- 133035 (8→3): a retrieval-augmented NMT method paper became *"comprehensive investigation of
  various integration strategies"* — a survey framing.
- 45854 (6→3): a drug-response prediction model became *"we review recent progress"*.
- 129587 (7→3): the RUBi *learning strategy* became *"introducing a new dataset called VQA-CPv2"*.
- 57045 (6→3): the reverse direction — a genuine survey of Reed-Muller code results became
  *"We present a new construction"*.
- 99111 (7→3): drifted out of the paper genre entirely — *"In this thesis, I focus…"*.

Enumerating the genres invited the model to pick one, and a wrong pick reframes the whole
abstract.

### Mechanism 2 — asking what the paper "proposes" and "departs from" produces invented specifics

The new instruction asks how the paper *"builds on and departs from"* its sources and *"what it
proposes"*. The model answers by naming a concrete contribution that does not exist: TextBoxes++
(67595), M3SDA and DomainNet (87254), a CNN+LSTM+SVM pipeline where the gold is a capsule network
(133492), *"a novel meta-learning algorithm called MRL"* (99111).

### Why these two are so expensive: the old prompt's vagueness was protective

*"A new application paper that builds on and combines these works"* was a fixed and often wrong
framing, but it kept the output at the level of the **shared topic of the bibliography** — which
is exactly what the accuracy rubric rewards. A vague statement about the right problem scores 6;
a confident statement about the wrong method scores 2-3.

The new prompt trades vagueness for specificity. When the specific guess is right, that is worth
about +2 (cell C). When it is wrong, it costs about −4 (cell A). The old prompt could not score
the wins, but it could not take the losses either.

## What genuinely helped, and must be kept

Cause (d), the larger generation reserve, is cited in 11/12 rescues and 8/10 survivals, and
several gains are literally the presence of an ending: node 14379 gained *"Our experimental
results indicate that the proposed approach outperforms the baseline methods…"* where Version 1
simply stopped mid-sentence. This is consistent with the run-level counts — truncations fell from
238 to 11 and duplicated sentences from 84 to 3. It has no observed downside and is independent
of the risky clauses.

## The floor is not bounded by the task

`RECOVERABLE` came back **YES for 45 of 48** nodes, including 8 of the 11 that are stuck at 1-2.
The papers in the stuck group are inferable from their bibliographies; the model is failing on
them, not being asked the impossible. A 4-5 floor is therefore a model/prompt target, not a task
limit.

One caution on that group: the blind judge preferred Version 1 on 10 of 11 stuck nodes and
answered `GAINED: nothing` on 7 — a stronger regression than the scores (1.64 → 1.91) suggest.
The bottom of the distribution got worse in ways the 1-2 band cannot express.

## What to change for the next run

The `FIX` field converges on nearly the same sentence across all four cells: keep reconstruction
framing, stop inventing specifics, stop defaulting to a genre.

1. **Remove the genre enumeration.** Do not list method / application / empirical study / survey.
   If genre must be inferred, condition it on evidence (reference count is already in the prompt)
   rather than offering a menu.
2. **Forbid unsupported specifics.** Add an explicit instruction not to name a method, model,
   system, or dataset that does not appear in the cited works. This is the single most repeated
   recommendation in the `FIX` column.
3. **State the calibration rule directly.** Prefer a correct general statement about the shared
   problem over a specific guess. Under this rubric, hedging beats confident error — that is what
   the +2 / −4 asymmetry means, and the model should be told it.
4. **Keep** the larger reserve, the repetition penalty, and the reconstruction framing itself.
   None of the three is implicated in the collapses.

Expected effect if the ceiling loss is removed while the floor gain is kept: roughly **+0.28 on
mean accuracy**, about twice the 0.135 detection threshold at n = 945.

## Limitations

- The two runs differ by four configuration changes *and* a retrain, so attribution rests on the
  judge's reading of textual fingerprints, not on a controlled ablation. Only a prompt-only
  regeneration would settle it.
- 48 nodes across four cells; per-cell n is 10-15.
- Cells were defined using the same scores the analysis then explains, so cell membership is not
  independent of the outcome. The blind judge's 15/15 agreement on cell A is the main protection
  against circularity, since it never saw the scores.
- `RECOVERABLE` is the judge's opinion about inferability, not a measurement.
