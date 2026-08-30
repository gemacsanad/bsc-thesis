# Era 2 → era 3: what changed, what it produced, and what holds up

Companion to `report.md`, written 2026-08-22 from the full 945-node eval sets rather than the
48-node sample. `report.md` is left as written. Unlike the era 1 → 2 transition, this one is
clean: decoding, reserve and reference block are identical, so **the instruction is the only
difference** apart from the retrain.

## 1. What changed

Four clauses, all wording.

| | era 2 | era 3 |
|---|---|---|
| genre | menu — *"a new method, an application, an empirical study, or a survey"* | clause removed entirely |
| invention | *"how it builds on **and departs from** the work it cites, what it proposes"* | *"Do not name a method, model, system, or dataset that does not appear in the cited works"* |
| calibration | — | *"a correct general statement is worth more than a specific guess"* |
| voice | — | *"never mention the cited works as a collection, how many there are, or 'the authors'"* |
| `MAX_NEW_TOKENS` / `REPETITION_PENALTY` | 320 / 1.15 | **unchanged** |

## 2. What it produced

Full eval set, n = 945:

| | era 2 | era 3 |
|---|---|---|
| mean accuracy (adapter) | 3.259 | 3.214 |
| sd | 1.250 | 1.287 |
| ≤ 2 | 29.3% | 31.6% |
| ≥ 6 | 6.2% | 7.2% |
| coins a name (`called/named/termed X`) | 21.5% | **33.9%** |
| third-person, **baseline** arm | 58.8% | **40.5%** |
| third-person, adapter arm | 0.2% | 0.1% |
| overlap with nearest cited reference | 0.144 | 0.146 |
| overlap with the gold | 0.027 | 0.027 |
| GNN gap | +0.431 | **+0.343** |

Mean accuracy did not move (paired p = 0.229). Since decoding was untouched, the whole −0.045
is attributable to the wording — there is no engineering gain hiding underneath it as there was
in era 2.

Two things did move, in opposite directions: the clause that banned naming was followed by the
naming rate rising by half again, and the clause aimed at the baseline arm cut its
bibliography-describing artifact by a third.

## 3. Interpretation — the lever is real, the prohibition is not the way to pull it

`report.md` §4 reports that the ban failed and estimates that eliminating the behaviour would
still be worth about +0.15. Both halves check out, and they are not in tension.

The behaviour is genuinely costly, and stopping it genuinely helps. Comparing nodes that stopped
coining a name between era 2 and era 3 against nodes that kept coining, **stratified by era-2
score** so regression to the mean affects both groups equally:

| era-2 score | n stopped | Δ stopped | n kept | Δ kept | within-stratum DiD |
|---|---|---|---|---|---|
| 2 | 21 | +0.952 | 52 | +0.423 | +0.529 |
| 3 | 18 | +0.278 | 39 | +0.026 | +0.252 |
| 4 | 12 | −0.167 | 33 | −0.455 | +0.288 |
| 5 | 6 | −0.500 | 5 | −1.600 | +1.100 |

Weighted DiD = **+0.451** per node (raw, uncontrolled: +0.361, p = 0.039), positive in every
stratum. At a 33.9% rate that is **+0.153 on the mean** — independently reproducing the report's
+0.15 estimate.

So the picture is:

- Coining a name **causes** roughly −0.45 on the node where it happens.
- Instructing the model not to do it **increased** the rate from 21.5% to 33.9%.

The lever exists and the instruction pulled it backwards. This is the strongest evidence in the
project for the report's own design principle for era 4 — *stop prohibiting behaviours, give the
model a shape with no slot for them* — and it is why era 4 replaced the ban with *"described by
the mechanism it uses … rather than by a name"* instead of dropping the idea.

One caveat: stopping was not randomly assigned. Nodes where the model happens not to coin a name
may be nodes where the bibliography gave a clearer signal. The stratification controls for the
era-2 score level, not for that. The estimate is strongly suggestive, not proven causal.

**Note on the marker.** This is the narrow `called/named/termed X` construction. A broader marker
— any acronym or CamelCase token absent from both the references and the gold, 34.6% of era-3
outputs — shows **no** effect when the same DiD is applied (+0.009, p = 0.95). The two markers
have similar rates but catch different nodes, and only the explicit act of christening something
carries the penalty. Using unfamiliar technical vocabulary is free; announcing a named
contribution is not.

## 4. What `report.md` says, and what survives

**Holds up.**

- **§4, the naming lever.** Confirmed above, including the magnitude, by an independent method.
- **§2, the structural failure.** Overlap with the nearest cited reference is 5× the overlap
  with the gold (0.146 vs 0.027), unchanged by the rewrite. Over all 945 nodes, 95.1% of
  generations resemble a cited reference more than they resemble the gold.
- **§3, copying.** 10.4% of outputs exceed 0.5 overlap with a single reference, stable across
  runs. Their mean accuracy is *higher* than non-copies (3.31 vs 3.20), and the ≥ 6 band has the
  highest reference overlap of any score band (0.226 vs 0.112–0.169). Part of the top tier is
  retrieval, not reconstruction. This is a validity problem, not an accuracy problem.
- **§6, the voice clause.** The only clause that took cleanly, and the baseline-arm numbers
  reproduce (15.6% → 58.8% → 40.5%). The measured GNN gap tracks it downward: +0.473 → +0.431 →
  +0.343, and era 1 vs era 3 is significant (−0.130, p = 0.025). The report's reading is right —
  the gap is shrinking because the control is being repaired, so +0.343 is the more honest
  number and still an upper bound while 40% of baseline outputs carry the artifact.
- **§1, do not tune on a preference judge.** Supported: the blind judge preferred era 3 by 28–16
  while the accuracy score fell.

**Does not hold up — the cell/tier analysis, as before.**

Tiers defined on era-2 scores move exactly as regression to the mean predicts (r(e2,e3) = 0.581):

| era-2 tier | n | observed Δ | RTM prediction | residual |
|---|---|---|---|---|
| 1–2 | 277 | +0.50 | +0.50 | −0.00 |
| 3–4 | 566 | −0.11 | −0.13 | +0.02 |
| 5–8 | 102 | −1.18 | −1.09 | −0.08 |

Weighted residual **−0.000**, the same result as for era 1 → 2. Cell-level movements
(*"cell A recovers +0.73 while cell C gives back −1.00"*) carry no information beyond the
correlation between runs.

This matters much less here than it did in round 2. `report.md` states the problem in its own
limitation 2 (*"cell means overstate movement through regression to the mean"*) and, unlike
round 2, rests its quantitative claims on the 945-node marker rates rather than the cells. Every
load-bearing finding above survives.

## 5. Consequences

- Era 3 was designed to reverse round 2's tier redistribution, which section 4 of
  `era1_to_era2.md` shows was never real. That it changed nothing is therefore expected, and is
  not evidence about the clauses it introduced.
- **Three eras, no mean movement**: 3.288 / 3.259 / 3.214, no pairwise difference significant.
  Each rewrite changed 70–90% of the generated text. The report's conclusion — prompt
  engineering is not the lever that reaches 6–7 — stands.
- But it is not nothing either. Two levers with measured magnitudes remain inside the prompt:
  the naming behaviour (+0.15 if eliminated) and the decoding-style markers (LaTeX 12.0% at
  −0.25, stock opener 8.9% at −0.20, truncation 2.5% at −0.86; ~+0.07 combined, and free,
  since they are post-processing). Together ~+0.22 — roughly three times the total spread
  across all three prompt eras.
- The GNN gap is the number degrading fastest, and for a defensible reason. Finishing the
  baseline repair matters more for the thesis than another +0.05 of accuracy.

## Open questions

- Does the era-4 template actually suppress coining, where the ban raised it? That is the single
  measurable prediction era 4 makes, and it is a generation-only test on the existing checkpoint.
- Why does explicit christening cost −0.45 while unfamiliar vocabulary costs nothing? The
  hypothesis worth testing is that a named contribution forces a commitment the rubric then
  scores as a specific error, whereas vocabulary alone does not commit.
- What is the accuracy rubric's floor? Scoring mismatched pairs (generation for node A against
  gold for node B, ~50 calls) would say how much of 3.214 is signal, and would make the +0.343
  gap far easier to defend.
