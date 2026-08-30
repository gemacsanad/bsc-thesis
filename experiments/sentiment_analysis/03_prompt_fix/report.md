# Why the third prompt did not help, and what the fourth should be

Diagnosis of the 2026-08-21 adapter run against the 2026-08-20 run, on the same 48 nodes round 2
judged. Judge: gpt-5.4-mini, blind to the accuracy scores and to the four sampling cells. Unlike
round 2, the two runs differ **only in the instruction** (same decoding, same reserve, same
reference block) plus a retrain, so attribution is much cleaner here. Source data:
`comparison_dataset.csv`; prompts in `prompt.md`; supporting measurements over all 945 eval nodes
in `ref_overlap_all945.csv`.

## Summary

The fix did what it was designed to do at the level of wording, and the blind judge rewards it —
it picks the new version 28 times to 16. The accuracy score moved the other way, 3.259 → 3.214.
Both are correct, and the gap between them is the finding: **the rewrite improved how the
abstracts are written without improving which paper they are about.**

Three prompt eras have now produced 3.288, 3.259, 3.214. Each rewrite changed 70-90% of the
generated text. None moved the mean by more than judge noise. Prompt engineering is not the
lever that gets this system from 3.2 to 6-7, and the rest of this report argues that the
remaining prompt work should target two *measurable artifacts* rather than the mean.

## 1. The blind judge and the scoring judge disagree, and both are right

| blind verdict | n | mean score change (v2→v3) |
|---|---|---|
| new version closer | 28 | +0.14 |
| old version closer | 16 | −0.69 |
| equivalent | 4 | +0.50 |

The blind judge tracks the score in direction but not in magnitude: the nodes it calls losses
really did lose (−0.69), while the nodes it calls wins mostly did not move (+0.14, median 0.00).
Counting node by node, the two agree on 17, disagree on 14, and 17 are ties.

The reason is visible in the two new fields. `SPECIFICITY` says the new run commits *less* hard
to a specific contribution on 23 nodes and *more* on 18 — it hedges, as instructed. The blind
judge, asked which text is closer to gold, rewards a well-formed cautious abstract. The accuracy
rubric does not: it rewards saying what the paper is about, and a fluent hedge about the right
neighbourhood is worth the same 3 as a fluent hedge about the wrong one.

**Implication for prompt search: do not tune on a preference judge.** A pairwise "which is
closer" judge and the accuracy rubric optimise different things, and this round is a concrete
case where they point in opposite directions.

## 2. The dominant failure is not invention — it is writing a plausible neighbour paper

Measured over all 945 eval nodes, bigram overlap between the generated abstract and:

| | v2 (08-20) | v3 (08-21) |
|---|---|---|
| the **gold** abstract | 0.027 | 0.027 |
| the **most similar cited reference** | 0.144 | 0.146 |

The output is consistently about **5x closer to one of the papers the target cites than to the
target itself**, and the prompt rewrite did not change this at all. On 47 of the 48 sampled
nodes, the generation resembles a reference more than it resembles the gold.

This is the structural failure underneath every symptom below. The model is doing topic
modelling on the bibliography and writing a generic, competent paper from that topic. Node 99111
is the clearest case: the gold is MetaGenRL, a meta-RL algorithm; the generation is a
well-written abstract about model distillation for interpretability. Same neighbourhood, wrong
paper, score 2.

## 3. About 10% of outputs are near-verbatim copies of a reference — and they inflate the top tier

Pushing that measurement further, outputs with >0.5 overlap against a single cited abstract are
not paraphrases; they are regurgitations.

| | v2 | v3 |
|---|---|---|
| near-verbatim copies of a cited reference (>0.5) | 98 / 945 (10.4%) | 98 / 945 (10.4%) |
| their mean accuracy | 3.42 | 3.31 |
| share of the top tier (accuracy ≥ 6) that are copies | 14 / 59 (**24%**) | 13 / 68 (**19%**) |

Node 5164 reproduces a cited abstract at overlap 1.000 — the jigsaw-puzzle self-supervision paper,
copied whole, while the gold is SimCLR; it scores 2. Node 105709 copies a reference at 0.826 and
scores **8**, because that reference happens to be the target's direct predecessor. Nodes 149271
(0.933 → 7), 145477 (0.717 → 7) and 12469 (0.443 → 8) are the same phenomenon.

This matters beyond quality: **roughly a fifth of the high scores are not reconstruction at all.**
Any claim about the ceiling — including round 2's "the top collapsed by −2.00" — is partly a claim
about how often the model copied a well-placed reference. It is stable at 10.4% across both runs,
so it does not bias the *comparison* between them, but it does inflate the absolute numbers.

Copying is rare at low out-degree (1.7% at 5-10 references) and common above that (~10-15%),
consistent with the centroid cap sometimes selecting one dominant reference.

## 4. The ban on inventing names did not work — the rate went up

The era-3 instruction says *"Do not name a method, model, system, or dataset that does not appear
in the cited works."* Rate of outputs that coin a name (`called/named/termed X`):

| | era 1 (08-15) | era 2 (08-20) | era 3 (08-21) |
|---|---|---|---|
| coins a name | 10.7% | 33.1% | **35.2%** |
| acronym in parentheses | 48.0% | 53.5% | 52.2% |
| "novel/new method\|approach\|framework" | 38.0% | 40.3% | 38.6% |
| "we propose/present/introduce" | 83.9% | 82.5% | 81.3% |

The prohibition raised the rate slightly. This is the single most important negative result in
this round: **a negative instruction did not suppress the behaviour it named.** Mentioning
"method, model, system, or dataset" appears to have primed the slot rather than closed it.

The cost is real and it is the largest single-marker effect measured:

| marker (v3, n=945) | n | accuracy with | without | diff |
|---|---|---|---|---|
| coins a name | 333 | 2.93 | 3.37 | **−0.43** |
| truncated | 24 | 2.38 | 3.24 | −0.86 |
| LaTeX artifacts (`\emph`, `$...$`) | 112 | 3.00 | 3.24 | −0.24 |
| "Recent advances…" opener | 148 | 3.01 | 3.25 | −0.24 |
| acronym in parentheses | 493 | 3.18 | 3.25 | −0.07 |

On the 48 judged nodes the same effect is much larger, because the judge identifies invention
rather than a regex: when the new run invents, it averages **2.77**; when it does not, **4.42**.
The eight nodes where neither version invents average 5.38 in both runs.

Inventions observed: DrugNet, DeepFakeDetector (DFD), Geometric Feature Transformer (GFT), TIRE,
IQCNN, Diverse Image Synthesis Network (DISN), Semantic Convolutional Network (SCN), Face
Annotation Tool (FAT), BTSUL, Precision-aware Training (PAT), Graph Feature Maps (GFMs), Score
Regularization (SR), TEXTBOXES.

**Hedging is free.** 31.5% of gold abstracts name their own system, and their mean accuracy
(3.20) is identical to those that do not (3.22). The rubric does not reward getting the name
right, so there is no downside to never naming anything — only the upside of removing a −0.43
marker.

## 5. Removing the genre menu was aimed at the wrong cause

Round 2 blamed the genre enumeration for the ceiling collapse. Era 3 removed it. The blind judge
cites cause (a) — changed genre guess — on only **3 of 48** nodes, and genre errors persist
unchanged: node 57045's gold is a survey of Reed-Muller code results and the generation still
opens *"We present a new construction"*; node 2508's gold is a GAN survey organised into six
"fronts" and the generation still proposes a training procedure; node 145477's gold is a thesis
and the generation is a WMT shared-task system description.

The menu was not the cause. The cause is the *"we propose"* prior, which sits at 81-84% in all
three eras and has not responded to any instruction yet. Removing the menu removed the model's
only invitation to consider a non-method genre without removing its default.

Cause counts across the 48: (c) hedging rule 34, (b) invention 25, (a) genre 3, (d) voice 2.

## 6. What did work: the authorial-voice clause — with a consequence for the GNN result

Clause (d) was aimed at the **baseline**, not the adapter, and it is the only change that took
cleanly. Rate of "the authors / these works / the papers / the cited":

| | era 1 | era 2 | era 3 |
|---|---|---|---|
| adapter arm | 0.3% | 0.1% | **0.0%** |
| baseline arm | 15.4% | **58.1%** | **40.0%** |

Era 2's reference-count disclosure sent the no-graph baseline from 15% to 58% "describing the
bibliography", and era 3 pulled it back to 40% — a real repair, but only partial. The measured
graph effect follows it:

| run | baseline | adapter | gap (95% CI) |
|---|---|---|---|
| 08-15 | 2.815 | 3.288 | **+0.473** ± 0.098 |
| 08-20 | 2.829 | 3.259 | **+0.431** ± 0.095 |
| 08-21 | 2.871 | 3.214 | **+0.343** ± 0.094 |

The gap is shrinking because the baseline is being repaired, not because the adapter got worse.
Some of the +0.473 was an artifact: era 2 handicapped the no-graph arm with an instruction the
adapter could ignore. The gap remains clearly non-zero, but **40% of baseline outputs still carry
the artifact**, so +0.343 is still an upper bound. The honest figure will only be known once the
baseline arm is clean.

This is worth stating plainly in the thesis: a prompt change that lowers the headline gap can be
the more honest measurement.

## 7. What the good outputs look like

Among the 55 outputs scoring ≥6 that are *not* reference copies, versus the 259 scoring ≤2 that
are not copies:

| | acc ≥ 6 | acc ≤ 2 |
|---|---|---|
| coins a name | 18% | 35% |
| mean words | 127 | 125 |

Length is not the variable; naming is. Node 12469 scores 8 with four plain sentences —
*"we study the problem of using machine-learned predictions to improve the performance of online
algorithms. We consider the classic ski-rental problem…"* — no invented system, no acronym, no
claim it cannot support. That is exactly the behaviour clause (c) asked for. It fires rarely.

Accuracy also rises with bibliography size (2.83 at 5-10 references, 3.14 at 10-20, 3.20 at
20-40, 3.27 at 40-100), and the baseline rises with it in parallel, so this is prompt information
rather than graph information.

## 8. What the fourth prompt should do

The judge's `FIX` column converges: 24 of 48 answers ask for less invention, 14 for staying
general. But era 3 already asked for both in plain words and the invention rate rose. The design
principle for era 4 therefore has to change:

> **Stop prohibiting behaviours. Give the model a shape with no slot for them.**

Concretely, in descending order of expected value:

1. **Replace the ban with a structural template.** Instead of *"Do not name a method, model,
   system, or dataset…"*, prescribe the sentence plan: one sentence on the problem, one on what
   prior work established, two on what this paper does *described by its mechanism rather than its
   name*, one on what it evaluates. A template with no naming slot should beat a prohibition that
   mentions the slot. Expected value: the −0.43 marker applies to 35% of outputs, so eliminating
   it is worth roughly +0.15 on the mean — the largest single lever identified.
2. **Attack the copying directly.** Nothing in any era tells the model that the paper it is
   writing is *not* one of the papers shown. Add it: *"The paper you are reconstructing is not
   among those listed; it is a later work that cites all of them. Do not restate any single one of
   them."* 10.4% of outputs and ~20% of the top tier are at stake, though note this will *lower*
   the headline mean by removing lucky copies — it buys validity, not score.
3. **Give the genre a data-grounded default rather than a menu or nothing.** The reference count
   is already in the prompt. Use it: *"A paper citing more than ~50 works with no shared method is
   usually a survey; otherwise assume it contributes one method or study."* Era 2 offered a menu
   (coin flip), era 3 offered nothing (unchanged "we propose" prior). Neither has tested a prior.
4. **Suppress LaTeX and the template opener.** `\emph{}` and `$...$` are at 11.9% and rising, and
   "Recent advances…" at 15.7%; each costs −0.24. These are cheap formatting wins with no
   accuracy risk.
5. **Keep** the reserve, the repetition penalty, the reconstruction framing, and the voice clause
   — and **extend the voice clause until the baseline artifact reaches 0%**, since the measured
   GNN effect is not trustworthy until it does.

### How to test it without another retrain

The prefix and the instruction interact weakly: the era-2 → era-3 rewrite changed 74% of bigrams
for a −0.045 change in mean accuracy. That bounds the interaction well enough to justify
**generation-only sweeps on the existing 08-21 checkpoint** — no retraining, one GPU pass per
variant.

Design, from the variance measured here (σ of the per-node change ≈ 1.16):

- ~150 nodes per variant to detect a 0.3-point difference at 80% power; 55 for 0.5.
- Sweep on **validation** nodes, confirm the winner once on the eval set. At n=10 the minimum
  detectable difference is 1.17 and picking the best of 6 variants inflates the winner by ~0.5
  points of luck.
- Score with the **accuracy rubric**, not a pairwise preference judge — see §1.
- Tune on the **baseline arm**. A prompt tuned on the adapter flatters the graph; a prompt tuned
  to make the no-graph baseline as strong as possible leaves a residual gap that is attributable.

## Limitations

- 48 nodes across four cells; per-cell n is 10-15. Marker rates and the copy analysis are over all
  945 and are the more reliable numbers here.
- The cells were defined on the 08-15/08-20 scores. For this round that is a strength — the sample
  was frozen before the 08-21 run existed — but the cells are still extreme groups, not a random
  sample, so cell means overstate movement through regression to the mean.
- The two runs differ by a retrain as well as the instruction, so a prompt-only regeneration would
  still be needed to attribute cleanly. This is the sweep proposed above.
- Bigram overlap is a crude copy detector. It will miss paraphrased regurgitation, so 10.4% is a
  floor, not an estimate.
- `INVENTED` and `SPECIFICITY` are the judge's readings, not measurements; the regex marker rates
  over 945 nodes are what the quantitative claims rest on.
