# Era 1 → era 2: what changed, what it produced, and what holds up

Companion to `report.md`, written 2026-08-22 from the full 945-node eval sets rather than the
48-node sample. `report.md` is left as written; section 4 below corrects one of its findings.

## 1. What changed

Four things moved at once between the 2026-08-15 and 2026-08-20 adapter runs, plus a retrain.

| | era 1 | era 2 |
|---|---|---|
| framing | *"write a concise abstract for a **new application paper** that builds on and **combines** these works"* | *"The publications listed below are cited by one paper, whose own abstract is missing. **Reconstruct** that missing abstract"* |
| genre | fixed to "application paper" | open menu — *"a new method, an application, an empirical study, or a survey"* |
| bibliography size | not disclosed | *"This paper cites {total} works; the {shown} most representative are listed below"* |
| length target | "concise" | 150–180 words |
| `MAX_NEW_TOKENS` | 256 | 320 |
| `REPETITION_PENALTY` | none | 1.15 |

The first four are **wording**; the last two are **decoding**. That split turns out to be the
one that matters.

## 2. What it produced

Full eval set, n = 945, adapter arm:

| | era 1 | era 2 |
|---|---|---|
| mean accuracy | 3.288 | 3.259 |
| sd | 1.456 | **1.250** |
| ≤ 2 | 34.9% | 29.3% |
| ≥ 6 | 10.8% | **6.2%** |
| truncated mid-sentence | **25.2%** (238) | 1.2% (11) |
| GNN gap vs no-graph baseline | +0.473 | +0.431 |

Mean accuracy did not move (paired t-test p = 0.555). Two things did move: the distribution
**narrowed**, and truncation was essentially eliminated.

## 3. Interpretation — the flat mean is two effects cancelling

Truncation is an independent marker, so its value can be estimated without conditioning on the
score. Comparing nodes truncated in era 1 but complete in era 2 against nodes complete in both,
**stratified by era-1 score** so that regression to the mean affects both groups equally:

| era-1 score | n fixed | Δ fixed | n never | Δ never | within-stratum DiD |
|---|---|---|---|---|---|
| 1 | 14 | +1.286 | 22 | +0.955 | +0.331 |
| 2 | 97 | +1.041 | 191 | +0.649 | +0.392 |
| 3 | 51 | +0.667 | 207 | +0.145 | +0.522 |
| 4 | 43 | +0.116 | 182 | −0.549 | +0.666 |

Weighted DiD = **+0.434** per affected node (raw, uncontrolled: +0.620; p < 0.0001). Since
25.2% of era-1 outputs were truncated, the decoding fix was worth **+0.109 on the mean**.

The accounting then closes:

| | effect on mean |
|---|---|
| decoding fix (320 tokens + repetition penalty) | **+0.109** |
| era-2 wording rewrite (residual) | **−0.138** |
| net observed (3.288 → 3.259) | −0.029 |

**Era 2 was an engineering fix worth +0.11 cancelled by a prompt rewrite that cost −0.14.** The
flat mean concealed both. This also implies an untested configuration — era-1 wording with era-2
decoding — worth an estimated **3.397**, better than any of the three eras actually run. It
requires a generation pass only, not a retrain, since the 08-15 checkpoint was trained under
era-1 wording. Assumes additivity between wording and budget, which is plausible but unverified.

## 4. What `report.md` says, and what survives

**Holds up.**

- *Cause (d), the larger generation reserve, "has no observed downside" and is cited in 11/12
  rescues.* Confirmed and now quantified: +0.434 per affected node. This is the largest effect
  measured anywhere in the prompt work.
- *The genre menu makes genre a free variable that is guessed wrong in both directions*
  (133035, 45854, 129587, 57045, 99111) and *"departs from" / "what it proposes" produces
  invented specifics* (TextBoxes++, M3SDA, "a novel meta-learning algorithm called MRL"). These
  are direct observations of text and they are real behaviours.
- *The old prompt's vagueness was protective.* Consistent with the sd falling 1.456 → 1.250 in
  the opposite direction from what specificity should do, and with the −0.138 wording residual.

**Does not hold up — the tier redistribution.**

`report.md`'s headline table reports the change as redistributive: floor +0.81, middle −0.07,
top −2.00. Those tiers are defined on era-1 scores, and the two runs correlate at only
r = 0.405. Conditioning on a noisy first measurement mechanically produces exactly that pattern.
Predicted from r and the marginal distributions alone, against observed:

| era-1 tier | n | observed Δ | predicted by regression to the mean | residual |
|---|---|---|---|---|
| 1–2 floor | 330 | +0.81 | +0.88 | −0.08 |
| 3–4 middle | 486 | −0.07 | −0.15 | +0.07 |
| 5–8 top | 129 | −2.00 | −1.92 | −0.08 |

Weighted residual: **+0.000**. The redistribution is what two runs correlating at 0.405 look
like when you sort by one of them. It is not evidence that the prompt raised the floor or
destroyed the ceiling.

The report's two defences do not rescue it. The `..._copy/` control (r = 0.933 on identical
text) bounds *judge* noise, not *generation* variance — and the gap between 0.933 and 0.405 is
precisely the generation variance that drives the artifact. The blind judge's 15/15 on cell A
does not help either: those nodes were selected because version 2 scored far lower on them, so
the text really is worse there; blindness prevents the judge from knowing which run is which,
but it cannot undo selection on the outcome. `report.md`'s own limitation 3 half-anticipates
this.

What is real, and produces the same appearance, is the **compression**: sd 1.456 → 1.250, top
tier 10.8% → 6.2%, bottom 34.9% → 29.3%. Era 2 made outputs more uniform. That is a genuine
finding; "it rescued the floor at the cost of the ceiling" is not a further one on top of it.

## 5. Consequences

- `report.md`'s prediction — *"roughly +0.28 on mean accuracy"* if the ceiling loss were removed
  and the floor gain kept — did not materialise. Era 3 implemented all three of its
  recommendations and scored 3.214, i.e. −0.045. That follows directly from section 4: era 3 was
  built to reverse a redistribution that was never there.
- **Do not define analysis cells by conditioning on the metric being explained.** With r = 0.405
  between runs, any such split will manufacture a large, symmetric, entirely spurious effect.
  Use an independent marker (as with truncation here) or a within-stratum comparison.
- The one clearly banked gain of era 2 is decoding, not wording. Everything the wording bought
  is negative or indistinguishable from zero.

## Open questions

- Is era-1 wording + era-2 decoding actually 3.40? One generation pass on the 08-15 checkpoint
  settles it, if that checkpoint still exists on Komondor.
- Which half of the decoding fix did the work — the token budget or the repetition penalty? The
  DiD bundles them.
- The +0.434 estimate assumes the era-2 wording affected truncated and untruncated nodes alike.
  A prompt-only regeneration would remove that assumption.
