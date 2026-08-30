# The two prompts behind this round

Version 1 is the 2026-08-20 adapter run, Version 2 the 2026-08-21 run. Decoding is identical
(`MAX_NEW_TOKENS = 320`, `REPETITION_PENALTY = 1.15`) and the reference block is unchanged, so
**the instruction is the only difference** apart from the retrain — a cleaner contrast than
round 2, which moved four things at once.

| | Version 1 | Version 2 |
|---|---|---|
| prompt era | 2 | 3 |
| `src/adapter/text.py` | `4105279` (2026-08-19) | working tree, 2026-08-20 |
| generated | 2026-08-20 | 2026-08-21 |

## Version 1 — era 2

```
You are a scientific writing assistant. The publications listed below are cited by one paper,
whose own abstract is missing. Reconstruct that missing abstract: the problem the paper
addresses, how it builds on and departs from the work it cites, what it proposes, and what it
concludes. Infer from the references what kind of paper it is — a new method, an application,
an empirical study, or a survey — rather than assuming. Write 150-180 words. Output ONLY the
abstract text, without any heading or explanation.
```

## Version 2 — era 3

```
You are a scientific writing assistant. The publications listed below are cited by one paper,
whose own abstract is missing. Reconstruct that missing abstract: the problem the paper
addresses, how it builds on the work it cites, what it proposes, and what it concludes. Do not
name a method, model, system, or dataset that does not appear in the cited works. Where the
sources do not determine the paper's specific contribution, state what they do determine about
its problem and approach, and stop there — a correct general statement is worth more than a
specific guess. Write in the voice of the paper's own authors: never mention the cited works
as a collection, how many there are, or "the authors". Write 150-180 words. Output ONLY the
abstract text, without any heading or explanation.
```

Both are wrapped identically:

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives
helpful, detailed, and polite answers to the user's questions. USER: {INSTRUCTION}

This paper cites {total} works; the {shown} most representative are listed below:
- Title: {title}
  Abstract: {abstract}
  ... up to 15 references, chosen by distance to the neighbourhood centroid

Abstract: ASSISTANT:
```

## The four changes the judge is asked to attribute

1. **Genre menu removed** — `Infer from the references what kind of paper it is — a new
   method, an application, an empirical study, or a survey — rather than assuming.` is gone.
   Round 2 found that enumerating the kinds made genre a coin flip the model lost in both
   directions: a retrieval-augmented NMT method paper written as a survey, a genuine survey of
   Reed-Muller results written as *"We present a new construction"*.
2. **Invention forbidden** — `and departs from` dropped, and an explicit ban added on naming a
   method, model, system or dataset absent from the cited works. Round 2 traced invented
   specifics (TextBoxes++, M3SDA, *"a novel meta-learning algorithm called MRL"*) to the
   *departs from* / *what it proposes* framing.
3. **Hedging rule** — the calibration asymmetry stated outright. Under the accuracy rubric a
   correct general statement about the shared problem scores ~6 while a confident statement
   about the wrong method scores ~2-3, so a right specific guess is worth about +2 and a wrong
   one about −4. Era 1's vagueness was protective by accident; era 3 makes it deliberate.
4. **Authorial voice** — never mention the cited works as a collection, their number, or
   "the authors". This one targets the **baseline**, not the adapter: disclosing the reference
   count in era 2 made the no-graph arm describe the bibliography instead of the paper
   (third-person "the authors" went 145 → 540 of 945, and 15 outputs reviewed the reference
   list outright), which would inflate the adapter's margin with an artifact.

## What the scores already say

Mean accuracy 3.259 → 3.214 over 945 eval nodes; on the 48 nodes judged here, cell A
(the collapse this was written to undo) recovers +0.73 while cell C gives back −1.00.

One check the judge is asked to make directly, via the `INVENTED` field: change 2 does not
appear to have taken. Across all 945 outputs the phrase family *novel / new
method|approach|framework* occurs 329 times in era 2 and 313 in era 3, despite the explicit
ban — while overall bigram overlap between the two runs is only 0.260, so the text changed a
great deal without the invention behaviour changing.

The judge's own prompt is in `compare_abstracts.py`.
