# The two prompts behind this round

Version 1 is the 2026-08-15 adapter run, Version 2 the 2026-08-20 run. Both are the adapter;
what changed is the instruction, the decoding, and a retrain.

| | Version 1 | Version 2 |
|---|---|---|
| prompt era | 1 | 2 |
| `src/adapter/text.py` | `3ae4c3b` | `4105279` (2026-08-19) |
| generated | 2026-08-16 | 2026-08-20 |
| `MAX_NEW_TOKENS` | 256 | 320 |
| `REPETITION_PENALTY` | none | 1.15 |

## Version 1 — era 1

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives
helpful, detailed, and polite answers to the user's questions. USER: You are a scientific
writing assistant. Based on the source publications listed below, write a concise abstract
for a new application paper that builds on and combines these works. Output ONLY the abstract
text, without any heading or explanation.

Source publications:
- Title: {title}
  Abstract: {abstract}
  ... up to 15 references, chosen by distance to the neighbourhood centroid

Generated abstract: ASSISTANT:
```

## Version 2 — era 2

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives
helpful, detailed, and polite answers to the user's questions. USER: You are a scientific
writing assistant. The publications listed below are cited by one paper, whose own abstract
is missing. Reconstruct that missing abstract: the problem the paper addresses, how it builds
on and departs from the work it cites, what it proposes, and what it concludes. Infer from the
references what kind of paper it is — a new method, an application, an empirical study, or a
survey — rather than assuming. Write 150-180 words. Output ONLY the abstract text, without any
heading or explanation.

This paper cites {total} works; the {shown} most representative are listed below:
- Title: {title}
  Abstract: {abstract}
  ... up to 15 references, chosen by distance to the neighbourhood centroid

Abstract: ASSISTANT:
```

When the cap drops nothing, the header reads `This paper cites {total} works, all listed
below:`; with a single reference, `the most representative one is listed below`; with none,
`No cited publications are available for this paper.`

## The four differences the judge is asked to attribute

1. **Instruction reframed** — invention (`a new application paper`) became reconstruction of
   one existing paper, the genre became an open choice from a menu, and the model was asked
   how the paper *departs from* its sources.
2. **Bibliography size disclosed** — the true out-degree is now stated, and the shown
   references are described as the most representative. Era 1 disclosed neither.
3. **Repetition penalty** — 1.15, from none.
4. **Longer generation budget** — 320 tokens, from 256. Truncations fell 238 → 11.

`report.md` finds (1) is the lever in both directions and (4) is the only unambiguous gain.
Its two named culprits — the genre menu and `departs from` — are what era 3 removes; see
`../03_prompt_fix/prompt.md`.

The judge's own prompt is in `compare_abstracts.py`.
