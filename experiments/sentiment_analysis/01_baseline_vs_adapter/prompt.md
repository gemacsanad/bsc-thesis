# The prompt behind this round

Both versions judged here — the no-graph baseline and the adapter — were generated on
2026-08-15 from the **same** prompt, `src/adapter/text.py` at commit `3ae4c3b`. Nothing about
the text differs between the two arms; the only difference is whether the GNN prefix is
active. That is what makes this round a measurement of the graph rather than of the wording.

This is prompt **era 1**, in use until 2026-08-19.

## What the model received

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives
helpful, detailed, and polite answers to the user's questions. USER: You are a scientific
writing assistant. Based on the source publications listed below, write a concise abstract
for a new application paper that builds on and combines these works. Output ONLY the abstract
text, without any heading or explanation.

Source publications:
- Title: {title}
  Abstract: {abstract}

- Title: {title}
  Abstract: {abstract}

  ... up to PROMPT_MAX_REFS = 15 references, chosen by distance to the neighbourhood centroid

Generated abstract: ASSISTANT:
```

With no cited works available, the source block becomes
`(no neighbouring sources available)`.

## Properties worth noting

- The target paper's **own title and abstract never appear**. The model sees only the papers
  it cites and has to infer the topic from the neighbourhood.
- The reference list is capped at 15 but the **count is not disclosed**, so a paper citing 96
  works is indistinguishable from one citing 6. Era 2 added that line.
- The instruction asks for **a new application paper** that **combines** the sources. Both are
  wrong for the task: the target is reconstruction of one existing paper, not invention, and
  fixing the genre means surveys and empirical studies cannot be written under it at all.
  Those are the failure modes `report.md` measures, and the reason for the era-2 rewrite.

## Decoding

`MAX_NEW_TOKENS = 256`, no repetition penalty. 238 of the adapter's 945 outputs ended
mid-sentence under this budget — raised to 320 for era 2.

The judge prompt for this round is not part of the generation pipeline; round 1's judgements
are in `comparisons.md`.
