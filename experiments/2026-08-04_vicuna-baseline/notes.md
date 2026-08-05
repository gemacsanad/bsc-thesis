# 2026-08-04 — Vicuna-7B baseline, no prefix

The control for [`../2026-08-04_adapter-ep2`](../2026-08-04_adapter-ep2). Frozen Vicuna-7B
reading the same 1-hop reference titles and abstracts, with no graph-conditioned prefix.

Scores and the paired comparison are recorded in the adapter run's `notes.md`; this directory
exists so the baseline is a first-class run with its own config and outputs rather than a column
in someone else's table.

## Why this was regenerated

`_archive/vicuna-7b-8bit` is also "Vicuna with no prefix", but it is **not** a usable control: it
ran in 8-bit and sampled at temperature 0.35, so quantisation error and sampling noise would both
sit on top of the effect being measured. This run is bf16 and greedy, matching the adapter run
exactly. It also runs on the 495-node joint-graph eval set rather than the superseded 200-node
ogbn-only set.

## Note

The 10 prefix positions are charged against the 4,096-token context **in this run too**, even
though no prefix is injected. That is deliberate: it makes the two runs overflow on identical
nodes, so they compare on the same 432 rows.
