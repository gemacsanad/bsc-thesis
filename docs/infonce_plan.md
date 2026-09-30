# Step 7 — design rationale for the contrastive losses

Companion to [`soft_prompt_only.md`](soft_prompt_only.md) step 7, which records what was built
and how to run it. This file records the decisions *behind* it — the alternatives considered
and why they were rejected — because step 6 was misdirected by a conclusion whose reasoning had
not been written down.

Written 2026-09-28. The first draft of this file proposed carrying a **voyage InfoNCE** into LM
training as the auxiliary term; that was rejected in review and is not what was built. The
reasoning for the change is §2.

---

## 1. The problem, in one paragraph

Step 6 posted the lowest training loss of any run in this project (1.83, below the oracle and
the text runs) and its worst conditioning at the same time. Cross-entropy over ~230 abstract
tokens is dominated by generic scientific English, so fluent boilerplate outscores a correct
paper on every token while correctness pays on a handful. **Nothing in that objective ever asked
the prefix to be about *this* paper.** Full evidence in
[`step6_failure_analysis.md`](step6_failure_analysis.md).

The fix therefore has to change the *objective*, not the capacity — and specifically not by
adding information, because a plain untrained mean of a node's references already retrieves that
node at 64.34% top-1 while the pipeline delivers 0.32%.

---

## 2. Why Stage B does not use voyage

The first draft proposed InfoNCE against `voyage(target)` as the auxiliary term throughout
training. Two objections were raised in review. One dissolves; one does not.

**"Does this make the project supervised rather than unsupervised?"** It does not change
anything. The main objective is already cross-entropy against the gold abstract, teacher-forced.
An auxiliary loss computed over the same golds introduces **no new label** — it is a different
view of one already in use. The project was never unsupervised. What *would* be a new label is
arXiv categories or human annotation, and none of that is here.

**"Does voyage become a ceiling?"** Yes, and this one is real. Training the anchor toward
`voyage(target)` means information about the target not present in `voyage(target)` receives no
gradient, and may be actively suppressed as useless. A learned head on the target side does
**not** escape it — anything applied to `voyage(abstract)` is bounded by `voyage(abstract)`, by
the data-processing inequality. That is a tempting non-solution worth naming.

There is a third cost the objection surfaced indirectly and which matters for the thesis: using
a commercial embedding model as a training target makes the encoder partly **distilled from
voyage**. That has to be declared, and a reviewer can reasonably ask how much of the result is
voyage's.

**Resolution.** Split by where the ceiling binds.

| stage | objective | ceiling | why acceptable |
|---|---|---|---|
| A (`pretrain_encoder.py`) | InfoNCE vs `voyage(target)` | voyage | initialisation only; the ceiling sits ~50× above the pipeline's 0.32% |
| B (`rank.py`) | ranking over real gold abstracts | the LLM's own | no voyage, no third-party model in the objective |

---

## 3. Alternatives considered for Stage B

**Learned head on both sides.** Rejected: does not remove the ceiling (§2).

**Target the frozen LLM's own representation of the gold abstract**, precomputed once. Cheap
(no extra forwards during training), in the LLM's space rather than voyage's, and it avoids the
third-party dependence. Rejected as second-best: it is still a *representation-matching* loss,
and the quantity actually wanted is "does the prefix make the right abstract more likely". Worth
revisiting if the ranking loss proves too expensive in practice.

**Contrastive LM ranking** — chosen. It is the identifiability metric turned into a
differentiable loss, with no intermediate space.

---

## 4. Implementation decisions worth recording

**Negatives are precomputed, anchors are not.** InfoNCE normally needs large batches because
negatives come from other anchors. Here the candidates are rows of an embedding matrix (Stage A)
or tokenised abstract openings (Stage B) — neither needs an encoder pass. So `BATCH_SIZE = 1`,
forced by a 29.6 GB model on a 40 GB card, is **not** a constraint on the number of negatives.
This asymmetry is what lets a contrastive objective survive into LM training at micro-batch 1.

**Half the negatives are the node's own references.** A prefix carrying only its neighbourhood's
topic would rank the references as highly as the target, because they *are* the topic. Random
arXiv negatives separate on topic alone — the exact shortcut every run so far has taken. This is
the term that attacks "carries topic but not paper identity" directly.

**Candidates are scored behind the prefix, not behind the prompt.** Replaying the 241-token
prompt per candidate is +460%; scoring behind the prefix alone is +61%. The cache fix is
unavailable because gradient checkpointing forces `use_cache=False` — the same constraint
`PrefixCache` exists to work around. `--rank-context prompt` exists to check the cheap variant
ranks the same way. Full reasoning in `rank.py`'s module docstring.

**T = 32 is measured.** TF-IDF retrieval using only an abstract's first N words, against the
*remainder* of all 945 so there is no lexical overlap: 21 tokens → 47.8% top-1, 32 → 57.4%,
43 → 63.2%, against a chance of 0.106%. Lexical matching is a floor for what the LLM reads from
the same tokens. T=32 buys more negatives rather than longer ones.

**The gate is calibrated, not set.** `delta`'s scale depends on the encoder's output
distribution, and a pretrained encoder's differs from a random one's. Step 6 reached
`graph_fraction` 0.70 at a gate of 0.0076, so any literal would mean something different per
run.

**Model selection changed, and it is a correction rather than a preference.** `best.pt` was the
epoch with the lowest LM validation loss — the metric step 6 proved is anti-correlated with
quality. It is now the highest validation ranking accuracy.

---

## 5. What the contrastive losses do *not* fix

Both act on the encoder output, **before** `PrefixAdapter.gate`. Neither makes the LLM read the
result. So the run also had to open the gates at initialisation and drop LoRA back to r=16 — and
those are separate changes, not consequences of the loss.

There is still no direct objective on the **neighbour tokens**. They share the encoder and
benefit indirectly. A token-level contrastive objective is conceivable and explicitly out of
scope.

---

## 6. Open parameters

`λ = 1.0`, `τ = 0.05`, `--gate-init 0.2`, and the Stage A learning rate are all untuned. Stage A
is cheap enough to sweep properly; λ is the one doing real work in Stage B, since `L_LM` and
`L_rank` genuinely conflict — one rewards likely text, the other discriminative text. Both terms
are logged separately rather than only their sum, for exactly that reason.

Also unsettled: whether the encoder should be **frozen** after Stage A rather than fine-tuned.
Frozen cannot drift, which removes the failure mode outright, at the cost of the LM never
shaping the representation. Cheap to test once Stage A exists.
