# Ch. 9 Discussion — plan (2026-11-21; intro and 9.1 WRITTEN)

Budget about 4–5 pages, prose only. Interpretations marked as such. Only current instruments
(Ch. 7–8 numbers, NNLS in embedding space, failure-pattern counts on the joint outputs, Ch. 4
coverage numbers). Not usable: text baseline, field/contribution/claim scores, old-judge quintile
and degree tables, run 10a.

## Skeleton
- Intro (written): interprets Ch. 7–8, limitations, design choices of §1.4.
- 9.1 What the soft prompt carries (written, `sec:disc-carries`): zero input → one text vs 941
  distinct; top-1 18.8% vs 0.11%; ≥ 2 shares 43.0 / 42.5 / 44.6; A.4 examples; interpretation:
  references fix research context more than contribution; not tested; judge does not score
  methods or findings.
- 9.2 Where target-specific information is lost: oracle gap (+9.3 pts top-1, +0.18–0.31
  content); oracle itself 27.8%; NNLS (mean 0.823, GNN 0.833, NNLS fitted to the answer 0.894;
  unrelated-paper control 0.496 → 0.583, i.e. +0.087 vs +0.071); joint fine-tuning (validation
  MRR 0.592 → 0.643, no consistent evaluation gain, GNN cosine loss flat; hypotheses; keep the
  idea of joint fine-tuning justified).
- 9.3 Observed failure patterns: content scores move together (36.9 / 50.4 / 14.2 / 1.2%);
  proposal language 93.4% vs 63.4%, "novel" 69.1% vs 23.1%; 60 shared openings; recorded-reference
  description (ResNet, A.4.4); quality–content observation (§8.3); causes are hypotheses (vector,
  prompt, training distribution, reader) not separated by the checks.
- 9.4 Design choices relative to the assignment: prefixes vs input-embedding soft prompts;
  separate + joint fine-tuning vs end-to-end; reference mean as structure-agnostic baseline (no
  learned encoder run); information carried by the soft prompt (zero/oracle controls + 9.2, no
  dedicated probe); one dataset.
- 9.5 Limitations:
  - data: S2 coverage confound (44% keep only ogbn-arxiv citations; closes the Ch. 4 TODO);
  - method: single-vector compression; oracle embedding includes the title; one training run per
    configuration, no seeds; untuned joint hyperparameters;
  - evaluation: one judge model, no human agreement; judge sees the original; TF-IDF lexical;
    evaluation set over-represents large neighbourhoods; differences describe these 945 papers.

## Excluded by the author (2026-11-21) — do not mention
- training median 5 references vs evaluation mean 51.8;
- "the simulation withholds the text of existing papers rather than testing new nodes over time"
  (the work uses soft prompts, not text);
- possible memorisation of arXiv abstracts by Qwen3 (n-gram repetition was tested; not compared
  with the originals) — leave out.

## Still open (asked 2026-11-21)
1. Development-run evidence for 9.4 (a: brief with numbers / b: "did not produce usable models" /
   c: design reasons only).
2. NNLS in prose or a small table.
3. Re-cut the reference-quality quintiles with the current judge, or drop.
4. Section order.
5. Joint-phase framing.
6. Optional caveats to drop.

## Notes kept from earlier
- Failure patterns: aspects scored in one call, not independent failures; "related area,
  different task" from inspected examples only; all-three-at-1 (26%) is "related but different"
  per the judge, not this pattern's frequency; recorded-reference description: verified case only.
- 9.1 revised 2026-11-21 (review): no 'marked as interpretations' sentence; zero vs GNN estimate (same reader, 1 vs 939 distinct); identification = distinguishing vocabulary; 43–45% only; explanation paragraph as possible explanation incl. information lost in encoding/generation.
- 9.2 WRITTEN 2026-11-21 (`sec:disc-lost`): oracle gap two points; NNLS in prose (source thesis_context/06_results.md §6.2); joint: idea open, hypotheses untested.
- 9.2 rewritten 2026-11-21 (review). Main message: learned neighbourhood representations improve the
  abstracts and the reader benefits from more informative conditioning; open question is how to
  preserve and use target-specific details. NNLS kept as 2–3 sentences, narrow conclusion (no
  limit, no unrelated-paper control); oracle = conditioning quality matters, does not locate the
  loss; joint: improved validation identification, no consistent additional improvement under
  the tested settings, future work on learning rates / loss weighting.
- 9.2 rewritten again 2026-11-21 (review): title "Representation quality and content agreement"
  (`sec:disc-representation`); content scores only (no cosine, no identification); progression:
  learned estimation helps → more informative conditioning helps → joint fine-tuning.
  Fitting diagnostic REMOVED from Ch. 9 → candidate appendix section (source code not yet
  located; control papers' selection unverified). Removed text kept here:
  > A diagnostic in the embedding space asks how closely a weighting of the references could approximate the target embedding if that embedding were known. Non-negative weights fitted to each target's own embedding reach a mean cosine similarity of 0.894, against 0.823 for the reference mean, although part of this gain may come from the flexibility of the fit itself. Because the fit uses the answer and only reweights the existing vectors, it shows that the references can be combined into a closer approximation of the target, but it does not set a limit for a learned estimator such as the GNN, which also transforms the embeddings.
- 9.3 WRITTEN 2026-11-21 (`sec:disc-patterns`, "Patterns in the generated abstracts"): co-occurrence
  36.9 / 50.4 / 1.2% (positive reading); proposal phrasing 93.4% vs 63.8% (recomputed 2026-11-21 with
  the pattern we|this paper|this work + propose|present|introduce; earlier note said 63.4%), "novel"
  69.1% vs 23.1%, 60 shared eight-word openings; explanation: next-token loss + greedy decoding →
  common formulations when conditioning is incomplete (untested); no prompt blame; ResNet case in
  two sentences with appendix reference; quality–content observation left out (no good explanation).
- 9.3 rewritten 2026-11-21 (review): starts with writing quality (84.0% coherence 3) and the land-cover
  example; four example-based patterns (unsupported details: Google Earth; nearby task: retinal
  vessels; contribution type: face-recognition analysis → method; reference replaces target:
  ResNet); only "novel" 69.1 vs 23.1% and the 60 shared openings kept; co-occurrence paragraph
  and proposal-phrase percentages removed; explanation shortened to two hedged sentences.
- §9.3 review 2026-11-21: recurring-wording sentence replaced (six-token constraint restricts repeated token sequences within one output; it does not prevent all repetition).
- §9.3 2026-11-21 (author): removed 'This is an observed example; its frequency was not measured.'
- §9.4 WRITTEN 2026-11-21 (`sec:disc-design`): dataset choice skipped (author); reasons in prose, no new tables; end-to-end paragraph uses earlier runs with numbers (author approved): frozen ≤ 4.2% top-1, LoRA 34–453 distinct of 945 (04 §4/§6), reason = gate at zero + weak gradient + generic tokens; not a controlled comparison.
- §9.4 revised 2026-11-21 (review): no historical numbers (7,750 / 4.2% / 34–453 removed; available for supporting material); gradient explanation removed (some runs started with open gates; prefixes enter every layer; content-word claim not established); staged-training rationale leads; joint phase = end-to-end training; prefix reason brief, no superiority claim; reference mean in three sentences.
- §9.5 Limitations DROPPED 2026-11-21 (author): most items are design choices, already addressed, or irrelevant. Pointers to fix: Ch. 4 line 53 TODO, Ch. 9 intro ('discusses the limitations'), intro outline (\autoref{chap:discussion} '...discusses the limitations and design choices').
- 2026-11-21: pointers fixed (Ch. 4 line 53 sentence + TODO removed; Ch. 9 intro and §1.6 outline no longer mention limitations).
