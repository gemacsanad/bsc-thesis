# Ch. 9 Discussion — notes (not an approved plan)

## Observed failure patterns (moved from §7.5, 2026-11-17)
Title in Ch. 9: "Qualitative analysis" or "Observed failure patterns" (not "typical failures").
Evidence, joint model, 945 papers (`experiments/2026-10-09-14541449_run11-joint/`):
- Judge content scores move together: all three ≥ 2 36.9% (349 papers); all three ≤ 1 50.4%;
  all three 0 14.2%; subfield ≥ 2 but problem and task ≤ 1 only 1.2%. These describe the
  judge's assessments; the aspects are scored in one call, so not independent failures.
- Related area, different task: from inspected examples only; all-three-at-1 (26%) is
  "related but different" per the judge, not the frequency of this pattern.
- Bias towards proposing a new method: "we/this paper propose/present/introduce" in 93.4% of
  generated vs 63.4% of original abstracts; "novel" 69.1% vs 23.1%.
- Formulaic openings: 60 generated abstracts share the opening "We present a novel approach to
  the probl…" (shared opening, not collapsed content).
- Description of a recorded reference: verified qualitative case only (47781 → ResNet); no
  frequency. A TF-IDF check could only flag candidates.
- Explanations are hypotheses (vector carries too little specific information; the prompt; the
  training distribution; the reader's use of the representation); current checks do not
  separate them.

## From Ch. 1 (2026-11-19)
- Limits of the simulation: the experiments withhold the text of existing papers in an existing
  graph; they do not directly test new nodes arriving over time (moved here from §1.2).

## Differences from the assignment — briefly justify here (decided 2026-11-19)
Mentioned (not justified) in Ch. 1 §1.4. Justify briefly in Ch. 9:
- prefix tuning instead of projection into the input embedding space (an input-token variant,
  one token per reference, failed in development: step 6);
- separate training + joint fine-tuning instead of end-to-end (end-to-end from scratch collapsed
  in development; in the joint phase the GNN's training cosine loss did not decrease, §7.3);
- reference mean as the structure-agnostic baseline; a learned structure-agnostic encoder was
  not run;
- "information carried by the soft prompts" only partly covered (oracle/zero controls, gates;
  NNLS analysis of how close the references can get belongs here);
- one dataset.
