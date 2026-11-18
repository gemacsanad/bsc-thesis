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
