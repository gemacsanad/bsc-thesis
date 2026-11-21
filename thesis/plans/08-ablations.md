# Ch. 8 Ablations — plan (2026-11-18, decisions taken, not yet written)

Budget 3–4 pages. Interpretation stays short ("this suggests"); the synthesis is in Ch. 9.
Not here: GNN grid (§7.2), n-gram ban (§5.5).

## Author decisions (2026-11-18)
- **No text baseline**: the thesis investigates what the soft prompt contributes.
- **End-to-end runs rejudged** with the current judge (content + quality): step 1
  (`experiments/2026-09-16-14393649_adapter-qwen14b-graphonly-gnn2`, best end-to-end run,
  879 distinct of 945). Old 1–10 judge not used anywhere.
- **No SE anywhere** (no space): differences are reported as plain numbers.
- "How much the vector carries" (NNLS, ceiling of the references) → Ch. 9, not here.
- Frozen vs LoRA reader stays inside 8.4.

## Skeleton
- Intro (2 sentences): each configuration removes or replaces one part; end-to-end last.
- 8.1 Configurations and measures (short): back to Tab. 6.1; zero = no paper-specific
  information; reference mean = no GNN; GNN estimate = no joint phase; oracle = no estimation
  error (a ceiling, not a result). Measures of Ch. 6.
- 8.2 WRITTEN 2026-11-18 (`sec:ablation-results`, `tab:ablation-results`): no figure (space);
  distinct-text column; rows: main three, then controls. End-to-end NOT in Tab. 8.1 (author,
  2026-11-18: earlier answer was a mix-up).
  Earlier plan: Table 8.1, five rows (zero, reference mean, GNN, joint,
  oracle) × measures of Tab. 7.1 (ROUGE-L, BERTScore, top-1, five judge means). One figure,
  designed with the author once the numbers are in (proposal: share ≥ 2 per content aspect).
- 8.3 WRITTEN 2026-11-18 (`sec:ablation-contributions`): prose only, no difference table (all 945
  papers in every configuration, so mean paired difference = difference of means). Order:
  conditioning (mean − zero: top-1 +16.4 pts; content +1.10/+1.25/+1.30; quality +0.85/+0.54) →
  learned estimation (GNN − mean: +2.0 pts; +0.16/+0.15/+0.14; link §7.2 cosine +0.010) → joint
  (joint − GNN: +0.3 pts; +0.03/−0.03/−0.01; no consistent change; link §7.3) → estimation error
  (oracle − GNN: +9.3 pts; +0.31/+0.19/+0.18; why → Ch. 9). One sentence: quality slightly lower
  with more content (GNN − mean language −0.09; oracle − mean −0.10/−0.13), observation only.
  Earlier plan:
- 8.3 What each component contributes (paired mean differences, no SE): reference mean − zero;
  GNN − reference mean; joint − GNN; oracle − GNN.
- 8.4 Comparison with end-to-end training: six runs on 7,750 papers, LM loss only; frozen ≤ 4.2%
  top-1; LoRA collapse 453 → 342 → 34–195 → 59 distinct; why (gate starts at zero, weak signal,
  conditioning collapse); frozen vs LoRA (step 3 0.3% vs step 4 12–16.5% top-1, oracle input);
  not controlled (smaller training set, earlier pipeline). Step 1 with the current judge.

## Evaluation runs — ALL DONE 2026-11-18 (`judge_scores.py`, gpt-5.4-mini, 0 failures)
Judge means (subfield / problem / task / coherence / language), 945 papers each:
- zero 0.00 / 0.00 / 0.00 / 2.00 / 2.00 (one identical text for all papers)
- reference mean 1.10 / 1.25 / 1.30 / 2.85 / 2.54 (content reused from evaluated_result_detailed.csv)
- GNN estimate 1.26 / 1.39 / 1.44 / 2.82 / 2.45 (content reused)
- joint 1.30 / 1.36 / 1.43 / 2.82 / 2.44
- oracle 1.58 / 1.58 / 1.62 / 2.75 / 2.41
- end-to-end step 1 (`2026-09-16-14393649_adapter-qwen14b-graphonly-gnn2`) 0.43 / 0.54 / 0.58 / 2.68 / 2.36
ROUGE-L / BERTScore F1: zero 0.141 / 0.810; reference mean 0.175 / 0.844; GNN 0.179 / 0.845;
joint 0.176 / 0.844; oracle 0.184 / 0.848 (zero and oracle new, `--no_llm`).
Section 8.1 and the intro written 2026-11-18 (`sec:ablation-setup`).

## Removed: comparison with end-to-end training (author, 2026-11-18)
No space to explain fully end-to-end training. Removed from the text: §8.4, the Ch. 8 title
("Ablations and end-to-end comparison" → "Ablations"), the second sentence of the Ch. 8 intro,
and the Ch. 5 §5.6 sentence pointing to it. Kept: Ch. 9 placeholder (deviation "staged instead of
end-to-end training", required by assignment subtask 4) and Ch. 10 future work.
**Possible addition / extra ablation** if space allows: the former §8.4 plan above, plus the
step-1 run rejudged with the current judge (`experiments/2026-09-16-14393649_adapter-qwen14b-
graphonly-gnn2/judge_scores.csv`): distinct 879, ROUGE-L 0.166, BERTScore 0.837, top-1 4.2%,
judge 0.43 / 0.54 / 0.58 / 2.68 / 2.36.
