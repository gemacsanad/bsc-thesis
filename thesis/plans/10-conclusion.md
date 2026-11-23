# Ch. 10 Conclusion and future work — approved plan (author, 2026-11-22)

About 1.5–2 pages; no chapter intro sentence.

## 10.1 Conclusion (`sec:conclusion-summary`) — WRITTEN 2026-11-22
Thematic (not per subtask), with the four headline numbers:
- P1 what was built: task; pipeline (enriched citation graph → GNN estimate → prefix adapter →
  Qwen3-14B + LoRA); separate stages then joint fine-tuning.
- P2 what it achieves, positive first: coherence 84.0% top score; top-1 18.8% vs 0.11%;
  43–45% ≥ 2 per content aspect; GNN > reference mean by 0.14–0.16 (Ch. 7, 8).
- P3 where it stops: research context, not methods/findings; plausible study vs actual
  contribution (§9.3); oracle → better conditioning representations promising (§9.2).
- P4 assignment: all subtasks addressed, design choices in §9.4; GitHub URL (red TODO).

## 10.2 Future work (`sec:future-work`) — WRITTEN 2026-11-22
- Only the diffusion direction (author: no space). Synthetic TAG by chaining DROPPED, as are
  the smaller items (higher GNN lr, per-reference prefixes, learned structure-agnostic encoder,
  other datasets, human check of the judge).
- Framing: a different generation/conditioning approach to test, not a fix for generic text.
- Citation: nie2025llada (arXiv 2502.09992, checked 2026-11-22: fully masked start, all masked
  positions predicted, low-confidence remasking, no causal mask).

## Review 2026-11-22 (applied)
- 10.1: three paragraphs; built outcomes (dataset, pipeline, evaluation workflow); no numbers; qualitative answer + no faithful reconstruction of methods/findings; no oracle, no assignment-compliance sentence; closing 'working basis' + GitHub TODO.
- 10.2: short; reader replacement, iterative masked-token generation with bidirectional context (accepted tokens not necessarily revisited), test on research problem and task (contribution type is not assessed by the evaluation).

## Appendix A.2 Judge prompts — WRITTEN 2026-11-22
- Verbatim system prompts (COMPARE_SYSTEM, QUALITY_SYSTEM extracted with ast, not executed) as lstlistings lst:judge-content, lst:judge-quality; intro gives the user-message format and says claims and 1-10 fields of the content call are not used. 'Listing~\ref' used (no autoref name for listings).

## Appendix A.3 Training resources — WRITTEN 2026-11-23 (`app:resources`, `tab:resources`)
- Sources: run11_all.sh SBATCH; adapter-qwen-gpu.def; W&B metadata (A100-SXM4-40GB) and system metrics (exported to diagrams/data/run11-*/system.csv); _runtime in summary.json; train.py / lora.py / config.py.
- GiB throughout; evaluation time omitted (no measurement); power and host memory omitted; pointer added to §5.6.
- A.3 revised 2026-11-23 (review): two-column timing table; no memory/utilisation, batch sizes, CPU/host memory, package list or weight-memory paragraph.
