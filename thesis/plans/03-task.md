# Ch. 3 Task formulation — plan and decisions (author, 2026-11-21) — WRITTEN 2026-11-21

Role: the one formal, model-independent definition of the task (§1.2 is informal; Ch. 4–5 the
implementation). ~1½ pages, one numbered equation (`eq:task-input`), no figure.
- 3.1 `sec:task-graph`: $G=(V,E)$, $t_u$, $\mathbf{x}_u$, edge (u,v) = v cites u, $\mathcal{N}(v)$
  = recorded references; targets $T \subset V$, original abstract $a_v$; general about the dataset
  (no "ogbn-arxiv papers"/"added papers" terms).
- 3.2 `sec:task-io`: neighbourhood in the wider sense, $\mathcal{N}_k(v)$ (2-hop variants were
  tried); input $I_v$ = subgraph + neighbour embeddings; withheld: own text/embedding, received
  citations (temporal leakage), neighbour texts as text; neighbourhood reaches the LM as a soft
  prompt (assignment wording) + fixed instruction; output $\hat{a}_v$.
- 3.3 `sec:task-success`: content agreement, not wording; plausible ≠ correct; → Ch. 6.
Decisions: stop at input/output (no learning objective); estimator/reader split only in Ch. 5;
overlap with §4.3 kept for now; no "at least one reference" condition.
- Review 2026-11-21: opening states the task directly (no chapter pointers); G_v^(k) contains v, N_k(v) and the path edges; no k=1 / model-selection sentence; one input restriction (v identified, own title/abstract/embedding excluded; abstracts available for training and assessment); no leakage repetition, no 'neighbour texts' item; soft prompt in one sentence; area/problem/task as important aspects, not exhaustive.
