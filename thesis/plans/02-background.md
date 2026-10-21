# Chapter 2 — Background and related work (detailed plan)

Approved 2026-10-14. File `content/02-background.tex`, title "Background and related work".
Sources: `thesis_context/01_literature.md` (all URLs for bib entries).

**Decisions:** related work is its own section (2.6); key equations for each method (message
passing, GATv2 attention, self-attention, prefix, LoRA), reused as notation in Ch. 5; text
embeddings / Voyage are **not** here (Ch. 4 only); the items marked [added] were accepted.
**Budget ≈ 8.5–9 pages**, written to budget rather than cut afterwards.

### 2.1 Graph neural networks (≈ 2 pages)

- **2.1.1 Graphs with text on the nodes** [added] — one paragraph: text-attributed graph,
  notation (nodes, edges, neighbourhood, node features); homophily (linked nodes are similar),
  which later explains why the plain reference mean is strong (Ch. 7–8).
  Cite: Zhu et al. 2020 (optionally McPherson et al. 2001).
- **2.1.2 Message passing** — message / aggregate / update as one equation; k layers = k-hop
  receptive field; over-smoothing in one sentence [added] (explains 2 layers < 1 in the grid).
  Cite: Gilmer et al. 2017; Li, Han & Wu 2018.
- **2.1.3 GraphSAGE** — separate self and neighbour weights, mean aggregator, inductive; one of
  the two architectures in the GNN grid. Cite: Hamilton et al. 2017.
- **2.1.4 GAT and GATv2** — learned attention weight per neighbour; GATv2's fix of static
  attention [added], with its equation, since the final GNN is GATv2.
  Cite: Veličković et al. 2018; Brody et al. 2022.

### 2.2 Large language models (≈ 2–2.5 pages)

- **2.2.1 The Transformer and self-attention** — queries, keys, values; scaled dot-product
  attention (the one attention equation); multi-head; causal masking. Cite: Vaswani et al. 2017.
- **2.2.2 Next-token prediction and decoding** — cross-entropy objective; autoregressive
  generation; greedy decoding and n-gram blocking in two sentences [added] (the final decoding,
  and the loop-fix ablation in Ch. 8). Cite: Paulus et al. 2017 (optionally Holtzman et al. 2019).
- **2.2.3 KV cache and grouped-query attention** — what the cache stores and that it can hold
  "past" positions that were never tokens (the bridge to prefix tuning); GQA: query heads share
  key/value heads. Cite: Ainslie et al. 2023.
- **2.2.4 Qwen3** — short paragraph of figures (40 layers, hidden size 5120, 40 query / 8 KV
  heads, head dimension 128); RoPE in one or two sentences [added]: position encoded by rotating
  queries and keys, so attention depends on relative position — Ch. 5 refers back to this for
  the prefix positions. Cite: Yang et al. 2025; Su et al. 2021.

### 2.3 Soft prompting (≈ 1.5 pages)

- **2.3.1 Hard and soft prompts** — discrete tokens vs continuous vectors; soft prompts can carry
  information that has no words.
- **2.3.2 Prompt tuning** — learned vectors at the input embedding layer.
  Cite: Lester et al. 2021 (optionally P-tuning, Liu et al. 2021).
- **2.3.3 Prefix tuning** — learned keys/values at every layer, placed in the KV cache (the
  prefix equation); how it differs from prompt tuning (the basis of the Ch. 5 deviation);
  P-tuning v2 as the deep variant. Cite: Li & Liang 2021; Liu et al. 2021 (P-tuning v2).

### 2.4 Parameter-efficient fine-tuning: LoRA (≈ 0.75 page)

- Why PEFT: cost of full fine-tuning; families in one sentence (adapters, soft prompts, LoRA).
- LoRA: ΔW = BA (the equation), rank, alpha, choice of target matrices, mergeable at inference.
  Cite: Hu et al. 2021; Houlsby et al. 2019 and/or Han et al. 2024 (survey).

### 2.5 Evaluating generated text (≈ 1 page)

- **2.5.1 Overlap and embedding metrics** — ROUGE-1/2/L; BERTScore; weak agreement with human
  judgement. Cite: Lin 2004; Zhang et al. 2020; Novikova et al. 2017.
- **2.5.2 LLM-as-a-judge** — rubric scoring; known biases (position, verbosity,
  self-preference). Cite: Zheng et al. 2023; Liu et al. 2023 (G-Eval); Wang et al. 2023;
  Panickssery et al. 2024.

### 2.6 Related work: generating text from graphs with LLMs [added] (≈ 1.25 pages)

Required by the faculty guideline (prior work, similar solutions, conclusions drawn).
One paragraph each:
- graph encoder → soft tokens → LLM: GraphToken, GraphGPT, G-Retriever, LLaGA;
- the same pattern with other modalities: MolCA (optionally BLIP-2 / LLaVA);
- one vector → text: xRAG, ICAE;
- the closest task: TANS (node text from neighbours, but via text prompts and evaluated by
  downstream classification).
Closing paragraph: the gap this thesis fills.

### Figures and tables

Saved 2026-10-14. All three are planned. Keep or drop each depending on the space left when the
chapter is written. None exists yet; each is a new figure, made as a separate task.

| # | where | what it shows | why |
|---|---|---|---|
| Fig. A | 2.3.3 | **Prompt tuning vs prefix tuning**, two panels of the same layer stack. Left: soft tokens only at the input embedding layer, flowing up. Right: separate key/value vectors inserted into the attention of every layer, attended to by real tokens. | The distinction behind the Ch. 5 deviation (prefix, not input-embedding projection). Ch. 5 refers back to it. |
| Fig. B | 2.1.2 | **Message passing on a citation neighbourhood**: a target node, neighbours, messages flowing in along edges, aggregate + update; 1-hop and 2-hop neighbourhoods shaded. | Makes the message-passing equation concrete. Can be adapted for the Ch. 3 task figure (target masked, references only, citing papers excluded). |
| Tab. A | 2.6 | **Related-work comparison**: one row per method (GraphToken, GraphGPT, G-Retriever, LLaGA, MolCA, xRAG, TANS, this thesis); columns: input; how the graph reaches the LLM (input tokens / KV prefix / text); LLM training (frozen / LoRA / full); task; how the output is evaluated. | Shows the gap at a glance. Filled only from the papers; gaps marked as visible TODOs. |

- **Not planned:** a GQA/KV-cache figure (could become an annotation on Fig. A instead); the
  Transformer, LoRA, GAT and evaluation metrics (prose or an equation is enough).
- **Space:** the two figures take ≈ 0.6–0.8 page of the 8.5–9-page budget.
- **Style:** the same as the existing diagrams: 112.5 mm wide (included without `width=`),
  Latin Modern, Okabe–Ito palette, no titles inside the figure.
- **Open:** the drawing tool (standalone TikZ compiled separately to PDF, which needs no
  template change, or another tool exported to PDF) and the location (`figures/` or
  `../diagrams/`).

### Deliberately left out of Ch. 2

| topic | where instead |
|---|---|
| text embeddings / Voyage | Ch. 4 |
| InfoNCE | Ch. 5 or 8 (GNN grid option) |
| posterior / conditioning collapse | Ch. 8 |
| TF-IDF identification, detailed evaluator | Ch. 6 |
| GCN, QLoRA | optional one-liners only |

### Citations

None are in `bib/mybib.bib` yet; each is added from its arXiv / ACL record when its subsection
is written. Removing the template's sample bib entries still needs the author's go-ahead.
