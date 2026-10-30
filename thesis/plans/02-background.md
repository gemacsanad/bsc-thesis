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
- **2.1.3 GraphSAGE + 2.1.4 GATv2** — detailed plan approved 2026-10-24. Together ≈ 1.1 pages,
  hard cap 1.5. Decisions: only GATv2 gets a subsection, and GAT is mentioned only as its origin
  and for the static-attention problem; matrices are bold capitals in the split form
  $\mathbf W_1, \mathbf W_2$ in both layers; the node itself is included in the neighbourhood
  sum, as in the GAT paper; one sentence per layer maps it onto the MPNN equations
  (2.2)–(2.3); **no results from the papers** (not GraphSAGE's $K=2$ finding, not GATv2's
  ogbn-arxiv numbers).
  - **2.1.3 GraphSAGE** (≈ 0.5 page). Cite: Hamilton et al. 2017.
    - P1: transductive per-node embeddings cannot embed unseen nodes; GraphSAGE learns a
      function that aggregates features from the local neighbourhood → inductive.
    - P2: the layer (Alg. 1): aggregate, concatenate with own state, $\mathbf W$, $\sigma$,
      L2 normalisation; the concatenation as a skip connection; split into
      $\mathbf W_1\mathbf h_v + \mathbf W_2\,\mathrm{mean}(\cdot)$ (separate self and
      neighbour weights). MPNN mapping sentence.
    - P3: aggregator must be symmetric (no natural order of neighbours); mean defined, LSTM
      and pooling in one clause; fixed-size uniform neighbour sampling in one sentence.
  - **2.1.4 Graph attention: GATv2** (≈ 0.6 page). Cite: Veličković et al. 2018; Brody et al. 2022.
    - P1: the mean weights all neighbours equally; GAT learns a weight per neighbour, but its
      scoring function computes only static attention (ranking independent of the query node,
      Brody Thm. 1).
    - P2: GATv2 score $e_{vw} = \mathbf a^\top\mathrm{LeakyReLU}(\mathbf W_1\mathbf h_v +
      \mathbf W_2\mathbf h_w)$ (Brody Eq. 7 with App. G's split; bias omitted, as in the
      paper); softmax over the neighbourhood; output $\sum_w \alpha_{vw}\mathbf W_2\mathbf h_w$;
      dynamic attention (Thm. 2) at the same time complexity. MPNN mapping sentence.
    - P3: multi-head attention, $K$ heads concatenated (GAT Eq. 5; motivated as stabilising
      learning). Needed for the 4 concatenated heads in Ch. 5.
  - **Sourcing caveats (checked in the GATv2 PDF, 2026-10-24):**
    - the GATv2 paper gives only the new score (Eq. 7) and does not restate the output
      equation; with $\mathbf W \in \mathbb R^{d'\times 2d}$, GAT's Eq. (4) $\mathbf W\mathbf h_j$
      no longer type-checks. Using $\mathbf W_2$ in the output follows PyG's `GATv2Conv`
      ($\Theta_t$), not the paper; say so or cite PyG (needs a bib entry, Fey & Lenssen 2019);
    - the GATv2 paper defines $\mathcal N_i = \{j : (j,i)\in E\}$ without $i$; including $i$
      comes from the GAT paper ("including $i$") and matches PyG's default self-loops.

### 2.2 Large language models (≈ 2.4 pages + Fig. C and Fig. D)

Detailed plan approved 2026-10-24. Decisions: no forward pointers to the project in the text
(as in 2.1); multi-head attention and causal masking in prose only; no citation for the
repetition problem of greedy decoding (Holtzman dropped); the KV cache is stated as common
practice, without a citation; RoPE gets its relative-position equation.

**Notation** (avoids the 2.1 symbols $T$, $y_v$, $d$, $d'$): tokens of the generated text
$y_1, \dots, y_N$; prompt / context $c$; hidden states $\mathbf X$; model width
$d_{\text{model}}$; key/query dimension $d_k$; matrices in bold capitals ($\mathbf W^Q$,
$\mathbf W^K$, $\mathbf W^V$); positions $m, n$ in the RoPE equation.

- **2.2.1 The Transformer and self-attention** (≈ 0.8 page). Cite: Vaswani et al. 2017.
  - P1: tokens → embeddings; a stack of layers, each with self-attention and a feed-forward
    block, joined by residual connections and normalisation (one sentence, no equation). The
    original Transformer is an encoder–decoder; most modern LLMs, including the Qwen3 model
    used in this project, use only the decoder (stated without a citation, as the author
    decided). Refers to Fig. C.
  - P2: queries, keys and values as projections of $\mathbf X$; scaled dot-product attention
    $\mathrm{softmax}(\mathbf Q\mathbf K^\top/\sqrt{d_k})\mathbf V$ (the one numbered equation);
    one sentence: query–key products decide where a position attends, values carry what is read.
  - P3: multi-head attention in prose (parallel heads, concatenated, projected).
  - P4: causal masking in prose (scores of later positions set to $-\infty$ before the
    softmax). Refers to Fig. D.
- **2.2.2 Language modelling and decoding** (≈ 0.6 page).
  - P1: autoregressive factorisation $p(y_{1:N} \mid c) = \prod_n p(y_n \mid c, y_{<n})$;
    next-token cross-entropy as one equation, over the target tokens only. Project details
    (prompt masking, sliced loss) go to Ch. 5.
  - P2: generation one token at a time; greedy decoding ($\argmax$, deterministic), sampling as
    the alternative in one clause; n-gram blocking forbids any n-gram from occurring twice,
    introduced as trigram blocking. Cite: Paulus et al. 2017. The 6-gram setting and its effect
    go to Ch. 5 / Ch. 8.
- **2.2.3 The key–value cache and grouped-query attention** (≈ 0.5 page). Cite: Ainslie et al. 2023.
  - P1: KV cache as common practice (no citation): keys and values of earlier positions stored
    per layer and reused; a new token computes only its own query, key and value; attention
    sees only the cached keys and values, so the cache can hold vectors that never came from a
    token (the bridge to 2.3).
  - P2: GQA: query heads in groups sharing one key/value head; multi-query and multi-head
    attention as the two extremes; motivation (memory bandwidth of loading keys and values
    during decoding, quality close to multi-head). The cache is sized by the number of KV heads.
- **2.2.4 Qwen3 and rotary position embeddings** (≈ 0.5 page). Cite: Yang et al. 2025; Su et al. 2021.
  - P1: Qwen3 is a family of dense and mixture-of-experts models; Qwen3-14B: 40 layers, hidden
    size 5120, 40 query / 8 KV heads, head dimension 128 (`01_literature.md` §2); its building
    blocks (GQA, RMSNorm, QK-norm, SwiGLU — only those the report states); thinking and
    non-thinking modes in one model (one sentence).
  - P2: RoPE: queries and keys rotated by a position-dependent angle, so that
    $\mathbf q_m^\top\mathbf k_n = g(\mathbf x_m, \mathbf x_n, m-n)$ depends only on relative
    position (numbered equation).

**To verify in the papers before writing:** the Qwen3-14B table and building blocks; GQA's
motivation; Paulus's trigram blocking; RoPE's relative property and its exact form.

### 2.3 Soft prompting (≈ 1.5 pages; written 2026-10-27, ≈ 1.1 pages)

Detailed plan 2026-10-27 (written in the same step at the author's request). Decisions: no
forward pointers to the project (as in 2.1–2.2); compression methods (gist, ICAE, xRAG) stay in
2.6; P-tuning (v1) left out (minimal background); Fig. A not drawn for now. Notation: prefix /
prompt length $M$; layer index $\ell$; $\mathbf P_e$ (prompt tuning); $\mathbf P^K_\ell$,
$\mathbf P^V_\ell$ (prefix); $[\cdot\,;\cdot]$ stacks rows. Reuses $d_{\text{model}}$, $d_k$, $d_v$,
Eq. (2.9) from 2.2.

- **2.3.1 Hard and soft prompts** — prompting = tokens prepended, model conditions on them;
  hard prompt = vocabulary tokens, discrete search; soft prompt = continuous vectors with own
  parameters, backprop, frozen LM (Lester); strictly more expressive (Li & Liang §4.1).
- **2.3.2 Prompt tuning** — $\mathbf P_e \in \mathbb R^{M\times d_{\text{model}}}$ concatenated
  with input embeddings, only $\mathbf P_e$ trained; matches model tuning beyond billions of
  parameters (Lester); two limitations: parameters limited by sequence length, indirect impact
  (P-tuning v2 §3.2).
- **2.3.3 Prefix tuning** — prefix of $M$ "virtual tokens", activations at every layer are free
  parameters (Li & Liang §4.2); activations = key–value pairs (footnote 1) → Eq. `eq:prefix`
  (attention with $[\mathbf P^K_\ell;\mathbf K]$, $[\mathbf P^V_\ell;\mathbf V]$); causal mask
  lets all tokens see the prefix; tokens depend on it through the left context; KV-cache / GQA
  link (one K and V matrix per KV head per layer, uncited, follows from 2.2.3). Second
  paragraph: MLP reparametrisation (§4.3), 0.1% parameters comparable in full data (abstract);
  P-tuning v2 = deep prompt tuning for NLU, 0.1–3% vs 0.01%, matches fine-tuning across scales
  and tasks.
- Cite: lester2021power, li2021prefix, liu2022ptuning (ACL Anthology records).

### 2.4 Parameter-efficient fine-tuning: LoRA (≈ 0.75 page; written 2026-10-27, ≈ 0.55 page)

Detailed plan 2026-10-27 (written in the same step at the author's request). No subsections.
Notation: $\mathbf W_0 \in \mathbb R^{d_{\text{out}}\times d_{\text{in}}}$, $\mathbf B$, $\mathbf A$,
rank $r$, scale $\alpha$ (note: $\alpha_{vw}$ is the GATv2 attention weight in 2.1.4), input
$\mathbf x$, output $\mathbf h$ (column-vector form of the paper; 2.2 uses row vectors, so the
attention matrices are named in words, not as $\mathbf W^Q$, $\mathbf W^V$).

- P1 why PEFT: full fine-tuning = a full copy per task (Hu §1, abstract); PEFT freezes the
  model; adapters (Houlsby) and soft prompts (§2.3) in one sentence; adapter latency (Hu §3).
- P2 LoRA: intrinsic-rank hypothesis; $\Delta\mathbf W = \mathbf B\mathbf A$, shapes, $r \ll
  \min$; Eq. `eq:lora` with $\alpha/r$ (Hu §4.1, Eq. 3); init A Gaussian, B zero; parameter
  count $r(d_{\text{in}}+d_{\text{out}})$ (derived; Ch. 5 uses it: 16·(5120+5120)+16·(5120+1024)).
- P3 where: original experiments adapted attention only, MLP frozen (§4.2); W_q + W_v best on
  GPT-3 at equal parameter budget (§7.1, Table 5); merge $\mathbf W_0 + \frac{\alpha}{r}\mathbf
  B\mathbf A$, no inference latency (§4.1).
- Left out: QLoRA, the 10,000× / 3× numbers, the rank-deficiency analysis.
- Cite: hu2022lora (ICLR 2022, listed on iclr.cc/virtual/2022), houlsby2019parameter (PMLR 97).

- Revised 2026-10-27 on review: opens with the purpose of fine-tuning (examples of a task or
  style; Hu §2 context–target pairs), then what it can and cannot do (content and presentation;
  no guarantee of factual correctness or generalisation; uncited), then the cost of full
  fine-tuning and PEFT; $\alpha/r$ in one sentence (retuning remark dropped); added the rank-$r$
  limitation; q/v result kept as a scoped GPT-3 finding, the project's choice is justified in Ch. 5.

### 2.5 Evaluating generated text (≈ 1 page)

- **2.5.1 Overlap and embedding metrics** — ROUGE-1/2/L; BERTScore; weak agreement with human
  judgement. Cite: Lin 2004; Zhang et al. 2020; Novikova et al. 2017.
- **2.5.2 LLM-as-a-judge** — rubric scoring; known biases (position, verbosity,
  self-preference). Cite: Zheng et al. 2023; Liu et al. 2023 (G-Eval); Wang et al. 2023;
  Panickssery et al. 2024.
- **Written 2026-10-28** (≈ 1 page). Decisions taken without asking, open for review: ROUGE-N/L
  in prose, BERTScore R/P/F1 as `eq:bertscore`; baseline rescaling mentioned (the project uses
  raw scores); Novikova scoped to sentence level (they found metrics *more* reliable at system
  level); Zheng's three judge variants + human agreement without the number; G-Eval in two
  sentences; biases: position (Wang, Zheng; both-orders countermeasure), verbosity (Zheng),
  self-preference (Panickssery; Zheng's own self-enhancement test was inconclusive, so not
  cited for it). TF-IDF identification and the detailed evaluator stay in Ch. 6.

### 2.6 Related work: conditioning language models on graphs (≈ 0.5 page, no subsections)

Expanded 2026-10-30 (author's request, ≈ 1.5 pages with Fig. 2.4): retrieval made precise (frozen pre-trained text encoder, cosine similarity, fixed number of nodes and edges); new paragraph on subgraph construction (rank prizes k…1, fixed cost per edge, max. prize − cost connected subgraph, a PCST variant, prize-less connector nodes); input order soft prompt → subgraph text → question; new paragraph on why both forms (WebQSP ablation: w/o textualized graph and w/o graph encoder both worse; scene-graph hallucination check vs a GNN-soft-prompt-only baseline, authors' single-embedding explanation, retrieval confound stated); contrast paragraph ends with: the thesis is closest to the ablated configuration without the textual subgraph.

Revised 2026-10-28 (third version): **one model only, G-Retriever** (He et al., NeurIPS 2024,
`he2024gretriever`), replacing TANS, which had replaced LLaGA; `wang2025llms` and
`chen2024llaga` removed from the bib. Purpose: show how the components of Ch. 2 fit together in
an existing method, not a paper with the same task; not an experimental baseline. Source: arXiv
2402.07630 v3 §5.4, §6.1–6.2, App. B.2. Three paragraphs: (1) QA over textual graphs; retrieval
in plain words (encode node/edge text once, retrieve the most similar to the question, build a
connected subgraph of manageable size), no PCST; (2) GNN → mean pooling → MLP projection → one
soft-prompt vector before the input, plus the subgraph as text with the question; main
configuration: frozen Llama-2-7B, GNN + projection trained; a further configuration adds LoRA;
(3) shared principle (a learned graph representation conditions an LLM); differences: input
vector vs per-layer KV prefix; subgraph text given vs only the representation from neighbour
embeddings; QA vs generating the node's own text. No benchmark numbers, no other models, no
"first" claims.

### Figures and tables

Saved 2026-10-14. All three are planned. Keep or drop each depending on the space left when the
chapter is written. None exists yet; each is a new figure, made as a separate task.

| # | where | what it shows | why |
|---|---|---|---|
| Fig. A | 2.3.3 | **Prompt tuning vs prefix tuning**, two panels of the same layer stack. Left: soft tokens only at the input embedding layer, flowing up. Right: separate key/value vectors inserted into the attention of every layer, attended to by real tokens. | The distinction behind the Ch. 5 deviation (prefix, not input-embedding projection). Ch. 5 refers back to it. **Done 2026-10-28:** `../diagrams/figures/background/prompt_vs_prefix.{drawio,pdf}`, `fig:prompt-prefix`, referenced in the first sentence of §2.3.3. |
| Fig. B | 2.1.2 | **Message passing on a citation neighbourhood**: a target node, neighbours, messages flowing in along edges, aggregate + update; 1-hop and 2-hop neighbourhoods shaded. | Makes the message-passing equation concrete. Can be adapted for the Ch. 3 task figure (target masked, references only, citing papers excluded). **Done 2026-10-28:** `../diagrams/figures/background/message_passing.{drawio,pdf}`, `fig:message-passing`, three 1-hop references w1–w3 matching panel (b). |
| Fig. C | 2.6 | **The G-Retriever pipeline**, one left-to-right flow: question → larger unlabelled graph with the selected nodes and the connected subgraph → the cut-out subgraph → GNN → one vector → projection → soft prompt, plus the subgraph's nodes and edges as text and the question → frozen LLM → answer. Only text: the question, the answer, GNN / projection / LLM, and "nodes and edges as text" under the document icon. | Makes the G-Retriever paragraphs readable for a new reader. A detailed four-step version was rejected as too complicated. **Done 2026-10-30:** `../diagrams/figures/background/g_retriever.{drawio,pdf}`, `fig:g-retriever`, referenced at the end of the first §2.6 paragraph and placed right after it; included at `width=\textwidth` (author's request, 2026-10-30; the faculty guideline sets no figure width, only the caption is limited to 0.75 text width). |
| Fig. C | 2.2.1 | **Decoder-only Transformer**: token embeddings → a stack of layers (masked multi-head self-attention, feed-forward block, residual connections and normalisation) → next-token distribution. | Marked by the author (2026-10-24); makes the architecture prose concrete. |
| Fig. D | 2.2.1 | **Causal mask** as a small matrix: rows = query positions, columns = key positions; allowed entries on and below the diagonal, $-\infty$ above. Could be a small table or a figure. | Marked by the author (2026-10-24); explains causal masking without an equation. |

- **Not planned:** a GQA/KV-cache figure (could become an annotation on Fig. A instead); LoRA,
  GAT and evaluation metrics (prose or an equation is enough). The Transformer is now Fig. C.
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

### Moved out of Ch. 2 (2026-10-27)

Removed from §2.2.4 on review feedback; the model specifications belong to the method chapter
(Ch. 5), which should also state which mode (thinking / non-thinking) was used. Text as it
stood, with its sources (the bib entries were removed from `mybib.bib` while uncited; they are kept below for Ch. 5):

> Qwen3~\cite{yang2025qwen3} is a family of open LLMs with six dense and two mixture-of-experts
> models. The dense models use GQA and rotary position embeddings~\cite{yang2025qwen3}. The
> dense Qwen3-14B model has 40 layers, with 40 query heads that share 8 key--value
> heads~\cite{yang2025qwen3}, a hidden size of 5120 and a head dimension of
> 128~\cite{qwen2025qwen3config}. A key feature of Qwen3 is that it integrates a thinking mode,
> for complex multi-step reasoning, and a non-thinking mode, for rapid responses, into a single
> model~\cite{yang2025qwen3}.

Sources: Qwen3 report Table 1 (layers, heads); HF `config.json` (hidden size, head dim;
`vocab_size` 151,936 vs the report's tokenizer vocabulary 151,669). §2.2.4 is now
"Positional information" (`sec:positional`), RoPE only.

Removed bib entries (verified records; re-add when Ch. 5 cites them: Qwen3 for the model, Paulus et al. for n-gram blocking in decoding):

```bibtex
@inproceedings{paulus2018deep,
  author    = {Paulus, Romain and Xiong, Caiming and Socher, Richard},
  title     = {A Deep Reinforced Model for Abstractive Summarization},
  booktitle = {International Conference on Learning Representations},
  year      = {2018},
}

@misc{yang2025qwen3,
  author        = {Yang, An and others},
  title         = {Qwen3 Technical Report},
  year          = {2025},
  eprint        = {2505.09388},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  howpublished  = {arXiv preprint arXiv:2505.09388},
}

@misc{qwen2025qwen3config,
  author       = {{Qwen Team, Alibaba Cloud}},
  title        = {{Qwen3-14B} model configuration (\texttt{config.json})},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json}},
  note         = {Accessed: 2026-10-24},
  year         = {2025},
}
```
