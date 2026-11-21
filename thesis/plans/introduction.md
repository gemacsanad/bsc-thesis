# Ch. 1 Introduction — approved plan (author, 2026-11-19, revised: new-node framing)

Frame: generating text for **new nodes whose connections to existing nodes are known**. Keep
three things distinct throughout: the motivating setting (a new node), the citation-graph
experiment used to study it, and the conditioning (graph relationships + semantic node
representations). Ch. 3 stays separate for the formal and mathematical task definition.
About 2½–3 pages; no equations, architecture dimensions or results tables.

## 1.1 Motivation
- P1. A new node is added to a graph; its relationships to existing nodes are known; the
  neighbours have textual descriptions, the new node's text is unavailable. Question: do those
  relationships and neighbouring descriptions give enough context to generate useful text for
  it? Introduce text-attributed graphs through this situation, not a formal definition.
- P2. Why graph-conditioned generation: the neighbourhood gives context about the new node's
  likely role, topic or characteristics; the graph decides which nodes give that context; their
  text-derived embeddings represent their content. Idea: encode this neighbourhood information
  into a continuous prompt for a language model instead of inserting neighbouring documents into
  a textual prompt. Intuitive only; definitions in Ch. 2.
- P3. Citation graphs and the central difficulty: a newly considered paper has references to
  existing publications; they indicate the research it builds on and give context for its topic
  and objectives; they do not uniquely determine its contribution, methods or findings. Existing
  abstracts give target texts to evaluate against. No homophily terminology; intuition that
  connected papers often concern related research.

## 1.2 Research question and goal
- P1. Central question: can the text of a new node be generated from its known graph
  neighbourhood using continuous prompting, without supplying the neighbouring texts directly to
  the language model? Citation-graph instance: target text = a paper's abstract; neighbourhood =
  the papers it cites; their titles and abstracts are encoded into vectors before generation;
  the target's own title, abstract and embedding are withheld from the ordinary generation
  input; later citing papers are excluded (information unavailable when the target text was
  written). Precise definitions → Ch. 3.
- P2. Experimental interpretation and goal: evaluated on papers whose abstracts are known, with
  their text withheld from the input, so generated abstracts can be compared with the originals.
  The experiments simulate unavailable target text within an existing graph; they do not
  directly test a chronological stream of entirely unseen nodes. Goal: target-specific content
  agreement, not reproducing the wording; distinguish useful content recovery from fluent but
  generic scientific prose.
- No notation, conditional probabilities, masking details or formal learning problem.

## 1.3 Approach, related work and scope
- P1. G-Retriever as one illustrative graph–LLM approach: continuous graph representation plus
  graph information as text. Restriction investigated here: paper-specific neighbourhood
  information reaches the LLM through continuous representations, alongside a fixed textual
  instruction. Do not claim all approaches need raw graph text or that graph-conditioned
  continuous prompting is unexplored. Details → Ch. 2.
- P2. Pipeline in information order: encode neighbouring titles + abstracts into embeddings; GNN
  estimates the target's embedding from its reference neighbourhood; prefix adapter turns the
  estimate into attention prefixes; prefixes condition generation by Qwen3-14B adapted with
  LoRA. The graph supplies relationships, the neighbouring embeddings supply semantic
  information. Not "generation from topology alone". No layer counts, shapes, heads, gate
  equations or hyperparameters.
- P3. Training strategy and completed work: reader and GNN estimator trained separately, then
  fine-tuned jointly; enriched ogbn-arxiv graph with Semantic Scholar records; the implemented
  pipeline and the comparative evaluation as the engineering work; comparisons test learned
  estimation against the parameter-free reference mean, with diagnostic controls. No separate
  contributions list.
- P4. Scope and assignment-related choices (brief; reasons and limitations → Ch. 9): one
  enriched citation graph; layer-wise key/value prefixes instead of input-token embeddings;
  separate initial training followed by joint fine-tuning of all trainable components (no
  "end-to-end" here; that discussion is in Ch. 9); structure-agnostic baseline =
  parameter-free reference mean. Do not describe the model simply as "not end-to-end".
  Code availability: red TODO until the public repository exists.

## 1.4 Main findings (one paragraph, a preview)
- Word-based identification over 945 original abstracts: the matching abstract ranks first for
  18.8% of generated abstracts.
- About 43–45% of generated abstracts agree (judge, allowing differences in detail) on each of
  subfield, problem and task — each aspect separately.
- Learned GNN estimation improves on the reference mean (no "modest"); joint fine-tuning gives no consistent
  improvement across the final measures.
- Generated texts generally judged coherent; agreement with the target's research content
  remains limited.
- Leave the oracle, detailed differences and explanations to later chapters. Do not equate
  identification or the three content scores with reconstruction of methods and findings.

## 1.5 Thesis outline
Connected prose, chapters grouped: 2 background and related work; 3 formal task formulation;
4–5 dataset preparation and the implemented method; 6 evaluation methodology; 7–8 training
results, final-model evaluation and ablations; 9 interpretation, limitations and
assignment-related scope choices; 10 conclusions and future work; appendices (implementation
details, prompts, training resources, example generations). Re-check Ch. 9–10 sentences later.

## Decisions (2026-11-19)
- §1.4: "improves on the reference mean" (no adjective, no number).
- §1.3 P4: "separate initial training followed by joint fine-tuning of all trainable components".
- Do not mention that the evaluation papers are later than the training papers.
- §1.1 written 2026-11-19 (review applied). §1.2 written 2026-11-19: no comparison questions
  (GNN vs mean is only an ablation, not the centre of the thesis); no pointer to Ch. 6; pointer to
  Ch. 3, which must define the task and restrictions precisely.

## §1.3 and §1.4 — WRITTEN 2026-11-19 (`sec:approach`, `sec:scope`)
§1.3 Approach and related work (3 paragraphs):
- P1 G-Retriever (he2024gretriever) as one illustrative approach: soft prompt + selected graph as
  text; here paper-specific neighbourhood information reaches the LM only through continuous
  representations, alongside a fixed instruction; no "unexplored" claim; pointer to
  `sec:related-work`.
- P2 pipeline in information order: a pre-trained embedding model (not named) encodes each
  reference's title + abstract; GNN estimates the target's embedding (graph decides which papers,
  embeddings supply content); prefix adapter → attention prefixes; Qwen3-14B (named once) with
  LoRA (small trainable matrices, original weights frozen). Pointer to Ch. 5.
- P3 training (separate, then joint fine-tuning) and the work done in first person (enrichment,
  pipeline, evaluation: similarity, identification, judge). Ablations in at most 1–2 sentences
  (reference mean; zero and oracle as diagnostic controls).
§1.4 Scope (own small section): MENTION the differences, no justification (that is in Ch. 9):
- conditioning: layer-wise key/value prefixes instead of projection into the input embedding
  space;
- training: separate initial training followed by joint fine-tuning of all trainable
  components (no mention of the end-to-end-from-scratch attempts);
- structure-agnostic baseline: the parameter-free reference mean (no learned
  structure-agnostic encoder);
- one enriched citation graph (no mention of the children's books dataset);
- additions beyond the assignment: Semantic Scholar enrichment; GNN trained to estimate the
  target's own text embedding (estimator/reader split, oracle as diagnostic); checks of the
  evaluation instruments;
- code: published on GitHub at the end; red TODO for the URL.
Findings move to §1.5, outline to §1.6.
- Review 2026-11-19: §1.4 frames the differences as "design choices within the assigned task" (not "differs from the assignment"); no "from the start", no "learned encoder"; "the work includes" instead of "beyond the assignment". Abbreviations introduced in §1.3: GNN, LoRA; "language-model judge".
- §1.5 Main findings plan approved 2026-11-19: one paragraph; order: setting (final model, 945 papers) → content agreement 43–45% per aspect (43.0/42.5/44.6) → identification 18.8% vs 0.11% random ranking → coherent but limited content → GNN improves on the reference mean; joint no consistent improvement → measures do not show methods/findings recovered. Zero input left out (author: too obvious). No oracle.
- §1.5 WRITTEN 2026-11-19 (`sec:findings`).
- §1.5 revised 2026-11-19 (review): no numbers in the introduction; capability first; 'do not establish reliable reconstruction of exact methods or findings'.
- §1.5 rewritten 2026-11-19 (author: too negative): capability first, distinguishing vocabulary, GNN > mean; joint fine-tuning omitted from the intro; methodology point as an interpretation ('the examples suggest'); ends with 'plausible study, not a faithful reconstruction'.
- §1.6 Outline plan 2026-11-19: one paragraph, short — about one sentence per chapter or chapter pair (2; 3; 4–5; 6; 7–8; 9–10; appendices), \autoref{chap:...}; Ch. 9 sentence names the reasons for the §1.4 design choices; re-check Ch. 3, 9, 10 sentences after they are written.
- §1.6 WRITTEN 2026-11-19 (`sec:outline`); appendices as one sentence. Re-check the Ch. 3, 9, 10 sentences once those chapters are written.
- §1.6 revised 2026-11-19 (review): two paragraphs (Ch. 2–5; Ch. 6–10 + appendices), no component lists, no back-reference to §1.4.
