# Literature — the methods this project uses, and related work

Every arXiv identifier below was checked against the arXiv API on 2026-10-13 (title, first
author, year). Non-arXiv items (ACL Anthology, journals, product documentation) are marked; check
their final bibliographic form before citing. For each entry: what it is, and **exactly where
this project uses it** (or how it relates). Entries marked **[used]** are components of the
final model or its evaluation; **[history]** were used in earlier runs; **[related]** are
context only.

---

## 1. Graph neural networks

### Foundations

- **Gilmer et al., 2017 — Neural Message Passing for Quantum Chemistry.**
  https://arxiv.org/abs/1704.01212 — **[related]** The message-passing framework (message,
  aggregate, update) that GCN, GraphSAGE and GAT are instances of. Useful as the general
  formalism in the background chapter.
- **Kipf & Welling, 2017 — Semi-Supervised Classification with Graph Convolutional Networks
  (GCN).** https://arxiv.org/abs/1609.02907 — **[related]** The standard GNN baseline;
  normalised neighbourhood averaging. Not used directly; motivates the aggregation discussion.
- **Hamilton, Ying & Leskovec, 2017 — Inductive Representation Learning on Large Graphs
  (GraphSAGE).** https://arxiv.org/abs/1706.02216 — **[used + history]** Inductive
  aggregation with a separate root (self) weight and a neighbour weight; mean aggregator. The
  encoder of every run up to step 5 (2-layer, mean aggregation). In run 11 it is one of the two
  architectures in the GNN grid (`SAGEConv`, mean).
- **Veličković et al., 2018 — Graph Attention Networks (GAT).**
  https://arxiv.org/abs/1710.10903 — **[related]** Learned attention weights per neighbour.
- **Brody, Alon & Yahav, 2022 — How Attentive are Graph Attention Networks? (GATv2).**
  https://arxiv.org/abs/2105.14491 — **[used]** Fixes GAT's "static attention" (the ranking of
  neighbours did not depend on the query node) by applying the nonlinearity before the attention
  vector. **The final GNN is a 1-layer GATv2 with 4 heads**, 256 channels per head,
  concatenated to 1024 (`GATv2Conv` in PyG).
- **Li, Han & Wu, 2018 — Deeper Insights into Graph Convolutional Networks for Semi-Supervised
  Learning.** https://arxiv.org/abs/1801.07606 — **[related]** Over-smoothing: repeated
  averaging makes node representations converge. Relevant to the measured result that 2-layer
  GNNs were worse than 1-layer ones in run 11's grid (`06_results.md`).
- **Zhu et al., 2020 — Beyond Homophily in Graph Neural Networks.**
  https://arxiv.org/abs/2006.11468 — **[related]** GNNs assume linked nodes are similar
  (homophily). Citation graphs are strongly homophilous, which is why a plain mean of the
  references already gets most of the signal; relevant to `07_other_datasets.md` (graphs whose
  neighbours are complements rather than look-alikes).
- **McPherson, Smith-Lovin & Cook, 2001 — Birds of a Feather: Homophily in Social Networks.**
  Annual Review of Sociology 27. https://doi.org/10.1146/annurev.soc.27.1.415 — **[related]**
  (not arXiv) The original sociological homophily reference.

### Software and benchmark

- **Fey & Lenssen, 2019 — Fast Graph Representation Learning with PyTorch Geometric.**
  https://arxiv.org/abs/1903.02428 — **[used]** PyG 2.8.0 (`NeighborLoader`, `SAGEConv`,
  `GATv2Conv`, `coalesce`, `remove_self_loops`).
- **Hu et al., 2020 — Open Graph Benchmark (OGB).** https://arxiv.org/abs/2005.00687 —
  **[used]** Source of **ogbn-arxiv** (169,343 CS arXiv papers, citation edges, time-based
  split). The core of the dataset (`02_data.md`).
- **Wang et al., 2020 — Microsoft Academic Graph: When experts are not enough.** Quantitative
  Science Studies 1(1). https://doi.org/10.1162/qss_a_00021 — **[used]** (not arXiv)
  ogbn-arxiv is built from MAG; the `mag_id` is the join key between ogbn-arxiv and Semantic
  Scholar.

### Text-attributed graphs and graph + LLM work

- **Jin et al., 2023 — Large Language Models on Graphs: A Comprehensive Survey.**
  https://arxiv.org/abs/2312.02783 — **[related]** Taxonomy of LLM-as-predictor /
  LLM-as-encoder / LLM-as-aligner; good for positioning the thesis.
- **He et al., 2023 — Harnessing Explanations: LLM-to-LM Interpreter for Enhanced
  Text-Attributed Graph Representation Learning (TAPE).** https://arxiv.org/abs/2305.19523 —
  **[related]** Uses LLM-generated text to improve node classification on TAGs (ogbn-arxiv
  among them). Opposite direction to this thesis (text → graph task, not graph → text).
- **Tang et al., 2023 — GraphGPT: Graph Instruction Tuning for Large Language Models.**
  https://arxiv.org/abs/2310.13023 — **[related]** A graph encoder's output is projected into
  the LLM input as graph tokens, with instruction tuning. Closest architectural relative of
  "GNN → projector → LLM".
- **Chen et al., 2024 — LLaGA: Large Language and Graph Assistant.**
  https://arxiv.org/abs/2402.08170 — **[related]** Node neighbourhoods serialised into
  sequences and projected into LLM token space.
- **Perozzi et al., 2024 — Let Your Graph Do the Talking: Encoding Structured Data for LLMs
  (GraphToken).** https://arxiv.org/abs/2402.05862 — **[related]** A GNN produces soft tokens
  for a frozen LLM; only the encoder is trained. The most direct precedent for "GNN output as a
  soft prompt to a frozen LLM".
- **He et al., 2024 — G-Retriever: Retrieval-Augmented Generation for Textual Graph
  Understanding and Question Answering.** https://arxiv.org/abs/2402.07630 — **[related]** GNN
  soft prompt plus retrieved textualised subgraph, frozen LLM (+ optional LoRA).
- **Zhang et al., 2024 — GraphTranslator: Aligning Graph Model to Large Language Model for
  Open-ended Tasks.** https://arxiv.org/abs/2402.07197 — **[related]** Translator module from a
  pretrained graph model's embeddings into LLM space, for open-ended generation.
- **Wang et al., 2024 — Can LLMs Convert Graphs to Text-Attributed Graphs? (TANS).**
  https://arxiv.org/abs/2412.10136 — **[related]** The closest *task*: generate node text from
  the neighbourhood (with GPT-4o-mini, via text prompts), evaluated through node
  classification on Cora, PubMed and airport graphs. Differs from this thesis in that it
  prompts with text, not soft prompts, and evaluates downstream classification, not the text.
- **Feng et al., 2024 — TAGLAS: An atlas of text-attributed graph datasets.**
  https://arxiv.org/abs/2406.14683 — **[related]** Dataset atlas; source of candidate datasets
  in `07_other_datasets.md`.
- **Zhang et al., 2024 — DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs.**
  https://arxiv.org/abs/2406.12072 — **[related]** Considered for other datasets; its rich text
  is on edges, not nodes.
- **CS-TAG — Yan et al., 2023, A Comprehensive Study on Text-attributed Graphs: Benchmarking
  and Rethinking.** NeurIPS 2023 Datasets and Benchmarks.
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/37d00f567a18b478065f1a91b95622a0-Abstract.html
  — **[related]** (not arXiv-checked; verify authors) Source of the Books-History graph proposed
  in `07`.

---

## 2. Large language models

- **Vaswani et al., 2017 — Attention Is All You Need.** https://arxiv.org/abs/1706.03762 —
  **[used]** The Transformer; key/value attention is what the prefix is injected into.
- **Su et al., 2021 — RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE).**
  https://arxiv.org/abs/2104.09864 — **[used]** Qwen3 and Llama use RoPE. Relevant because the
  injected prefix keys are stored already-rotated and therefore position-free, and because RoPE
  is relative, shifting all real tokens by the prefix length does not change token-to-token
  attention (measured, `04`).
- **Ainslie et al., 2023 — GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints.** https://arxiv.org/abs/2305.13245 — **[used]** Grouped-query
  attention. Qwen3-14B has 40 query heads served by 8 key/value heads; the prefix is a key/value
  tensor, so its width is set by the 8 KV heads. This is why the prefix length was raised from
  10 to 32 tokens when moving from Vicuna to Qwen (`03`, `04`).
- **Yang et al., 2025 — Qwen3 Technical Report.** https://arxiv.org/abs/2505.09388 —
  **[used]** The frozen LLM of the final model: **Qwen3-14B** (40 layers, hidden size 5120, 40
  query heads, 8 KV heads, head dimension 128, vocabulary 151,936), used in bf16 with "thinking"
  disabled in the ChatML template.
- **Touvron et al., 2023 — Llama 2: Open Foundation and Fine-Tuned Chat Models.**
  https://arxiv.org/abs/2307.09288 — **[history]** Vicuna-7B-v1.5 is a Llama-2-7B fine-tune
  (32 layers, 32 heads, no GQA).
- **Zheng et al., 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.**
  https://arxiv.org/abs/2306.05685 — **[history + used]** The Vicuna paper reference (Vicuna
  was released by LMSYS; blog: https://lmsys.org/blog/2023-03-30-vicuna/), *and* the standard
  reference for LLM-as-a-judge and its position / verbosity / self-enhancement biases.
- **Wolf et al., 2019 — HuggingFace's Transformers.** https://arxiv.org/abs/1910.03771 —
  **[used]** `transformers==4.51.3` (Qwen3 support begins at 4.51). The prefix injection depends
  on this version's cache internals (`DynamicCache`, `cache_position`).
- **Alayrac et al., 2022 — Flamingo.** https://arxiv.org/abs/2204.14198; **Li et al., 2023 —
  BLIP-2.** https://arxiv.org/abs/2301.12597; **Liu et al., 2023 — Visual Instruction Tuning
  (LLaVA).** https://arxiv.org/abs/2304.08485 — **[related]** Conditioning a frozen (or lightly
  tuned) LLM on a non-text modality through a learned projector. The graph → soft-prompt design
  is the same pattern with a graph encoder instead of an image encoder.
- **Liu et al., 2023 — MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and
  Uni-Modal Adapter.** https://arxiv.org/abs/2310.12798 — **[related]** Graph (molecule) encoder
  → projector → LLM with LoRA; generates text (captions) from a graph. Close analogue of
  "graph → text with soft prompts + LoRA".

---

## 3. Soft prompting

- **Li & Liang, 2021 — Prefix-Tuning: Optimizing Continuous Prompts for Generation.**
  https://arxiv.org/abs/2101.00190 — **[used]** Learned key/value vectors prepended at every
  layer of a frozen LM. **The final model's conditioning mechanism.** Difference from the paper:
  here the prefix is not a free parameter table but is *generated per paper* from a vector by
  the prefix adapter (a conditional prefix), plus a shared learned `base` prefix.
- **Lester, Al-Rfou & Constant, 2021 — The Power of Scale for Parameter-Efficient Prompt
  Tuning.** https://arxiv.org/abs/2104.08691 — **[related / history]** Soft tokens at the input
  embedding layer only. Step 6 of the history added input-embedding soft tokens (one per
  reference) and failed (`04`).
- **Liu et al., 2021 — GPT Understands, Too (P-tuning).** https://arxiv.org/abs/2103.10385 —
  **[related]** Continuous prompts produced by a small encoder (prompt encoder). Conceptually
  close to producing the prompt from a network rather than storing it.
- **Liu et al., 2021 — P-Tuning v2.** https://arxiv.org/abs/2110.07602 — **[related]** Deep
  prompt tuning (prompts at every layer), equivalent to prefix tuning in placement; shows deep
  prompts match fine-tuning across scales.
- **Mu, Li & Goodman, 2023 — Learning to Compress Prompts with Gist Tokens.**
  https://arxiv.org/abs/2304.08467 — **[related]** Compressing a prompt into a few soft tokens.
- **Ge et al., 2023 — In-context Autoencoder for Context Compression in a Large Language Model
  (ICAE).** https://arxiv.org/abs/2307.06945 — **[related]** Long context → memory slots (soft
  tokens) that a LoRA-adapted LLM decodes. Analogue of "one vector stands in for thousands of
  tokens of reference text".
- **Cheng et al., 2024 — xRAG: Extreme Context Compression for Retrieval-augmented Generation
  with One Token.** https://arxiv.org/abs/2405.13792 — **[related]** A retrieval *embedding* is
  projected into a single soft token. The closest precedent for the run-11 reader: an embedding
  vector → soft prompt → frozen LLM writes text consistent with the document.
- **Bowman et al., 2016 — Generating Sentences from a Continuous Space.**
  https://arxiv.org/abs/1511.06349 — **[related]** "Posterior collapse": a strong
  autoregressive decoder learns to ignore its conditioning vector. The same phenomenon as the
  "conditioning collapse" observed when LoRA was added to the end-to-end graph runs (`04`).

---

## 4. Parameter-efficient fine-tuning

- **Houlsby et al., 2019 — Parameter-Efficient Transfer Learning for NLP (Adapters).**
  https://arxiv.org/abs/1902.00751 — **[related]** Bottleneck adapter layers; origin of PEFT.
- **Hu et al., 2021 — LoRA: Low-Rank Adaptation of Large Language Models.**
  https://arxiv.org/abs/2106.09685 — **[used]** The final model applies LoRA with rank 16,
  alpha 32, dropout 0.05 to the **query and value projections** of all 40 Qwen3 layers
  (~10.5M parameters). Rationale for q/v: the prefix supplies keys and values directly, so the
  query projection decides how much of the prefix is attended to and the value projection
  adapts what real tokens carry.
- **Dettmers et al., 2023 — QLoRA.** https://arxiv.org/abs/2305.14314 — **[related]**
  Quantised base + LoRA. Not used (the base runs in bf16); mention as an alternative for
  smaller GPUs.
- **Han et al., 2024 — Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive
  Survey.** https://arxiv.org/abs/2403.14608 — **[related]** Survey to cite for the PEFT
  taxonomy (additive: adapters, soft prompts; reparameterised: LoRA; selective).
- **Hugging Face PEFT library** — https://github.com/huggingface/peft — **[used]** `peft==0.15.2`;
  LoRA is injected in place with `inject_adapter_in_model` rather than wrapped with
  `get_peft_model`, because the wrapper would override the method that the prefix patch
  replaces.

---

## 5. Representation learning used in the GNN stage

- **van den Oord, Li & Vinyals, 2018 — Representation Learning with Contrastive Predictive
  Coding (InfoNCE).** https://arxiv.org/abs/1807.03748 — **[used]** The InfoNCE loss, an
  optional term in the run-11 GNN grid (weight 0.05, temperature 0.05); also stage A of steps
  7–8.
- **Chen et al., 2020 — A Simple Framework for Contrastive Learning of Visual Representations
  (SimCLR).** https://arxiv.org/abs/2002.05709 — **[related]** Temperature-scaled contrastive
  loss with in-batch negatives; the run-11 InfoNCE uses in-batch plus 4,096 sampled negatives.
- **Bardes, Ponce & LeCun, 2021 — VICReg.** https://arxiv.org/abs/2105.04906 — **[history]**
  Variance and covariance regularisation, added in run 8 (step 8) against representation
  collapse; computed on the wrong tensor (`04`); off in run 11.
- **Mu & Viswanath, 2017 — All-but-the-Top.** https://arxiv.org/abs/1702.01417;
  **Su et al., 2021 — Whitening Sentence Representations for Better Semantics and Faster
  Retrieval.** https://arxiv.org/abs/2103.15316; **Ethayarajh, 2019 — How Contextual are
  Contextualized Word Representations?** https://arxiv.org/abs/1909.00512; **Gao et al., 2019 —
  Representation Degeneration Problem.** https://arxiv.org/abs/1907.12009 — **[history]**
  Anisotropy of embedding spaces and the whitening correction. ZCA whitening of the Voyage
  features was applied in runs 8 and 9 and then dropped (`02_data.md`, `04`).

---

## 6. Training and decoding techniques

- **Loshchilov & Hutter, 2017 — Decoupled Weight Decay Regularization (AdamW).**
  https://arxiv.org/abs/1711.05101 — **[used]** Optimiser (weight decay 0).
- **Chen et al., 2016 — Training Deep Nets with Sublinear Memory Cost (gradient
  checkpointing).** https://arxiv.org/abs/1604.06174 — **[used]** Needed to fit Qwen3-14B
  training on a 40 GB A100; it forces `use_cache=False`, which is the origin of the
  prefix-injection bug class (`04`).
- **Paulus, Xiong & Socher, 2017 — A Deep Reinforced Model for Abstractive Summarization.**
  https://arxiv.org/abs/1705.04304 — **[used]** Introduced blocking repeated n-grams at decoding
  time (trigram blocking). The final model decodes with `no_repeat_ngram_size = 6`.
- **Holtzman et al., 2019 — The Curious Case of Neural Text Degeneration.**
  https://arxiv.org/abs/1904.09751 — **[related]** Greedy decoding produces repetition loops;
  the run-10a outputs looped in 17.6% of oracle cases before the n-gram ban.

---

## 7. Data sources

- **Kinney et al., 2023 — The Semantic Scholar Open Data Platform.**
  https://arxiv.org/abs/2301.10140 — **[used]** The bulk datasets (`papers`, `abstracts`,
  `citations`) used to recover the real reference lists of ogbn-arxiv papers.
- **Lo et al., 2019 — S2ORC: The Semantic Scholar Open Research Corpus.**
  https://arxiv.org/abs/1911.02782 — **[related]** Predecessor corpus; cite alongside the
  platform paper.
- **Voyage AI — `voyage-4-large` embedding model.** Documentation:
  https://docs.voyageai.com/docs/embeddings — **[used]** (no paper; product documentation —
  verify the page and access date before citing). 1024-d embeddings, `input_type="document"`.

---

## 8. Evaluation

### Overlap and embedding metrics

- **Lin, 2004 — ROUGE: A Package for Automatic Evaluation of Summaries.** ACL Workshop "Text
  Summarization Branches Out". https://aclanthology.org/W04-1013/ — **[used]** ROUGE-1/2/L F1
  (`rouge-score` package, stemming on).
- **Zhang et al., 2020 — BERTScore: Evaluating Text Generation with BERT.**
  https://arxiv.org/abs/1904.09675 — **[used]** `bert-score` with `lang="en"` (default model
  `roberta-large`), **no baseline rescaling**, which is one reason all arms sit in 0.82–0.85.
- **Novikova et al., 2017 — Why We Need New Evaluation Metrics for NLG.**
  https://arxiv.org/abs/1707.06875 — **[related]** Overlap metrics correlate weakly with human
  judgements; supports the planned "ROUGE/BERTScore are uninformative here" experiment.
- **Sai, Mohankumar & Khapra, 2020 — A Survey of Evaluation Metrics Used for NLG Systems.**
  https://arxiv.org/abs/2008.12009 — **[related]** Metric taxonomy for the evaluation chapter.
- **Fabbri et al., 2020 — SummEval: Re-evaluating Summarization Evaluation.**
  https://arxiv.org/abs/2007.12626 — **[related]** Metric–human correlation methodology for
  summarisation; template for the planned judge-validation study.
- **Sun et al., 2022 — BERTScore is Unfair.** https://arxiv.org/abs/2210.07626 —
  **[related]** Another documented weakness of BERTScore (social bias); peripheral.

### LLM-as-a-judge

- **Zheng et al., 2023** (above, https://arxiv.org/abs/2306.05685) — **[used]** Main reference.
- **Liu et al., 2023 — G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.**
  https://arxiv.org/abs/2303.16634 — **[used as precedent]** Rubric-based LLM scoring of
  generated text.
- **Chiang & Lee, 2023 — Can Large Language Models Be an Alternative to Human Evaluations?**
  https://arxiv.org/abs/2305.01937 — **[related]** Agreement between LLM and expert human
  ratings; basis for the planned human-agreement check.
- **Gu et al., 2024 — A Survey on LLM-as-a-Judge.** https://arxiv.org/abs/2411.15594 —
  **[related]** Survey for the evaluation chapter.
- **Wang et al., 2023 — Large Language Models are not Fair Evaluators.**
  https://arxiv.org/abs/2305.17926 — **[used]** Position bias in pairwise judging; the detailed
  evaluator's pairwise metric asks in both orders to cancel it.
- **Panickssery, Bowman & Feng, 2024 — LLM Evaluators Recognize and Favor Their Own
  Generations.** https://arxiv.org/abs/2404.13076; **Liu et al., 2023 — LLMs as Narcissistic
  Evaluators.** https://arxiv.org/abs/2311.09766 — **[related]** Self-preference bias. Relevant
  because the judge (gpt-5.4-mini) and one archived generator were from the same vendor.
- **Koo et al., 2023 — Benchmarking Cognitive Biases in Large Language Models as Evaluators.**
  https://arxiv.org/abs/2309.17012 — **[related]** Catalogue of judge biases.
- **Min et al., 2023 — FActScore: Fine-grained Atomic Evaluation of Factual Precision.**
  https://arxiv.org/abs/2305.14251 — **[related]** Atomic-claim precision; the detailed
  evaluator's `claim_precision` (supported / total checkable claims) is the same idea against
  the gold abstract.

---

## 9. Privacy-relevant work (only if the thesis discusses what embeddings reveal)

- **Morris et al., 2023 — Text Embeddings Reveal (Almost) As Much As Text (vec2text).**
  https://arxiv.org/abs/2310.06816 — **[related]** Inverting text embeddings back to text. The
  run-11 reader ← oracle arm is a learned embedding → text inverter for Voyage embeddings.
- **Yue et al., 2022 — Synthetic Text Generation with Differential Privacy.**
  https://arxiv.org/abs/2210.14348; **Xie et al., 2024 — Differentially Private Synthetic Data
  via Foundation Model APIs 2: Text (Aug-PE).** https://arxiv.org/abs/2403.01749 —
  **[related]** Needed only if synthetic-data generation for sensitive graphs is discussed as
  future work.

---

## How the literature maps onto the final model (one line each)

| component | method | reference |
|---|---|---|
| node features | `voyage-4-large` text embeddings | Voyage docs |
| neighbourhood aggregator | 1-layer GATv2, masked target row | Brody et al. 2022; PyG |
| structure-agnostic baseline | mean of reference embeddings | (GraphSAGE mean aggregator without learned weights) |
| GNN objective | 1 − cosine to the target's embedding (± InfoNCE) | van den Oord et al. 2018 |
| conditioning | conditional prefix tuning (KV at every layer) | Li & Liang 2021 |
| LLM | Qwen3-14B, frozen, bf16 | Qwen3 report; GQA |
| PEFT | LoRA r16 on q/v | Hu et al. 2021 |
| decoding | greedy + 6-gram blocking | Paulus et al. 2017 |
| evaluation | LLM-as-a-judge, TF-IDF identification, ROUGE, BERTScore | Zheng et al. 2023; Lin 2004; Zhang et al. 2020 |
