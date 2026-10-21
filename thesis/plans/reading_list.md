# Reading list — Chapter 2 (Background and related work)

Saved 2026-10-15. The publications cited in `plans/02-background.md`. Titles and links are from
`thesis_context/01_literature.md` (arXiv IDs checked there on 2026-10-13).

Priority is a suggestion: **read fully** = a component of the final model (its equation goes in
the thesis and may come up at the defence); **skim** = abstract, main figure and the part cited.

**To verify:** the full titles of P-Tuning v2 and BLIP-2 are not in `01_literature.md`; the ones
below were completed from memory. Check them on arXiv before adding the bib entries.

## 2.1 Graph neural networks

| priority | publication |
|---|---|
| read fully | Gilmer et al., 2017. *Neural Message Passing for Quantum Chemistry*. https://arxiv.org/abs/1704.01212 |
| read fully | Hamilton, Ying & Leskovec, 2017. *Inductive Representation Learning on Large Graphs* (GraphSAGE). https://arxiv.org/abs/1706.02216 |
| read fully | Veličković et al., 2018. *Graph Attention Networks*. https://arxiv.org/abs/1710.10903 |
| read fully | Brody, Alon & Yahav, 2022. *How Attentive are Graph Attention Networks?* (GATv2). https://arxiv.org/abs/2105.14491 |
| skim | Li, Han & Wu, 2018. *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning* (over-smoothing). https://arxiv.org/abs/1801.07606 |
| skim | Zhu et al., 2020. *Beyond Homophily in Graph Neural Networks*. https://arxiv.org/abs/2006.11468 |
| optional | McPherson, Smith-Lovin & Cook, 2001. *Birds of a Feather: Homophily in Social Networks*. https://doi.org/10.1146/annurev.soc.27.1.415 |

## 2.2 Large language models

| priority | publication |
|---|---|
| read fully | Vaswani et al., 2017. *Attention Is All You Need*. https://arxiv.org/abs/1706.03762 |
| read fully | Ainslie et al., 2023. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. https://arxiv.org/abs/2305.13245 |
| skim (architecture section) | Yang et al., 2025. *Qwen3 Technical Report*. https://arxiv.org/abs/2505.09388 |
| skim (RoPE section) | Su et al., 2021. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. https://arxiv.org/abs/2104.09864 |
| skim (n-gram blocking) | Paulus, Xiong & Socher, 2017. *A Deep Reinforced Model for Abstractive Summarization*. https://arxiv.org/abs/1705.04304 |
| optional | Holtzman et al., 2019. *The Curious Case of Neural Text Degeneration*. https://arxiv.org/abs/1904.09751 |

## 2.3 Soft prompting

| priority | publication |
|---|---|
| read fully | Li & Liang, 2021. *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. https://arxiv.org/abs/2101.00190 |
| read fully | Lester, Al-Rfou & Constant, 2021. *The Power of Scale for Parameter-Efficient Prompt Tuning*. https://arxiv.org/abs/2104.08691 |
| skim | Liu et al., 2021. *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks* (title to verify). https://arxiv.org/abs/2110.07602 |
| optional | Liu et al., 2021. *GPT Understands, Too* (P-tuning). https://arxiv.org/abs/2103.10385 |

## 2.4 Parameter-efficient fine-tuning: LoRA

| priority | publication |
|---|---|
| read fully | Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685 |
| skim | Houlsby et al., 2019. *Parameter-Efficient Transfer Learning for NLP*. https://arxiv.org/abs/1902.00751 |
| skim (taxonomy section) | Han et al., 2024. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey*. https://arxiv.org/abs/2403.14608 |

## 2.5 Evaluating generated text

| priority | publication |
|---|---|
| skim | Lin, 2004. *ROUGE: A Package for Automatic Evaluation of Summaries*. https://aclanthology.org/W04-1013/ |
| skim | Zhang et al., 2020. *BERTScore: Evaluating Text Generation with BERT*. https://arxiv.org/abs/1904.09675 |
| skim | Novikova et al., 2017. *Why We Need New Evaluation Metrics for NLG*. https://arxiv.org/abs/1707.06875 |
| read fully | Zheng et al., 2023. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. https://arxiv.org/abs/2306.05685 |
| skim | Liu et al., 2023. *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. https://arxiv.org/abs/2303.16634 |
| skim | Wang et al., 2023. *Large Language Models are not Fair Evaluators*. https://arxiv.org/abs/2305.17926 |
| skim | Panickssery, Bowman & Feng, 2024. *LLM Evaluators Recognize and Favor Their Own Generations*. https://arxiv.org/abs/2404.13076 |

## 2.6 Related work

While reading, note for Tab. A: input; how the graph reaches the LLM (input tokens / KV prefix /
text); LLM training (frozen / LoRA / full); task; how the output is evaluated.

| priority | publication |
|---|---|
| read fully | Perozzi et al., 2024. *Let Your Graph Do the Talking: Encoding Structured Data for LLMs* (GraphToken). https://arxiv.org/abs/2402.05862 |
| read fully | Wang et al., 2024. *Can LLMs Convert Graphs to Text-Attributed Graphs?* (TANS). https://arxiv.org/abs/2412.10136 |
| read fully | Cheng et al., 2024. *xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token*. https://arxiv.org/abs/2405.13792 |
| skim | Tang et al., 2023. *GraphGPT: Graph Instruction Tuning for Large Language Models*. https://arxiv.org/abs/2310.13023 |
| skim | He et al., 2024. *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering*. https://arxiv.org/abs/2402.07630 |
| skim | Chen et al., 2024. *LLaGA: Large Language and Graph Assistant*. https://arxiv.org/abs/2402.08170 |
| skim | Liu et al., 2023. *MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter*. https://arxiv.org/abs/2310.12798 |
| skim | Ge et al., 2023. *In-context Autoencoder for Context Compression in a Large Language Model* (ICAE). https://arxiv.org/abs/2307.06945 |
| optional | Li et al., 2023. *BLIP-2* (full title to verify). https://arxiv.org/abs/2301.12597 |
| optional | Liu et al., 2023. *Visual Instruction Tuning* (LLaVA). https://arxiv.org/abs/2304.08485 |

**Total:** 34 — 13 read fully, 15 skim, 6 optional.
