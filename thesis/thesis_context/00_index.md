# Thesis context — index, ground rules, glossary

Written 2026-10-13 on branch `final_model` (from commit `4c57aa5`). This folder is **context for
an LLM that helps write the thesis**. Every file is self-contained and factual. Numbers carry
their source in brackets: an experiment folder under `experiments/`, a document under `docs/`,
or a script under `src/`. Where something is an interpretation, a plan, or not yet done, it is
labelled so.

## The thesis as assigned

**Title:** Neighborhood-Based Node Text Generation with Graph Neural Networks and Large Language
Models.

**Description (verbatim from the assignment):** Graph-structured datasets often contain rich
textual information associated with their nodes, while the graph topology captures contextual
relationships among the nodes. This thesis investigates whether the textual content of a target
node can be generated from the information available in its local graph neighborhood. The
proposed method combines graph neural networks (GNNs) and large language models (LLMs): a GNN
aggregates neighborhood information into node representations, which are projected into the LLM
embedding space and used as soft prompts. The architecture is trained end-to-end, while LoRA
adapters are applied for parameter-efficient fine-tuning. The proposed method is evaluated
primarily on academic citation networks.

**Subtasks:**
1. Literature review on GNNs, LLMs, soft prompting and parameter-efficient fine-tuning.
2. Prepare graph datasets with textual node attributes, primarily academic citation networks.
3. Design and implement a GNN-based method that aggregates neighbourhood information and
   conditions an LLM.
4. Train the architecture end-to-end with soft prompts and LoRA fine-tuning.
5. Evaluate the generated texts (e.g. LLM-as-a-judge), compare with baselines (generation
   without neighbourhood information; a structure-agnostic encoder in place of the GNN), and run
   ablations on the components and on the information carried by the soft prompts.
6. Document the workflow, publish an open-source implementation on GitHub, provide reproducible
   experiments.

## Which file serves which subtask

| file | contents | subtasks |
|---|---|---|
| `00_index.md` | this file: assignment, file map, glossary, naming rules, what is not done | all |
| `01_literature.md` | every method the project uses, with publication and URL, plus related work | 1 |
| `02_data.md` | the arXiv + Semantic Scholar citation graph, splits, embeddings, preprocessing (whitening, target cleaning, node-set filtering), and further data analyses to add | 2 |
| `03_final_model.md` | the final (run-11) architecture and training procedure in detail | 3, 4 |
| `04_development_history.md` | how the design was reached: every training run, what changed, what it showed, including why fully end-to-end training failed | 3, 4 |
| `05_evaluation_methods.md` | every metric and judge used, their validity checks, and the planned extra evaluation experiments | 5 |
| `06_results.md` | all results: the final ablation table, detailed metrics, history numbers. **Not final** — evaluation is still being extended | 5 |
| `07_other_datasets.md` | the non-citation datasets proposed for future runs, with access details and measurements | 2, 5 (future work) |
| `08_implementation_reproducibility.md` | repository layout, software stack, HPC, how to rerun each experiment, what publication on GitHub still needs | 6 |
| `09_limitations_and_discussion.md` | what the evidence supports and does not, limitations, open questions | discussion and conclusion chapters |

## Where the project deviates from the assignment wording (state these honestly)

1. **"Trained end-to-end."** The final model is trained in **three phases** (reader, GNN, joint
   fine-tune). The joint phase fine-tunes the GNN, prefix adapter and LoRA together, but at
   the GNN learning rate used (1e-5) the GNN did not measurably move (`03_final_model.md`). The
   earlier runs that *were* trained end-to-end from scratch (GNN + adapter, with and without
   LoRA) are documented in `04_development_history.md` as the evidence for why the staged design
   was adopted.
2. **"Projected into the LLM embedding space."** The final model does not project into the
   input embedding space. It uses **prefix tuning**: the graph vector is mapped to key/value
   vectors injected into the attention of all 40 layers. One run (step 6) also added
   input-embedding soft tokens (one per reference) and it failed (`04`).
3. **"Structure-agnostic encoder in place of the GNN."** The baseline used for this is the
   **plain mean of the reference embeddings** (no learned parameters, no attention, no
   multi-hop), fed into the same reader. A learned but structure-agnostic encoder (e.g. an MLP
   or DeepSets over the reference set) has not been run.
4. **"Generation without neighbourhood information."** Two such baselines exist: the
   **zero-vector** input to the same trained reader (TF-IDF identification only, not judged),
   and the **gate = 0** controls of earlier runs (one identical abstract for all papers).
5. **GitHub publication** has not happened yet (`08`).

## Glossary (use these names consistently)

| term | meaning |
|---|---|
| **target paper / node** | the paper whose abstract is generated. Its own text and embedding are never an input (except in the oracle ceiling). |
| **references** | the papers the target cites (its out-edges, 1-hop neighbourhood). Only references are used, never citing papers (would leak the future). |
| **core paper** | one of the 169,343 original ogbn-arxiv papers. Only core papers are targets. |
| **S2-only paper** | a paper added from Semantic Scholar because a core paper cites it. Can be a reference, never a target. |
| **Voyage embedding** | the 1024-d `voyage-4-large` embedding of a paper's `title + "\n\n" + abstract`, unit length. The node feature. |
| **reference mean (refs-mean)** | the normalised average of a paper's reference embeddings. The no-learning, structure-agnostic estimator. |
| **GNN estimate** | the output of the trained GNN + linear head: an estimate of the target's own Voyage embedding. |
| **oracle** | the target's own Voyage embedding fed to the reader. Leaks the answer; a **ceiling**, never a result. |
| **zero input** | a zero vector fed to the reader. A floor. |
| **reader** | prefix adapter + LoRA + frozen Qwen3-14B: turns one 1024-d vector into an abstract. |
| **prefix adapter** | the network mapping one vector to 32 key/value tokens at each of the 40 layers. |
| **soft prompt / prefix** | the continuous key/value tokens produced by the adapter. Not words. |
| **gate** | 80 scalars (40 layers × key/value) scaling the paper-specific part of the prefix. Gate = 0 removes the graph on the same weights. |
| **text baseline** | Qwen3-14B reading ~13 reference titles+abstracts as text, no prefix, no fine-tuning. |
| **judge** | `gpt-5.4-mini`, temperature 0, scoring generated vs gold abstract. |
| **TF-IDF identification / top-1** | rank of the generation's own gold abstract among all gold abstracts in the evaluation set by TF-IDF cosine; top-1 = share at rank 1. |
| **step 1 … step 8, run 9, run 10a, run 11** | the training runs in chronological order (`04`). Run 11 is the final model. |
| **eval set** | 945 test-split papers, frozen in `data/frozen/eval_nodes_capped.json`. |
| **validation set** | 806 validation-split papers, frozen in `data/frozen/training_sets_*.json`. |

## What is not done yet (as of 2026-10-13)

- Judge (old rubric and detailed evaluator) on the **oracle** arm of run 11; detailed evaluator
  on the **zero** and **joint** arms. Only TF-IDF identification exists for oracle and zero.
- The planned evaluation-validity experiments (`05_evaluation_methods.md`, last section):
  showing that ROUGE/BERTScore cannot separate the arms, and measuring how correct the LLM
  judge is (human agreement, repeat-scoring noise, judge swap).
- A true end-to-end run in which the GNN actually moves (GNN lr ~1e-4).
- Runs on non-citation datasets (`07_other_datasets.md`).
- GitHub publication and the README update for the final model (`08`).
- The root `README.md` still describes the original Vicuna-7B/GraphSAGE pipeline; it is out of
  date. The docs under `docs/` are current.

## Ground rules for writing from these files

- Never quote a number without its arm and its evaluation instrument (old judge rubric vs
  detailed evaluator vs TF-IDF). Judge coherence from the old rubric and from the detailed
  evaluator are **not** comparable (`05`).
- The oracle is always labelled a ceiling.
- Methodology is not claimed to be recoverable; the evidence says it is not (`06`, `09`).
- Results from runs before run 11 are history (`04`), not part of the final ablation table,
  unless they are the specific evidence for a design decision.
