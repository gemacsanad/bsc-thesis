# High-level thesis plan

Approved 2026-10-14. Chapter-level only. Each chapter gets its own detailed plan
(`plans/<chapter-file>.md`) before any writing.

## 1. Coverage of the assignment

Reviewers check the thesis against the task description, so each subtask has a named home.

| subtask (assignment wording, shortened) | where | status to state honestly |
|---|---|---|
| 1. Literature review: GNNs, LLMs, soft prompting, PEFT | Ch. 2 | done |
| 2. Prepare graph datasets with textual node attributes, mainly citation networks | Ch. 4; other datasets in Ch. 10 | One citation network built and used (ogbn-arxiv + Semantic Scholar). Other datasets are future work. |
| 3. Design and implement a GNN method that aggregates the neighbourhood and conditions an LLM | Ch. 5 | Done. **Deviation:** prefix tuning (key/value at every layer), not projection into the input embedding space. |
| 4. Train end-to-end with soft prompts and LoRA | Ch. 5 (staged training), Ch. 8 (end-to-end comparison and why it fails) | **Deviation:** the final model is trained in three phases; in the joint phase the GNN did not move. End-to-end training was implemented and run; it collapsed. |
| 5. Evaluate (LLM-as-a-judge), baselines, ablations, information in the soft prompt | Ch. 6, 7, 8 | Done, with pending results (space reserved, see §4). "Structure-agnostic encoder" = plain reference mean. |
| 6. Document, publish on GitHub, reproducible experiments | Appendix A; URL in Ch. 1 and Ch. 10 | Will be published before submission; the thesis gives the URL. |

Every deviation is stated in Ch. 1 (briefly) and Ch. 9 (in full, with its reason).

## 2. Chapters

| # | chapter | contents | sources | pages (est.) |
|---|---|---|---|---|
| — | Abstracts (Hungarian + English) | task, method, main numbers, honest conclusion; written last | all | 1 + 1 |
| 1 | Introduction | text-attributed graphs; the question; why citation graphs; interpretation of the assignment and a short note on where the design departs from it; contributions; GitHub URL; outline | `00`, `09` §1 | 3–4 |
| 2 | Background and related work | GNNs (message passing, GCN, GraphSAGE, GAT/GATv2, over-smoothing, homophily); LLMs (attention keys/values, RoPE, GQA, Qwen3); soft prompting (prompt/prefix tuning, P-tuning v2, compression: gist, ICAE, xRAG); PEFT (adapters, LoRA, QLoRA); graph + LLM (GraphGPT, GraphToken, G-Retriever, LLaGA, MolCA, TANS); text-generation evaluation (ROUGE, BERTScore, LLM-as-a-judge and its biases); the gap this thesis fills | `01` | 9–10 |
| 3 | Task formulation | the task stated formally: target paper, input = its references only, output = its abstract; why references and not citing papers (temporal leakage); what counts as success; the questions the thesis answers | `00`, `02` §1, §4 | 2–3 |
| 4 | Data | ogbn-arxiv; Semantic Scholar enrichment; joint dataset statistics; Voyage embeddings; splits and frozen node sets; target cleaning; S2 coverage confound; data exploration; train/eval degree mismatch; evidence that the neighbourhood is informative (reference mean cosine, retrieval) | `02` | 5–6 |
| 5 | Method: the final model | the estimator/reader split; GNN estimator (GATv2, `[MASK]`, head, cosine objective); prefix adapter (factorisation, base + gate); Qwen3-14B + LoRA, fixed prompt; decoding; the three training phases; design decisions with the alternative each one beat; forward reference to Ch. 8 for why training is staged | `03` | 8–9 |
| 6 | Evaluation methodology | the text baseline's reference selection and information budget (moved from Ch. 4, 2026-10-30); the arms ladder mapped onto the assignment's baselines (zero / gate = 0 = no neighbourhood; reference mean = structure-agnostic; text baseline; oracle = ceiling); instruments (judge, detailed evaluator, TF-IDF identification, ROUGE/BERTScore); validity evidence; paired statistics; **space for the judge-validation and metric-validity experiments** | `05` | 5–6 |
| 7 | Results | main table and paired differences; what is recovered (field → problem/task → identity → claims); how much information the soft prompt carries (ceiling ladder, input budget, NNLS control); where it fails (quintiles, degree); qualitative examples; **space for the pending arms** | `06` | 8–10 |
| 8 | Ablations and end-to-end comparison | ablations of the final model's components: input ladder (zero / refs-mean / GNN / joint / oracle), GNN grid (aggregator, depth, InfoNCE; figures exist), joint phase vs separate training, decoding (n-gram ban), gate = 0, frozen vs LoRA reader; comparison with end-to-end training and the explanation of why it fails here (closed gate → no gradient, conditioning collapse, weak signal in LM loss). **Not** a run-by-run history. | `03`, `06`, `04` §4, §6 | 6–7 |
| 9 | Discussion | what the evidence supports and what it does not; limitations (method, data, evaluation); deviations from the assignment, each with its reason | `09`, `00` | 4–5 |
| 10 | Conclusion and future work | one paragraph per subtask; future work: true end-to-end fine-tune, per-reference tokens, learned structure-agnostic encoder, other datasets (`07`, with room for the measured probes), evaluation validity | `09` §4, `07` | 2–3 |
| App. A | Implementation and reproducibility | repository and pipeline, software stack, HPC (Komondor), verification scripts, how to rerun, GitHub URL | `08` | — |
| App. B–C | Prompts; hyperparameters | system prompt, judge rubrics, detailed-evaluator definitions; training settings | `03`, `05` | — |
| App. | AI-use declaration | the template table; the author fills it when the thesis is finished | — | — |

Estimated main text: 52–63 pages.

## 3. Decisions taken (2026-10-14)

- **No development-history chapter.** The many earlier experiments are not narrated. Ch. 8
  shows the final model's ablations and compares it with end-to-end training to explain why
  that fails. Earlier runs appear only as evidence for a specific point.
- **The Vicuna text + graph phase is not used.** The Qwen text baseline remains a baseline.
- **Implementation is an appendix** (A). Where it was trained does not matter for the main
  argument.
- **Other datasets are future work** (Ch. 10), with space for them there.
- **GitHub is published before submission**; the thesis cites the URL.
- **Pending results get reserved space** in Ch. 6–8; they will be filled when available.
- **Task formulation and data are separate chapters** (3 and 4).
- **The ablation chapter comes after Results** so its numbers are read with the instruments
  already defined.

## 3a. Decisions taken (2026-11-13)

- The children's-books experiments are dropped completely; arXiv only.
- The text baseline is not used for now; Ch. 6 compares five configurations (zero input,
  reference mean, GNN estimate and oracle with the separately trained reader; the joint model
  with the jointly fine-tuned reader).
- Ch. 6 metric set, framed as three questions: text similarity (ROUGE/BERTScore) →
  identification (TF-IDF) → content agreement (judge: subfield/problem/task). Ch. 6 defines;
  Ch. 7 reports whether each discriminates. The old 1–10 rubric is dropped. Plan: `plans/06-evaluation.md`.

## 4. Pending results with reserved space

- run-11 oracle arm: old judge and detailed evaluator (Ch. 7, 8);
- zero and joint arms: detailed evaluator (Ch. 7);
- pairwise preference metric (Ch. 7);
- judge validation and ROUGE/BERTScore validity experiments (Ch. 6);
- GitHub URL (Ch. 1, 10, App. A).

## 5. Still open

- **Page budget — settled 2026-11-02.** Maximum 50 pages of main text (author). Write each
  chapter fully first; trim at the end, when the whole thesis exists (estimate 52–63, so some
  trimming will be needed). Status 2026-11-02: Ch. 2 ≈ 11 pp., Ch. 4 §4.1–4.4 ≈ 4.5 pp.
- **The end-to-end comparison in Ch. 8 is not a controlled comparison.** The end-to-end runs
  were trained on 7,750 papers, not 90,461, with an earlier pipeline; they share the 945 eval
  papers and the judge. Decide in the Ch. 8 plan which runs to show and how to state this.


## 6. Detailed chapter plans

Each chapter's detailed plan is in its own file, named after its `.tex` file:

- Ch. 2: `plans/02-background.md`
- Ch. 4: `plans/04-data.md`
- Ch. 5: `plans/05-method.md`
