# Limitations, discussion points and open questions

Sources: the limitations sections of `docs/run11_model.md`, `docs/run9_embedding_decoder.md`,
`docs/chosen_evaluations.md`, `docs/detailed_evaluation.md`, `docs/adapter.md`,
`docs/project_overview.md` §7, `experiments/rubric_sensitivity/README.md`, and the results in
`06_results.md`. This file separates **what the evidence supports** from **what it does not**.
Interpretive statements are labelled.

---

## 1. Claims the evidence supports

| claim | evidence |
|---|---|
| A single 1024-d vector, injected as a 32-token prefix, lets a LoRA-adapted LLM write an abstract of the right paper well above chance | reader←GNN: TF-IDF top-1 18.5% vs 0.11% chance; zero input 0.1%; judge 2.98 vs constant-output controls 1.29–1.35 |
| The content comes from the vector, not from LoRA | zero-vector arm 0.1% top-1; LoRA has no per-paper information |
| The broad field is recovered as well as by reading the references as text | field match 0.784 (GNN) vs 0.789 (text), majority 0.338 |
| Problem / task / subfield are partly recovered, and better with the GNN than with the plain mean | 0–3 scores ~1.3–1.4; GNN − mean +0.14–0.16 (z ≈ 5) |
| The specific contribution and methodology are not recovered | contribution match at majority level for all arms; identification among near neighbours 25% (chance 20%); claim precision 0.14; earlier: methodology wrong 95% of the time even for gpt-5.4-mini reading all references |
| A learned GNN beats the structure-agnostic mean, but only slightly | judge +0.113 (SE 0.037); TF-IDF +2.0 points; cosine +0.010 |
| The references limit the estimate, not the GNN architecture | 8 GNN variants within 0.02 cosine of the mean; fitting reference weights *to the answer* gains no more than fitting them to an unrelated paper (+0.071 vs +0.087) |
| Text still beats the soft prompt | text − joint +0.649 (SE 0.047); joint reaches 82% of the text baseline |
| A frozen LLM cannot read the prefix; LoRA is necessary | oracle frozen 0.3% → oracle + LoRA 12–16.5% top-1, same prefix |
| End-to-end training from scratch collapses on this task | steps 1–8: ≤ 4.2% top-1 frozen; 453 → 34–195 → 59 distinct outputs with LoRA |
| ROUGE and BERTScore do not separate the systems | one constant abstract for all papers: BERTScore 0.828 vs 0.846 text baseline; text has lower ROUGE-2/L than the GNN arm while judged better (planned experiments will formalise this) |
| The judge rubric changes the size of measured effects | same generations: graph gap +0.387 to +0.833 across four rubrics |

## 2. Claims the evidence does **not** support (avoid them)

- That the model is **trained end-to-end** in the sense of the GNN learning from the LM loss:
  in the joint phase the GNN did not move (cosine loss flat at 0.20 at lr 1e-5).
- That **graph structure beyond 1-hop** helps: 2-layer GNNs were worse than 1-layer; uniform
  1+2-hop means retrieve worse (33.2%) than 1-hop means (64.3%).
- That the soft prompt **beats text** on this task, or that it recovers methodology.
- That the GNN gain is large: +0.11 on a 1–10 scale.
- Anything about the **oracle arm's judge score** in run 11 (not judged yet); run 10a's oracle
  (3.41) is a different reader sample.
- That a positive result generalises beyond citation graphs (no other dataset run yet).

## 3. Limitations

### Method
- **Not true end-to-end training.** A real run would use GNN lr ~1e-4 and possibly start from
  the InfoNCE GNN; ~13 h; not done.
- **Single-vector bottleneck.** All references are compressed into one 1024-d vector before the
  LLM sees anything. Per-reference tokens were tried once (step 6) in an end-to-end setting and
  failed; they have not been tried with the staged reader.
- **The estimate, not the reader, is the larger remaining gap** (GNN 18.5% vs oracle 27.8%
  TF-IDF top-1), and it is worst where references do not describe the paper (Q1 quintile:
  joint 2.52 vs oracle 3.34).
- **The reader is one sample of its recipe** (run 10a's reader reached 32.6% oracle top-1, run
  11's 27.8%); it was still improving at epoch 6 and the learning rate never decayed.
- **Untuned joint weights** (cosine weight 1, GNN lr 1e-5) and **no seed sweep** for any run.
- **GQA addressability:** the prefix cannot address individual query heads within a KV group.
- **Gate = 0 is "graph off", not "prefix absent"**: the base prefix positions still absorb
  attention.
- **The InfoNCE GNN** (best retrieval) was never decoded by the LLM.

### Data
- **S2 coverage confound:** 44% of core papers lack a full bibliography; out-degree is not
  comparable across papers.
- **Train/eval shape mismatch:** training papers median 5 references, validation/eval 45–52.
- **The embedding encodes title + abstract**, so the oracle input carries the title; the text
  baseline does not see the target title.
- **Feature/target mismatch:** targets were cleaned of LaTeX, embeddings computed on raw text.
- **Voyage is an input representation, not ground truth:** a limit of the embedding model is
  indistinguishable from a limit of the graph; the GNN objective (cosine to the target's
  embedding) is capped by Voyage.
- **Possible memorisation by the LLM** of arXiv abstracts is not ruled out for Qwen (it was
  checked lexically only for the GPT prompt-sensitivity runs: 0 leaked names, 0% near-verbatim
  copies).
- **Mutual citations and self-loops** exist in the data (removed self-loops; mutual pairs kept).

### Evaluation
- **One judge, one vendor** (`gpt-5.4-mini`); no human agreement study yet; no test–retest noise
  floor; no judge swap (planned, `05` §8).
- **The judge sees the gold** in the old rubric and in parts of the detailed evaluator, so it
  can reward surface similarity.
- **The text baseline is era-1 prompt, graph arms era-4-derived**: the wording confound works
  against the graph arms (+1.46 for Qwen on 48 papers when moving era 1 → 4).
- **Coherence/language scores** are not comparable across evaluators.
- **TF-IDF identification is lexical** and understates paraphrase.
- **Detailed evaluator coverage** is three arms; the validity rule (zero < mean ≤ oracle) is
  not yet checked.
- **Generated abstracts are shorter** than golds (≈ 100 vs 170 words), affecting ROUGE and
  possibly the claim counts.

## 4. Open questions / future work (factual list)

1. A true end-to-end fine-tune (GNN lr ~1e-4, from the cosine or InfoNCE GNN).
2. Per-reference soft tokens on top of the staged reader.
3. Budget-matched text baselines (text truncated to 32 / 128 / 512 tokens vs the 32-position
   prefix) — to quantify what the soft prompt is worth per position.
4. A learned structure-agnostic encoder (MLP/DeepSets over the reference set) as a stronger
   "no structure" baseline than the plain mean.
5. Other datasets: Wikipedia (expanded WikiCS) and one Amazon co-purchase graph (`07`).
6. The evaluation-validity experiments (`05` §8).
7. Seeds and reader variance.

## 5. Discussion points (interpretive — labelled as such)

- *Interpretation:* the results fit a picture in which a paper's references fix its
  neighbourhood in topic space — field, subfield, problem and task — but not its contribution,
  because a contribution is by definition what the references do not already contain. The
  measured ordering field > problem/task > identity among near neighbours > claim precision is
  consistent with this.
- *Interpretation:* on homophilous graphs like citations, the average of neighbours is already
  a strong estimator of a node's own representation, which limits what a GNN can add; graphs
  with complementary neighbours are where a learned aggregator could matter more (`07`).
- *Interpretation:* the end-to-end failures are an instance of a strong decoder ignoring a weak
  conditioning signal (posterior/conditioning collapse); staging the training — teach the
  reader with a strong signal first — avoided it.
- *Interpretation:* the soft prompt is a large compression (all references in 32 positions vs a
  median 27% of references in ~3,500 text tokens) at a cost of ~0.65 judge points; whether that
  trade is worthwhile depends on the setting (e.g. when only embeddings, not texts, are
  available).
