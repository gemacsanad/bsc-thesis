# Chapter 6 — Evaluation methodology

**Approved 2026-11-13 (author); revised 2026-11-13 after the author's review.** File
`content/06-evaluation.tex`, label `chap:evaluation`. Budget ≈ 3–4 pages.

## Decisions taken (2026-11-13)

- **Keep it clear and simple; show the good metrics.** Every metric has a clear purpose and a
  clear decision behind it. Nothing old or unnecessary.
- **arXiv only.** The children's-books experiments are dropped completely.
- **No text baseline** in this chapter (not used for now).
- **No earlier runs** (gate = 0 controls, random-reference run, rubric/prompt sensitivity
  studies, Vicuna era). Evidence comes only from the final model's configurations.
- **No 128-paper validation procedure** here; Ch. 5 cross-references the general TF-IDF
  definition in this chapter.
- **Three questions, in this order:** text similarity (ROUGE, BERTScore) → identification
  (TF-IDF) → content agreement (judge). Ch. 6 defines the measures and the diagnostic tests;
  **Ch. 7 reports whether they discriminate between configurations.** No predetermined failure
  narrative in Ch. 6.
- **Coherence and language: left out.**
- **No strict validity rule.** Zero and oracle are diagnostic controls, not mathematical bounds;
  metrics are not selected because they reproduce a desired ordering (review point 3).

## Terms (use consistently)

- **configuration** (not "arm" or "input" in the chapter text): zero input, reference mean,
  GNN estimate, joint model, oracle input.
- **separately trained reader** (phase 1) vs **jointly fine-tuned reader** (phase 3).
- **original abstract** = the paper's own abstract that a generated abstract is compared with
  (author 2026-11-13: "reference abstract" is a synonym; keep "original abstract" throughout,
  as in Ch. 4–5, because "reference" already means a cited paper).
- **judge-assessed content-agreement scores** (no human validation).

## The three questions

1. **Text similarity:** how closely does the output resemble the original abstract?
2. **Identification:** does it distinguish the target from other papers?
3. **Content agreement:** which aspects of the target's topic and research problem does it
   recover?

## Skeleton

**Intro (1 paragraph).** What is held fixed: the 945 evaluation papers (`sec:splits`), greedy
decoding as in `sec:decoding-method`, comparison with the original (raw) abstract; the three
questions as the structure of the chapter.

### 6.1 Compared configurations
- P1: five configurations, defined once in Table 6.1 and named consistently afterwards; no
  separate explanation of the five under every metric.
- Table 6.1: configuration | reader vector | reader checkpoint | question it answers

  | configuration | reader checkpoint | question |
  |---|---|---|
  | zero input | separately trained | Does generation depend on paper-specific conditioning? |
  | reference mean | separately trained | What can an untrained, equal-weight aggregation achieve? |
  | GNN estimate | separately trained | Does learned aggregation improve on the reference mean? |
  | joint model | jointly fine-tuned | Does joint fine-tuning improve the separately trained pipeline? |
  | oracle input | separately trained | How well can the reader generate when given the target's own embedding? |

- P2: **the reader checkpoint matters.** Zero, reference mean, GNN and oracle use the separately
  trained reader; joint fine-tuning also changes the adapter and LoRA, so GNN vs joint is not a
  comparison of input vectors alone (review point 2).
- P3: **the main experiment is reference mean → GNN estimate → joint model.** Zero and oracle
  provide context for interpreting it. Mapping to the assignment's two baselines, one sentence
  each (generation without neighbourhood information → zero input; structure-agnostic encoder →
  reference mean; the deviation that it is not learned is stated in Ch. 9).
- P4: **diagnostic controls** (review wording): "Zero and oracle inputs provide diagnostic
  controls. Metrics are examined for whether they distinguish generation without
  paper-specific conditioning from generation conditioned on the target's own embedding. The
  reference-mean configuration provides an intermediate comparison." Floor/ceiling are
  diagnostic roles, not bounds: the oracle supplies more direct information, but the reader is
  not guaranteed to exploit it perfectly.

### 6.2 Text similarity: ROUGE-L and BERTScore — WRITTEN 2026-11-16, revised (author)
- Experiment: `experiments/2026-11-16_metric-check-random-citations/` (200 eval papers,
  `gpt-5.6-luna`), notebook `notebooks/eda/thesis_metric_check.ipynb`.
- Author's rules: no LLM judge in §6.2; no fully-random condition; the control is called
  **"random citations"** (= the same number of papers drawn at random from the paper's own arXiv
  category), never "same-category citations"; not claimed: "irrelevant" or "not significant".
- Text: definitions → experiment → Tab. 6.2 `tab:metric-check` (ROUGE-L 0.153 vs 0.112,
  0.041 (0.002), 94.5%; BERTScore 0.841 vs 0.822, 0.019 (0.001), 97.0%) → reading: ROUGE-L
  difference large, likely from the paper's specific vocabulary (stated as interpretation);
  BERTScore "only 0.019" → shared vocabulary ≠ shared meaning → Tab. 6.3 `tab:metric-pairs`:
  constructed sentence pairs (`example_pairs.py` → `example_pairs.csv`): reversed finding
  0.93/0.99, swapped roles 0.60/0.97, different application 0.73/0.96, paraphrase same meaning
  0.17/0.91; changed words in bold → conclusion: secondary measures, reported in Ch. 7.
- Review applied 2026-11-16: opening = complementary purposes (wording vs contextual embeddings,
  `zhang2020bertscore`), settings in a separate paragraph ("F1, which combines precision and
  recall"); experiment framed as "conditioned on randomly selected papers", "without task-specific
  fine-tuning"; reading = sensitivity, not failure (no "only 0.019", no vocabulary explanation);
  "High scores do not guarantee agreement…"; paraphrase revised to keep the architecture
  ("A transformer network is presented that locates cancerous growths in clinical scans",
  0.26 / 0.92), called a close paraphrase; sentence after Tab. 6.3 on "improves"/"reduces";
  transition to identification and content agreement (judge not named, per the author's rule).
- Fig. 6.1 (`metric_check.pdf`) removed from the thesis for now (author: "replace it, we will see
  the final product later"); the file and notebook cell are kept.

### 6.3 Identification — WRITTEN 2026-11-16 (author: keep TF-IDF short; no 128 validation papers)
- Three paragraphs: why identification (single-abstract comparison can't tell whether other
  papers score equally well) → TF-IDF, ranks, ties, top-1 and MRR explained, chance 0.11% /
  0.0079 (computed H_945/945) → two uses (Ch. 7 comparison; checkpoint selection §5.6 and the
  decoding decision §5.5) and limits (lexical; depends on the 945-abstract collection).
- Review applied 2026-11-16: TF-IDF spelled out; vocabulary and IDF fitted on originals,
  sublinear tf + English stop words (`src/adapter/identify.py:26`); ties = worst rank (empty
  output ranks last; identical output for every paper cannot score high); chance values "under a
  uniformly random ranking"; development uses "on validation papers"; added limitation:
  distinguishing vocabulary ≠ correct claims (leads to §6.4).
- Ch. 5 fixed at the same time: §5.6.1 "defined in \autoref{sec:identification}"; §5.5 decoding
  tolerance now stated (52.0% → 51.5%, tolerance 2 percentage points set before the test).

#### (earlier plan) 6.3 Identification: TF-IDF
- P1: definition — vectoriser (sublinear tf, English stop words) fitted on the original
  abstracts only; each generated abstract ranked against all 945 original abstracts by cosine;
  ties count against the generation; top-1 and MRR; chance 0.11% top-1, MRR ≈ 0.008 (compute
  (1/N)Σ1/k, N = 945, before quoting). Source `05` §2.3, `src/adapter/identify.py`.
- P2: what it adds (review wording): "Identification requires the generated text to resemble
  its own original abstract more closely than the other candidates, reducing the usefulness of
  overlap shared across many papers." No absolute claim that generic text gains nothing.
- P3: limits: lexical (understates paraphrase); **scores depend on this particular 945-paper
  candidate collection**, they are not probabilities of identifying a paper among all
  scientific literature.
- Fulfils the forward reference from §5.6 phase 1 ("defined in full in Chapter 6").
- Existing numbers (`06` §1) belong to Ch. 7.

### 6.4 Content agreement and text quality — WRITTEN 2026-11-16 (`sec:judge-evaluation`)
- Author: report all five aspects (subfield, problem, task, coherence, language); coherence and
  language on a new anchored 0–3 quality call that sees only the generated abstract; explain
  temperature 0 ≈ deterministic; repeat test with 3 runs; methods: **skip completely for now**
  (not mentioned in the text; the exclusion is visible only in the appendix prompt).
- P1 why (lexical measures) → P2 judge, temperature 0, two calls → P3 five aspects and scales
  → P4 Tab. `tab:judge-check` (metric check, content aspects; real vs random citations) + red
  (TODO for coherence/language removed 2026-11-17) → P5 Tab. `tab:judge-repeat`
  (`experiments/2026-11-16_judge-repeat-test/`) → P6 limits (consistency ≠ correctness, no
  humans, one judge, ordinal, coherence near ceiling 2.84).
- Appendix B `app:judge-prompt`: red TODO for both prompts.

### 6.5 Reporting differences — WRITTEN 2026-11-16 (`sec:comparing-configurations`)
- Review applied 2026-11-16: renamed "Reporting differences"; replaced by the reviewer's compact
  three-paragraph version (paired differences, SE equation with scope, higher/equal/lower);
  final paragraph on central comparisons and the evaluation-set scope sentence removed.
  Condition: Ch. 7 must actually report paired mean ± SE and higher/equal/lower shares.
- Paired per-paper differences (top-1 example −1/0/1); mean ± SE, SE = s_d/√945 (one equation);
  scope: no retraining or repeated-judge uncertainty; conclusions apply to the selected
  evaluation set (reference counts differ from training, `sec:splits`); judge scores ordered
  categories → higher/equal/lower shares; two central comparisons (GNN vs mean, joint vs GNN)
  "follow the components and training stages" (not "fixed in advance"); zero/oracle = diagnostic
  controls. No Holm correction. Uncertainty shown in Ch. 7 as mean ± SE (author removed SE
  columns from Ch. 6 tables; decide the Ch. 7 format there).

#### (earlier plan) 6.5 Comparing configurations
- P1: per-paper paired differences on the same 945 papers — mean, SE = sd/√945.
- P2: **main comparisons named in advance:** GNN estimate vs reference mean; joint model vs
  GNN estimate (separately trained); the practical configurations vs the diagnostic controls.
- P3: for the ordinal judge scores, report **better / equal / worse proportions** alongside the
  mean difference.

## Dropped (and why)

| dropped | why |
|---|---|
| old 1–10 judge rubric | "accuracy" mixes in methodology; replaced by the 0–3 content scores |
| blind facets (field, contribution, modality, method family) | several barely above the majority class; need extra explanation |
| claim counts / precision; coherence, language | do not measure whether the neighbourhood helps |
| judge identification among 5; pairwise preference | overlaps TF-IDF identification; pairwise never run |
| on-target margin | not needed |
| strict validity rule (zero < mean ≤ oracle) | an expectation, not proof of validity (review) |
| 128-paper validation MRR, rubric/prompt studies, earlier runs, text baseline | author |

## Needs to be run (elsewhere, not on this machine; author decides where)

**Added 2026-11-16 (author: mark, do not run yet):**
- **Done 2026-11-17: joint model**, content + quality (`src/evaluation/judge_scores.py` →
  `experiments/2026-10-09-14541449_run11-joint/judge_scores.csv`, 945/945, 0 failures). Remaining:
  zero, reference mean, GNN estimate, oracle (same script).
- Quality call (coherence, language, 0–3; `QUALITY_SYSTEM` in
  `experiments/2026-11-16_judge-repeat-test/run.py`) on the configurations used in Ch. 7: zero,
  reference mean, GNN estimate, joint, oracle (≈ 4,725 calls). The old 1–10 coherence/language
  scores are no longer used.
- (Dropped 2026-11-17: quality call on the metric check; the TODO under `tab:judge-check` was removed, nothing in §6.4 uses it.)
- Content call (`evaluate_detailed.compare`, unchanged) on zero, joint and oracle (item 2 below).


1. ROUGE-L and BERTScore F1 on zero and oracle (others exist; for Ch. 7). The shuffled-pairing control is no longer needed (one fixed
   derangement shared across configurations; generations exist; BERTScore loads
   `roberta-large`).
2. Judge content-agreement call (subfield/problem/task) on the zero, joint and oracle
   configurations; today only reference mean and GNN have it.

## Consequences elsewhere

- **Ch. 5 §5.5 (decoding):** "within the predefined tolerance (Chapter 6)" — move the number
  (at most 2 points of TF-IDF top-1; 52.0% → 51.5% on 200 papers, source
  `experiments/2026-10-09-14541449_run11-looptest-ngram6/notes.md`) into §5.5 and keep only the
  reference to the TF-IDF definition. Accepted with the plan; apply when the author says to.
- **Ch. 7:** reports whether each measure discriminates between configurations; focuses on the
  main comparisons; better / equal / worse for judge scores.
- High-level plan: the text baseline and the old judge are no longer part of Ch. 6–7.
