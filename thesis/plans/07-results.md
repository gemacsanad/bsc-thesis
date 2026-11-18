# Ch. 7 Training and results — approved plan (2026-11-16)

Chapter split (`high_level_plan.md` §3b): Ch. 7 = the three training phases + the evaluation of
the **final model = the jointly fine-tuned model** (the full pipeline of Ch. 5, not chosen for its
scores). All comparisons with other configurations are in Ch. 8. Not in Ch. 7: "how much the
vector carries" (oracle gap, NNLS, retrieval).

Binding rules (`09` §2): do not claim the joint phase moved the GNN; do not claim the GNN gain is
large; do not claim methodology/contribution is recovered. Only one validation count appears in
the thesis: **806** (author, 2026-11-17). Identification is described as on "a fixed subset of the
validation papers" with no count (it really is the first 128; writing 806 there would be false).

## Intro (2–3 sentences) — WRITTEN 2026-11-18; chapter title "Training and results"
Follows the three phases of §5.6, then evaluates the final model with the Ch. 6 measures;
comparisons with the other configurations → Ch. 8.

## 7.1 Phase 1: the reader
- Argues: the reader learned to use the vector.
- Evidence (`thesis_context/03_final_model.md` §4.1): validation identification MRR on the 128
  validation papers, reference-mean input 0.126 → 0.530 (epoch 5 selected); oracle input
  0.162 → 0.704, still rising at epoch 6; mean |gate| 0 → 0.10, still rising; training loss
  3.1 → ≈ 2.13.
- Figures (2): `../diagrams/data/run11-reader-14541449/figures/reader_identification_mrr.pdf`;
  `.../reader_gate_magnitude.pdf`. Captions from the notebook's suggested captions.

## 7.2 Phase 2: the GNN estimator — WRITTEN 2026-11-17 (`sec:results-gnn`)
- Final form (author): one full-width panel with all 8 variants, legend to the right
  (`diagrams/figures/gnn_comparison/gnn_selection.pdf`, last cell of `gnn_comparison.ipynb`);
  no panel (b), no table, no InfoNCE retrieval (not the best model; space). Evaluation-set
  cosine 0.833 vs 0.823 in one sentence. Ch. 5 §5.6.2 now points here.
- Appendix `app:training-diagnostics`: red TODO (top-1, prefix contrast, canary per phase).

(earlier plan below)
- Argues: why this variant — all eight compared; the selected one against the reference mean.
- Evidence (`03` §4.2; `diagrams/documentation.md`): 8 variants {GATv2, SAGE} × {1, 2 layers} ×
  {cos, cos + InfoNCE}; selected GATv2 1-layer cos, validation cosine 0.828 (epoch 28) vs
  reference mean 0.816; evaluation papers 0.833 vs 0.823; all variants within 0.02; two layers
  worse than one; InfoNCE trades cosine for top-1.
- Figures (2, side by side or consecutive): all variants
  `../diagrams/figures/gnn_comparison/gnn_cosine_validation.pdf`; the selected variant alone
  `../diagrams/data/run11-gnn-gatv2-l1-cos-14541449/figures/gnn-gatv2-l1-cos_cosine_validation.pdf`.
- Possibly a small table of the 8 selected-epoch validation cosines (`documentation.md`) — decide
  when writing.

## 7.3 Phase 3: joint fine-tuning — WRITTEN 2026-11-17 (`sec:results-joint`)
- Author: no uncertainty for the 128-paper MRR; no "no epoch-0 measurement"; GNN stillness shown
  by panel (c) only (no pairwise output cosine); mention the gate growth (0.092 → 0.108);
  framing: do not centre on "joint changed nothing" — the idea of joint fine-tuning must stay
  justified (reader adapts to its in-use input); the GNN estimator changing little is stated
  factually (binding rule).
- Figure `joint_overview.pdf` (Fig. 7.3). Numbers from `diagrams/data/run11-joint-14541449/history.csv`.

(earlier plan below)
- Argues: what the joint phase changed — the reader specialised; the GNN did not move.
- Evidence (`03` §4.3): GNN-input MRR 0.592 → 0.643 (epoch 3 selected); reference-mean MRR
  0.525 → 0.484; GNN cosine loss flat at 0.20 and pairwise output similarity 0.54 in all three
  epochs (lr 1e-5 → effectively frozen) → this phase adapted the reader, it is not end-to-end.
- Figures (2): `../diagrams/data/run11-joint-14541449/figures/joint_identification_mrr.pdf`;
  `.../joint_gnn_cos_loss.pdf`.

## 7.4 Evaluation of the final model — WRITTEN 2026-11-17 (`sec:results-final`, `tab:final-model`, `fig:judge-distribution`;
  notebook `notebooks/eda/thesis_results.ipynb`; level ≥ 2 shares from unrounded data: 43.0 / 42.5 / 44.6%)
- P1 setup: joint model, one abstract per evaluation paper (945), decoding of §5.5, measures of
  Ch. 6; comparisons → Ch. 8.
- Table 7.1 (all from `experiments/2026-10-09-14541449_run11-joint/`): ROUGE-L F1 0.176,
  BERTScore F1 0.844 (`evaluated_result.csv`); top-1 18.8% (chance 0.11%), MRR 0.301 (chance
  0.008); subfield 1.30, problem 1.36, task 1.43, coherence 2.82, language 2.44 (`judge_scores.csv`).
  Reference points: chance for identification only (no Ch. 6 calibration values). Only the
  measures defined in Ch. 6 (no ROUGE-1/2, no BERTScore P/R).
- Figure 7.4: five horizontal 100% stacked bars (content group | quality group), legend =
  numeric levels 0–3, caption points to Tab. 6.4 / App. B for meanings. Both table and figure.
  Shares (%) 0/1/2/3: subfield 24.7/32.4/31.7/11.2; problem 14.5/43.0/34.5/8.0;
  task 16.9/38.5/29.3/15.2; coherence 0.1/2.1/13.8/84.0; language 0.3/3.1/48.5/48.1.
- P2 identification far above chance (interpretation: paper-specific vocabulary).
- P3 content: levels ≥ 2 = 42.9 / 42.5 / 44.5%; level 0 = 24.7 / 14.5 / 16.9%; no methodology claim.
- P4 quality: coherence 84.0% at 3 (near ceiling caveat from §6.4); language 48.5% / 48.1% at 2 / 3.
- Length (95 vs 170 words): not now.

(earlier plan below)
- Table: joint model means on the 945 papers — ROUGE-L 0.176, BERTScore 0.844
  (`experiments/2026-10-09-14541449_run11-joint/evaluated_result.csv`); TF-IDF top-1 18.8%
  (chance 0.11%), MRR 0.301 (recomputed 2026-11-16 from `generations.csv` with
  `src/adapter/identify.py`); subfield, problem, task, coherence, language = red placeholders
  until the judge runs (`plans/06-evaluation.md`, "Needs to be run").
- Text: identification far above chance; generated abstracts shorter than the originals
  (median 95 vs 170 words).
- **New figure (to be designed with the thesis-figure skill; author wants it nicely planned):**
  score distribution of the joint model on the five judge aspects. Proposal to refine with the
  author before drawing:
  - one horizontal 100% stacked bar per aspect (subfield, problem, task | coherence, language),
    a visible gap between the content and quality groups;
  - segments = score levels 0–3, one sequential hue light → dark, level names from the rubric
    in a legend below ("different", "related but clearly different", "same with differences in
    detail", "same"; quality levels need their own short names);
  - shares printed inside segments ≥ 5%;
  - full text width or 112.5 mm; notebook `notebooks/eda/thesis_results.ipynb`;
    output `diagrams/figures/results/judge_distribution.{pdf,png}`.
  - Needs the judge runs on the joint model first.

## 7.5 Typical failures — DROPPED 2026-11-17 (review + author)
- No §7.5. §7.4 ends with a pointer to the examples (`app:examples`) and to Ch. 9; the failure
  analysis moves to Ch. 9 (notes in `plans/09-discussion.md`).
- Appendix A.4 "Example generations" written: 4 papers chosen for illustration (disclosed, not
  prevalence): 168028 stronger agreement (all content 3; random among the 54 all-3 cases, seed 0);
  169281 related area, different task; 152120 analysis recast as a proposed method; 47781
  description of a recorded reference (each title cited: liu2020ddcmnet, wei2021retinal,
  albiero2020gender, zhang2020exemplar, he2016resnet — Crossref records; ResNet verified among its recorded references in
  `data/processed/joint/edges.csv`; generated numbers match neither paper). Full abstracts.

(earlier plan below)
- Do **not** quote examples in the chapter. Name the types of failure (e.g. right subfield but a
  different task; generic abstract of the field; describes a cited paper) and point to example
  generations in a new appendix (original + generated abstract, a few papers).
- Types and appendix examples to be decided from the joint model's outputs and judge scores
  (after the runs). How the appendix examples are chosen: decide then.

## Consequences
- Ch. 5 §5.6.2 "The variants are compared in Chapter 8" → Chapter 7 (the grid is now in 7.2).
- Ch. 5 §5.6.3 "assessed in Chapter 7 and Chapter 8" — keep (Ch. 8 compares joint vs GNN).
- Ch. 8 does not repeat the GNN grid.
- New appendix: example generations.

## Appendix: training resources (TODO, added 2026-11-17)
- `app:resources` in `content/appendices.tex`, red TODO only (author: mark it, write later).
- Content: GPU (1 × A100 40 GB per job), precision (bf16 frozen Qwen; fp32 trainable), effective
  batch (16; GNN 256), wall-clock per phase (reader 11.5 h ≈ W&B `_runtime` 41,269 s, so likely
  incl. validation; GNN grid ~31 min; joint 5.7 h), GPU memory and utilisation during training
  (needs W&B system metrics export). Throughput trace only with its artefact explained.
