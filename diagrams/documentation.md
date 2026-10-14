# Diagrams

## W&B export (2026-10-14)

**Goal.** Thesis figures are drawn from local copies of the training logs, so plotting does not
call the W&B API and the figures do not depend on W&B staying available.

**What is exported.** Only the run-11 runs (SLURM job 14541449), all ten of them, from the
W&B project `dytag-adapter` (38 runs). They cover the final model and its ablations:

| run | what it is | rows |
|---|---|---|
| `run11-reader-14541449` | phase 1, reader (oracle/refs-mix input, 6 epochs) | 568 |
| `run11-joint-14541449` | phase 3, joint fine-tune (LM + cosine loss) | 284 |
| `run11-gnn-{gatv2,sage}-l{1,2}-{cos,cosnce}-14541449` | phase 2, the 8-variant GNN grid | 31 each |

**How to run.** From the repository root, with `WANDB_KEY` in `.env_local` (conda env `DyTAG`
has wandb 0.28.0):

    python diagrams/export_wandb.py

**Output.** `diagrams/data/<run name>/` with `history.csv` (every logged step, from
`scan_history`, not the downsampled `history()`), `config.json`, `summary.json` and `meta.json`
(id, URL, state, creation time), plus `histograms.json` for the reader and joint runs. 1.1 MB.

**Reading `history.csv`.** One row per logged step, columns sorted. The metrics are logged at
different rates, so most cells are empty. Drop NaNs per column before plotting.

- Reader/joint: `train/*`, `gate/*`, `prefix/*` and `encoder/output_cosine` every 10 optimiser
  steps (reader 562 points, joint 281). `valid/*` and `canary/distinct` every 250 steps (28 / 14
  points). `val_id/*`, `valid/best` and `valid/bad_epochs` once per epoch (6 / 3 points).
  Joint has `train/cos_loss`, and `val_id/gnn_*` instead of `val_id/oracle_*`.
- GNN grid: `pretrain/*` once per epoch (30 points), plus one row of `pretrain/refs_mean_*`
  (the plain-reference-mean baseline on the validation papers).

**Histograms.** `gate/per_layer` (the 80 prefix gates) is a W&B histogram, so it is kept apart from `history.csv` in `histograms.json`: one entry per
logged step with `packedBins` (`min`, bin `size`, `count` = 64) and the 64 bin `values`. The
exact per-gate values are not recoverable, but the bins are narrow (≈0.01 in run 11), enough for
a gate-distribution-over-training heatmap. Read the checkpoint if exact values are needed.

## Thesis training figures (2026-10-14)

**Goal.** Redraw the run-11 W&B charts as print-ready thesis figures, one diagram per file.

**Format decisions**, following common publication guidance (size figures at their print width
so LaTeX never rescales them; no text below 8 pt; vector output; the document's typeface):
- matplotlib, PDF for LaTeX plus a 300-dpi PNG preview.
- Default size 112.5 × 70 mm = 0.75 `\textwidth`. That is the caption width the BME VIK template
  sets (`\textwidth` = 150 mm from A4 with 35/25 mm margins). Include figures without `width=`.
  Each notebook sets the size in one line: `ts.TEXTWIDTH` for full width, `ts.HALFWIDTH`
  (72 mm) for two subfigures side by side.
- Latin Modern (the thesis font) via LaTeX, 9 pt labels, 8 pt ticks and legends.
- Okabe–Ito colour-blind-safe palette.
- No titles inside figures; each notebook cell has a suggested caption instead.
- Legends sit below the plot area, so they never cover data.
- Shared settings live in `thesis_style.py`.

**Notebooks.**

| notebook | figures | where they go |
|---|---|---|
| `data/run11-gnn-*/figures.ipynb` (8 identical copies) | 11 per variant: validation and training cosine, objective, top-1, top-5, median rank, output cosine, InfoNCE, in-batch accuracy, VICReg variance and covariance | `data/<run>/figures/` |
| `data/run11-reader-*/figures.ipynb`, `data/run11-joint-*/figures.ipynb` (identical) | 13 (reader) / 14 (joint): LM loss with validation on/off, loss reduction, gate magnitude, gate distribution, identification top-1 and MRR, prefix amplitude, prefix contrast, canary, input cosine, GNN cosine loss (joint only), learning rate, gradient norm, throughput | `data/<run>/figures/` |
| `gnn_comparison.ipynb` | the 11 GNN figures, with every variant switched on in the `INCLUDE` dict at the top | `figures/gnn_comparison/` |

Each copied notebook reads the run name from its own folder, so the copies are byte-identical.

**What the figures encode.**
- Reference-mean baseline: dashed (single run) or grey dotted (comparison).
- Selected epoch: a ring, following the rule the training code uses:
  - GNN: best validation cosine;
  - reader: best `val_id` MRR of the reference-mean arm;
  - joint: best `val_id` MRR of the GNN arm.
- In the comparison, colour = convolution × depth and line style = objective.
- Step plots have dotted epoch boundaries and an epoch axis on top.

**Results the notebooks print.** Selected epochs and validation cosines of the eight GNNs
(reference mean 0.8156, top-1 63.5%, on the validation papers):

| variant | epoch | cosine | top-1 |
|---|---|---|---|
| gatv2-l1-cos | 28 | 0.8280 | 70.5% |
| gatv2-l1-cosnce | 16 | 0.8203 | 72.3% |
| gatv2-l2-cos | 29 | 0.8219 | 64.4% |
| gatv2-l2-cosnce | 26 | 0.8142 | 71.0% |
| sage-l1-cos | 23 | 0.8199 | 67.2% |
| sage-l1-cosnce | 29 | 0.8180 | 72.0% |
| sage-l2-cos | 28 | 0.8191 | 66.0% |
| sage-l2-cosnce | 26 | 0.8097 | 70.6% |

Reader: selected epoch 5 (reference-mean MRR 0.530). Joint: epoch 3 (GNN MRR 0.643). Every gate
histogram holds exactly 80 values.

**Known limitations.**
- The gate distribution is reconstructed from W&B's 64-bin histograms (bins about 0.01 wide),
  not from exact values.
- The throughput spikes after each validation are a measurement artefact: the timer restarts,
  the token count does not. The axis is clipped at 1.5× the median.
- `valid/best` (a duplicate of one `val_id` MRR) and `valid/bad_epochs` have no figure.
