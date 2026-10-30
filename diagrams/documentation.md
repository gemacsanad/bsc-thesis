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

## Background-chapter diagrams in draw.io (2026-10-27)

**Goal.** Schematic figures for Ch. 2 (not plots), in the same print style as the training
figures.

**Tool.** draw.io Desktop (`brew install --cask drawio`); no MCP server is needed. The `.drawio`
file is the source: open it in the app to edit. Export from the command line:

    /Applications/draw.io.app/Contents/MacOS/draw.io -x -f pdf --crop \
        -o figures/background/transformer_causal_mask.pdf figures/background/transformer_causal_mask.drawio

**Style.** Latin Modern Roman (copied from TeX Live into `~/Library/Fonts`, so draw.io can
use it; embedded in the PDF as Type 3), 12 px labels ≈ 8.7 pt in print, 1 px black strokes,
light tints of the Okabe–Ito colours as box fills, no titles inside the figure. One draw.io px ≈
0.256 mm in the exported PDF, so a 440 px wide drawing prints at ≈ 113 mm (include it without
`width=`).

**Figures.**

| file | shows | size |
|---|---|---|
| `figures/background/transformer_causal_mask.{drawio,pdf}` | (a) decoder-only Transformer, post-norm as in Vaswani et al.: token embedding → *L* × [masked multi-head self-attention, add & norm, feed-forward network, add & norm] with residual connections → linear + softmax → next-token probabilities; (b) the causal mask for 5 positions: 0 on and below the diagonal, −∞ above | 113.3 × 85.7 mm |
| `figures/background/message_passing.{drawio,pdf}` | (a) citation neighbourhood of a target paper *v*: three 1-hop references w1–w3, six 2-hop nodes, two grey nodes outside, arrows in the direction messages flow (towards *v*), 1-hop / 2-hop regions as sky / yellow tinted circles; (b) one step of Eqs. (2.1)–(2.2): h_wi and h_v → M_t → Σ → m_v → U_t (with h_v) → h_v^(t+1). Maths via draw.io MathJax (`math="1"`), which the CLI export renders as vector paths | 106.2 × 50.5 mm |
| `figures/background/prompt_vs_prefix.{drawio,pdf}` | the same frozen layer stack twice (layers 1, 2, ⋮, *L*; columns = *M* = 2 soft/prefix positions + the tokens "Summarise the document"). (a) prompt tuning: trainable **P**_e at the input only; hidden states at the soft positions computed by the frozen layers (dashed vermillion outlines). (b) prefix tuning: trainable **P**^K_ℓ, **P**^V_ℓ inside every layer, dashed arrows prefix → tokens (they enter the tokens' attention), no states at prefix positions. Key: trainable (vermillion) / frozen (grey) | 112.2 × 66.7 mm |
| `figures/background/g_retriever.{drawio,pdf}` | the G-Retriever pipeline as one left-to-right flow; the only sentence text is the question ("Where was Pierre Curie’s wife born?") and the answer ("Warsaw"), plus the module words GNN / projection / LLM. Question → a larger unlabelled graph (dashed frame) with 3 retrieved nodes (sky) and the selected connected subgraph (yellow, incl. one unselected connector node) → the cut-out subgraph → two paths: GNN → one pooled vector → projection → one soft-prompt cell (vermillion = trainable), and the subgraph as text (document icon); the LLM input is a vertical column read top to bottom: soft prompt, subgraph-text tokens, question tokens → frozen LLM (grey) → "Warsaw". No key: the caption explains the colours and the reading order. A more detailed four-step version (v1) was rejected by the author as too complicated; its generator `make_gr_fig_v1.py` is kept in the session scratchpad next to `make_gr_fig.py`. Approved by the reviewer agent (9.0/10) | 112.2 × 41.7 mm |

**Open.** No positional-encoding box in (a): Qwen3 applies RoPE inside attention, which Ch. 2
explains separately.

The message-passing figure (2026-10-28) was drawn by a generator script and reviewed by a separate reviewer pass (approved in round 1, 9/10). Known compromise: the sky tint means both the 1-hop region in (a) and U_t in (b).

The prompt-vs-prefix figure (2026-10-28) was approved in review round 2 (9/10). Round 1 was rejected because the dashed arrows pointed from tokens to prefix (read as data flowing into the prefix) and the *M* positions were unmarked (the two cells read as one K + one V).
