---
name: thesis-figure
description: Make a figure for the BSc thesis in thesis/ in the established print style — either a schematic diagram (draw.io, generated from Python, reviewed by a separate agent) or a data plot (matplotlib with diagrams/thesis_style.py in a notebook). Use when the author asks for a new thesis figure or diagram, or to change one.
---

# Thesis figures

Two kinds of figure, one look. Both are drawn at their printed size, in Latin Modern, with the
Okabe–Ito palette, no title inside the figure, and the explanation in the caption.

| kind | when | tool | output |
|---|---|---|---|
| **diagram** | a mechanism, pipeline or data construction (Fig. 2.1, 2.3, 2.4, 4.1) | draw.io file written by a Python generator | `diagrams/figures/<chapter>/<name>.{drawio,pdf}` |
| **plot** | numbers from the data or the runs (Fig. 4.2, 4.3, GNN curves) | matplotlib + `diagrams/thesis_style.py` in a notebook | `diagrams/figures/<chapter>/<name>.{pdf,png}` |

`<chapter>` is `background`, `data`, ... Figures are referenced from `../diagrams/` in LaTeX, so
there is one copy of each.

## Rules that hold for both

- **Ask first** (AskUserQuestion) about content choices: what the figure must show, the example,
  how much text. The author has to defend it. Answering a question is not permission to draw.
- **Simple beats complete.** The author rejected a detailed four-step G-Retriever diagram as "too
  complicated" and approved one left-to-right flow. Keep words to the minimum; prefer a small key
  or the caption over labels on every element. Then add back only what the author asks for (e.g.
  the "nodes and edges as text" label under the document icon).
- **Facts from the code or the paper**, never from memory. Write them into the spec.
- **Colour meanings are fixed across the thesis:** trainable = vermillion tint `#F6D5C2` with a
  1.5 px outline (caption says "red"); frozen = grey `#EEEEEE`; ogbn-arxiv paper / retrieved node
  = sky `#DDEFFA`; added paper = orange `#FAECCC`; selected region = yellow `#FCF9D9`. In plots:
  main series blue `#0072B2` solid, comparison grey dashed or orange dashed.
- **No CPU-heavy or GPU work** for a figure. Data plots must run on the laptop in under a few
  minutes.
- **After it is in the thesis:** build (`latexmk -xelatex -interaction=nonstopmode thesis.tex` in
  `thesis/`), render the page (`pdftoppm -f P -l P -r 80 -png thesis.pdf ...`) and look at it.
  The figure must be referenced in the text before it appears, sit near that reference, and not
  float into the next section (if it does, place it after the paragraph that introduces it, or
  ask about `\newpage`; no new packages without asking).
- **Document it:** add one row to the figure table in `diagrams/documentation.md`: file, what it
  shows, how it was made (generator / notebook), review result, printed size.

## Diagrams (draw.io)

Tooling: draw.io Desktop, CLI export
`/Applications/draw.io.app/Contents/MacOS/draw.io -x -f pdf --crop -o OUT.pdf IN.drawio`.
Latin Modern Roman is installed in `~/Library/Fonts` (no Latin Modern Mono: never use a monospace
font). MathJax works in the export (`write(..., math=True)`, labels `\(...\)`).

1. **Spec.** Write a spec file (scratchpad) like `examples/spec_enrichment.md`: output paths,
   facts (with sources), the content of each panel, what text is allowed, style, print-size check,
   "do not touch thesis/ or other figures". Look at the approved sister figures first (render
   their PDFs) and name them in the spec.
2. **Generator.** A Python script that writes the `.drawio` using `scripts/drawio_kit.py`
   (`shape`, `circ`, `text`, `edge`, `link`, `panel_label`, `write`). Copy the structure of
   `examples/make_enrichment.py` (two panels + key) or `examples/make_g_retriever.py` (one
   flow). Keep the generator next to the figure as `diagrams/figures/<chapter>/make_<name>.py`
   so it survives the session.
3. **Build and look.** `scripts/build_drawio.sh GEN.py OUT_DIR NAME PREVIEW_DIR` generates,
   exports, renders a 300 dpi PNG and prints size and fonts. Read the PNG and zoomed crops
   (`pdftoppm -x -y -W -H`). Targets:
   - width ≤ 440 px ≈ 113 mm (the caption width); height as small as is clean (≈ 40–90 mm);
   - text 9–12 px (12 px ≈ 8.7 pt), panel labels "(a)", "(b)" 11 px centred under the panel;
   - 1 px black strokes, `endArrow=block;endFill=1;endSize=4`, no shadows, gradients, rounded
     cartoon shapes or titles; arrowheads touch their targets; no label on a line.
4. **Review loop.** The drawing is done by one agent (or you) and judged by a **separate
   reviewer agent** with `review_rubric.md`, the spec, the preview, crops and the sister figures.
   Fix and re-review until it is approved (overall ≥ 9, no criterion < 8, no correctness
   issue). Usually 1–2 rounds. Record the score in `documentation.md`.
5. **Show the author** the preview before inserting it. Iterate on their feedback; keep the
   previous generator as `make_<name>_v1.py` when the design changes.
6. **Insert.** Diagrams may be included at `width=\textwidth` when the author wants them larger
   (Fig. 2.4, 4.1); otherwise at natural size. Caption below, standalone: what is shown, what the
   colours and line styles mean, which direction arrows point.

## Plots (matplotlib)

1. **Notebook** in `notebooks/eda/thesis_<topic>.ipynb` (see `thesis_enrichment.ipynb`,
   `thesis_splits.ipynb`): a first markdown cell saying which thesis section it serves, the
   inputs (files, which node sets, which embeddings: run 11 uses the raw ones) and the output
   path; a cell that **prints every number the text uses**; one markdown cell per figure with a
   **suggested caption**; then the plotting cell.
2. **Style:**
   ```python
   sys.path.insert(0, "../../diagrams"); import thesis_style as ts; ts.apply()
   fig, ax = plt.subplots(figsize=(ts.FIGWIDTH, 55 * ts.MM), layout="constrained")  # 112.5 mm
   # two panels side by side: figsize=(ts.TEXTWIDTH, 52 * ts.MM), sharey=True, titles "(a)" via set_title(..., y=-0.42)
   ...
   ts.save(fig, "<name>", outdir="../../diagrams/figures/<chapter>")
   ```
   Distributions as step lines (`ax.step(..., where="mid"/"post")`, not filled bars, so the legend
   shows lines); lower-case axis labels with units ("papers (thousands)", "recorded references");
   a frameless legend inside an empty corner or `ts.legend_below`; a long tail capped into a
   "≥ N" last bin with tick label `r"$\geq$60"`.
3. **Run** with `jupyter nbconvert --to notebook --execute --inplace <nb>` and read the PNG.
4. **Insert** at natural size, without `width=`, so the fonts stay at 8–9 pt.

## Examples in this skill

- `examples/spec_g_retriever.md`, `examples/make_g_retriever.py`: Fig. 2.4, simple pipeline,
  approved 9/10 after the detailed v1 was rejected.
- `examples/spec_enrichment.md`, `examples/make_enrichment.py`: Fig. 4.1, two panels (ruled
  tables + a schematic graph) with a two-row key, approved 9/10.

(The example generators define their own helpers inline; `scripts/drawio_kit.py` is the same
code as a module.)
