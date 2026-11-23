# Spec: Figure "building the joint citation graph" (BSc thesis, Ch. 4 §4.2)

Two panels, draw.io, exported to PDF. Same process and style as the approved sister figures.

## Output
- `/Users/gemacsanad/egyetem/DyTAG/baseline/diagrams/figures/data/enrichment.drawio` and `.pdf`
  (create the folder `diagrams/figures/data/` if missing).
- Generator in the scratchpad: `/private/tmp/claude-501/-Users-gemacsanad-egyetem-DyTAG-baseline-thesis/875acc97-cd2f-434c-aad3-a472fdccdb5d/scratchpad/make_enrich_fig.py`,
  build script `build_enrich.sh`, preview `enrich_preview.png` (300 dpi).

## Tooling and style (binding; copy helpers from the scratchpad generators)
- Reference generators: `make_gr_fig.py` (latest, simple style the author liked), `make_pp_fig.py`,
  `make_mp_fig.py`. Approved outputs to LOOK at: `/Users/gemacsanad/egyetem/DyTAG/baseline/diagrams/figures/background/{g_retriever,message_passing,prompt_vs_prefix}.pdf`.
- draw.io CLI: `/Applications/draw.io.app/Contents/MacOS/draw.io -x -f pdf --crop -o OUT.pdf IN.drawio`.
- Latin Modern Roman (no mono font installed), 9–11 px text; width ≤ 440 px; height ≈ 45–70 mm.
  It will be included at full text width (scale ≈ 1.34), so 9 px is fine.
- Black 1 px strokes, `endArrow=block;endFill=1;endSize=4`, no shadows/gradients. Okabe–Ito tints.
- No title inside the figure; "(a)" and "(b)" centred under the panels, 11–12 px.
- The author dislikes clutter: keep words to the minimum needed to understand; prefer a small
  key over labels on every element.

## Facts (from the pipeline code; do not change)
- Semantic Scholar (S2) has three bulk datasets: **paper records** (Corpus ID + external ids incl.
  the MAG id), **abstracts** (Corpus ID → abstract), **citations** (citing Corpus ID → cited
  Corpus ID). Within S2, records are linked by the **Corpus ID**. The paper record gives the
  **MAG id**, which links to ogbn-arxiv (ogbn-arxiv papers have MAG ids).
- A citation record is translated into a MAG-id citation by mapping both endpoints' Corpus IDs to
  MAG ids through the paper records.
- Graph expansion: (1) papers cited by ogbn-arxiv papers (according to S2) are found; (2) those
  not in ogbn-arxiv are added if they have a MAG id and an abstract; (3) S2 citations whose two
  endpoints are both in the selected set are kept — this includes citations between two added
  papers and newly recorded citations between two ogbn-arxiv papers; (4) combined with the
  original ogbn-arxiv edges, duplicates removed. All original ogbn-arxiv papers and edges are
  kept, also papers with no usable S2 record. Expansion is NOT recursive: a paper cited only by an
  added paper is not added.
- Edge direction in the figure: arrow from citing paper to cited paper ("cites") — state it in
  the key or caption; be consistent.

## Panel (a): identifier matching, a worked example with ILLUSTRATIVE ids
- An ogbn-arxiv paper box: "MAG 100".
- Three small S2 tables (header + 1–2 rows), drawn as simple ruled boxes:
  - paper records: `Corpus ID | MAG` rows `700 | 100`, `900 | 200`
  - abstracts: `Corpus ID | abstract` rows `700 | …`, `900 | …`
  - citations: `citing | cited` row `700 | 900`
- Lines/arrows showing: the ogbn-arxiv paper MAG 100 ↔ paper record row 700|100 (MAG match);
  abstracts linked to paper records by Corpus ID; the citation 700 → 900 translated via the paper
  records into the result **MAG 100 → MAG 200** (a small result box, e.g. "citation: MAG 100 →
  MAG 200").
- Make it obvious which link uses the Corpus ID and which uses the MAG id (e.g. label the two
  kinds of connector lines "Corpus ID" and "MAG id", or colour them differently with a key).

## Panel (b): graph expansion (small schematic graph, unlabelled nodes)
- ~5 ogbn-arxiv papers (one style, e.g. white circles or sky tint) with their original ogbn
  citations (solid black arrows).
- ~3–4 added papers (another fill, e.g. orange tint `#FAECCC`), each cited by an ogbn-arxiv paper
  via an added citation (distinct edge style, e.g. dashed blue `#0072B2`).
- One newly recovered citation between two ogbn-arxiv papers (added-citation style).
- One citation between two added papers (added-citation style), both of which are cited by
  ogbn-arxiv papers.
- One paper cited only by an added paper, shown as NOT added (dotted grey outline, faint), with
  its dotted grey edge — to show the expansion is not recursive. Keep it subtle.
- One ogbn-arxiv paper with no added citations (it simply keeps its original edges) — just a
  normal node, no annotation needed.
- Small key under panel (b) or under both panels (no frame): ogbn-arxiv paper / added paper /
  ogbn-arxiv citation / added citation / not added. 9–10 px.

## Print-size check (mandatory)
pdfinfo width ≤ 113 mm; fonts embedded; render at 300 dpi and look at it plus crops; no overlaps,
arrowheads touching targets, labels not on lines.

## Do not
Touch anything in `/Users/gemacsanad/egyetem/DyTAG/baseline/thesis/` or other figures.
