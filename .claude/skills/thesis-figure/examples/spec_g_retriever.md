# Spec v2: Figure "the G-Retriever pipeline" — SIMPLE version (replaces v1)

The author rejected v1 as too complicated. Their words: "one diagram where the input is a question,
on the next step its visible that we select specific nodes from a bigger graph, then we construct a
subgraph from that nodes, We run a GNN on it, we send the embedding through an adapter to an LLM and
also the text through the LLM. the only text should be the question and the answer that the LLM
generates."

## Output (overwrite the v1 files; keep v1 generator as make_gr_fig_v1.py backup)
- `/Users/gemacsanad/egyetem/DyTAG/baseline/diagrams/figures/background/g_retriever.{drawio,pdf}`
- generator `.../scratchpad/make_gr_fig.py`, build `build_gr.sh`, preview `gr_preview.png`.

## One left-to-right flow (a single row, maybe with the text branch slightly below/above)
1. **Question** (the only input text): a box "Where was Pierre Curie’s wife born?".
2. **Big graph, nodes selected:** a larger graph of ~14–18 UNLABELLED small circles (no node or edge
   text) with undirected or directed thin edges. The question points at it. 3 nodes most similar to
   the question are highlighted (sky tint `#DDEFFA`, thicker outline). All other nodes white/grey.
3. **Subgraph:** the small connected subgraph built from those nodes, drawn on its own, to the
   right: the 3 highlighted nodes plus 1 white connector node with the edges between them (same
   relative positions as in the big graph, so the reader sees it is cut out). Optionally a pale
   yellow `#FCF9D9` region marks the same nodes in the big graph.
4. **Two paths from the subgraph into the LLM:**
   - **Graph path:** subgraph → GNN (vermillion `#F6D5C2`, trainable) → one embedding vector (a small
     row of cells or a single bar) → adapter/projection (vermillion) → ONE soft-prompt cell that
     enters the LLM input FIRST.
   - **Text path:** the subgraph as text, drawn as a small document/box with grey horizontal lines
     (NO actual text), plus the question → token cells (white) entering the LLM after the soft
     prompt. The question text should visibly also go to the LLM (e.g. an arrow from the question
     box, or the text icon is fed by both subgraph and question — keep it clean).
5. **LLM** (grey `#EEEEEE`, frozen) → **answer** box "Warsaw" (the only output text).

## Text allowed
- The question and the answer (Latin Modern, 10–11 px).
- Module labels are allowed and needed, as single words only: "GNN", "projection", "LLM".
  No other words: no step titles, no "node representations", no "mean pooling", no
  "top-k", no textualised graph content, no key/legend (the caption explains the colours).

## Style (unchanged, binding)
Latin Modern; black 1 px strokes; arrows `endArrow=block;endFill=1;endSize=4`; no shadows/gradients;
trainable = vermillion tint 1.5 px outline; frozen = grey; nodes as circles ~12–16 px.
Width ≤ 440 px (≤ 113 mm); height as small as is clean, target 45–65 mm. Fonts embedded. Look at a
300 dpi render; no overlaps, every arrow touching its target, calm and uncluttered.

## Content correctness (from the paper, keep)
- The question selects nodes by embedding similarity (just shown by the arrow + highlight).
- The subgraph is connected and may include a node that was not selected (the connector).
- GNN output is pooled to ONE vector; the projection maps it to ONE soft-prompt vector, placed
  before the token embeddings.
- The LLM is frozen; GNN and projection are trainable.

## Do not
Touch anything in `/Users/gemacsanad/egyetem/DyTAG/baseline/thesis/` or the other figures.
