# Run 11 — joint fine-tune (GNN + adapter + LoRA)

**Tested:** phase 3. Starting from the phase-1 reader and the phase-2 GNN, it trains both
together for 3 × 15k papers. The loss is the LM loss plus 1 × (1 − cosine to the paper's own
embedding), with GNN lr 1e-5 and reader lr 1e-4. Generated from the GNN estimate.

**Why it is needed:** this is the full pipeline, and the test of whether end-to-end training beats
pieces trained apart (compare with reader ← GNN).

**Result (judged, 945 papers):**
- **Accuracy:** 2.94; 60% of papers score ≥3.
- **Coherence:** 7.86.
- **Loops:** 0%.
- **Against run 10a's refs-mean arm (2.82):** +0.125 (SE 0.039), almost all from the loop fix.
  On papers where 10a did not loop, the gain is +0.05 (SE 0.04).
- **Against reader ← GNN, TF-IDF only:** 18.8% vs 18.5% top-1.

**Caveat:** the GNN did not move during this stage. `cos_loss` stayed at 0.20, so this tests
reader adaptation to the GNN, not true end-to-end training. Details are in
`docs/run9_embedding_decoder.md` §9.
