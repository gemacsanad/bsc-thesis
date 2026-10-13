# Run 11 — reader ← GNN (trained separately)

**Tested:** the phase-1 reader fed the phase-2 GNN winner's estimate (GATv2, 1 layer, cosine
loss). The two were trained separately, with no joint fine-tune. Same 945 papers and decoding as the other arms.

**Why it is needed:** it is the reference point for two comparisons.
- Against reader ← refs-mean: what the GNN adds over a plain mean.
- Against joint: what joint fine-tuning adds. The two share the GNN and the papers, and the
  joint stage starts from this reader.

**Result (judged, 945 papers):** accuracy 2.98, 61.2% ≥3, coherence 7.83. Against
refs-mean: +0.113 (SE 0.037). Against joint: joint is −0.032 (SE 0.038), so the joint stage
adds nothing. TF-IDF top-1 18.5% (MRR 0.305).
