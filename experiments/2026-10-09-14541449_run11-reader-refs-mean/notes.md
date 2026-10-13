# Run 11 — reader ← reference mean

**Tested:** the phase-1 reader fed the plain mean of the paper's reference embeddings. No
GNN, no training on the graph side. Same checkpoint, 945 papers and decoding as the other arms.

**Why it is needed:** it is the no-learning graph baseline. Reader ← GNN minus this arm is what
the GNN adds over simple averaging. It also separates run 11 from run 10a's refs-mean arm,
which had loops; this one has the loop fix.

**Result (judged, 945 papers):** accuracy 2.86, 55.3% ≥3, coherence 7.89. Reader ← GNN
beats it by +0.113 (SE 0.037). TF-IDF top-1 16.5% (MRR 0.271).
