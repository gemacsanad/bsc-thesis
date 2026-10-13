# Run 11 — reader ← oracle (own embedding)

**Tested:** the phase-1 reader fed the paper's own embedding, the vector the graph arms try to
estimate. Not achievable at test time.

**Why it is needed:** it is the ceiling of a single-vector reader. Oracle minus any graph arm
is the loss from the estimate rather than the reader, the pipeline's main bottleneck. Run
10a's oracle (3.41) had 17.6% looping outputs, so a looping-free ceiling is needed for the
run-11 table.

**Status:** not judged; judging it is recommended. TF-IDF top-1 is 27.8% (MRR 0.408), lower than run
10a's 32.6%, so this reader is a weaker sample of the same recipe.
