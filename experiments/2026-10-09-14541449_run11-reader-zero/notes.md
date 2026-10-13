# Run 11 — reader ← zero vector

**Tested:** the phase-1 reader with an all-zero vector in place of the graph estimate. Same
checkpoint, papers (945) and decoding (`--no-repeat-ngram 6`) as the other run-11 arms.

**Why it is needed:** it is the floor. It shows what the reader writes from the prompt and its
LoRA alone, so any gain in the other arms can be credited to the vector, not to fine-tuning.

**Status:** not judged. TF-IDF top-1 is 0.1% (MRR 0.008), so the reader writes nothing
paper-specific without the vector. Judging it is optional.
