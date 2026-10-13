# Run 11 — loop test, no fix

**Tested:** the phase-1 reader ← oracle on a fixed 200-paper subset (`--subset 200`) with
ordinary decoding.

**Why it is needed:** it is the control for the loop fix. Run 10a had 17.6% of outputs
repeating a sentence, and those scored coherence 3.8 against 7.7. The loop rate and top-1 here
are what the n-gram ban is judged against.

**Result:** 14.0% looping, TF-IDF top-1 52.0% (within 200 papers), 123 words on average.
