# Run 11 — loop test, `--no-repeat-ngram 6`

**Tested:** the same reader and 200 papers as `looptest-ngram0`, decoding with no 6-word
sequence allowed to repeat.

**Why it is needed:** it decides whether the loop fix is kept, before the long evaluation arms
run. The rule, set in advance in `loop_check.py`: keep the fix if it lowers the loop rate and
costs at most 2 points of top-1.

**Result:** 0% looping, top-1 51.5% (−0.5), 107 words. Kept, so every run-11 arm uses it.
