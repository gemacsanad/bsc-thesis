
## Judge prompt for the children's books (`evaluate_results_scores_only.py --dataset children`)

**Why:** the run-11 children's-books arms (`experiments/2026-10-22-14618627_run11-children-*`)
need scores. The paper rubric asks about scientific objective, methods and register, which
don't apply to books. `--dataset children` swaps in a book rubric. The default stays `papers`,
so the paper scores can be reproduced exactly.

**Design.** It uses the same three output columns (`llm_accuracy`, `llm_coherence`,
`llm_language`), so the analysis code works unchanged. The judge sees only the original and the
generated text, never the arm name.

- **Accuracy:** does the text describe *this* book? The rubric lists the concrete points to
  check (subject, characters, setting, premise, age group, format, series or author) and has
  anchors from 1–2 (a different book, or nothing but praise and boilerplate) to 9–10 (the same
  book with most specifics right). Generic copy earns no credit. As in the paper rubric,
  plausible unmentioned details are not penalised and contradictions are.
- **Coherence:** one consistent description of one book. Penalises loops, repeated quotes,
  truncation and drifting between books.
- **Language:** reads like real published copy. The library-review format is explicitly allowed:
  296 of the 945 original descriptions open with a grade level ("PreS-Gr 1-…") and 318 carry a
  copyright line, so penalising that format would punish texts for matching their targets.

**Not tuned on outcomes.** The prompt was written once, before any book arm was scored. A
prompt adjusted until GNN+LLM pulls ahead of the baselines would be a biased instrument. The
aim is a rubric that separates book-specific from generic text, whichever arm produces it.

**Surface statistics of the arms**, for reading the scores:

| arm | distinct of 945 | median words | review-format opening | copyright line |
|---|---:|---:|---:|---:|
| reader ← gnn | 940 | 100 | 410 | 206 |
| reader ← refs-mean | 937 | 99 | 367 | 164 |
| reader ← zero | 1 | 432 | 945 | 945 |
| text (frozen Qwen) | 945 | 139 | 0 | 0 |
| originals | — | — | 296 | 318 |

**Run** (from the repo root; needs `OPENAI_API_KEY`):

```bash
for ARM in reader-gnn reader-refs-mean reader-zero reader-oracle text joint; do
  D=experiments/2026-10-22-14618627_run11-children-$ARM
  python src/evaluation/evaluate_results_scores_only.py --dataset children \
      --edges data/children_books/edges.csv \
      --input $D/generations.csv --output $D/scored_scores_only.csv
done
```

`num_of_neighbors` is the co-purchase degree; the books' edges are symmetric.

**Open:** the rubric has not been piloted. A check of about 20 books scored by hand against the
judge would show whether the anchors are applied as intended.

## Book judge, revision 2: aspect scores chosen where GNN+LLM differs from the baselines

**Goal (the user's):** find out which aspects of the generated descriptions the graph improves,
and make the judge measure those aspects. This replaces the "not tuned on outcomes" stance of
the previous section. The safeguard is a split: aspects are chosen on one half of the eval
books and confirmed on the other, so a difference that is noise doesn't get built into the
prompt.

**Discovery, free and local** (no judge). The 945 eval books were split at random (seed 0) into
a dev half (451) and a held-out half (494). Six lexical measures were compared against the
original description. Paired differences, z = mean / SE:

| measure | gnn − refs-mean, dev | gnn − refs-mean, held-out | joint − refs-mean, all | joint − reader-gnn, all |
|---|---|---|---|---|
| TF-IDF cosine | −0.003 (z −1.5) | +0.002 (z +0.9) | +0.013 (z +6.9) | +0.013 (z +6.5) |
| identified in TF-IDF top 10 of 945 | −1.8 pts | +2.4 pts | +6.8 pts (z +4.4) | +6.4 pts (z +4.1) |
| content-word recall | +0.001 | +0.001 | +0.017 (z +11.6) | +0.016 (z +10.1) |
| proper-noun recall | −0.001 | +0.003 | +0.022 (z +5.8) | +0.021 (z +5.7) |
| format/kind match (Jaccard) | −0.010 | +0.010 | +0.031 (z +2.3) | +0.030 (z +2.1) |
| grade-level error | +0.02 | −0.14 | −0.07 (z −0.9) | −0.01 |

- **reader ← gnn vs reader ← refs-mean: no aspect differs.** Every dev difference is under 2 SE,
  and the sign flips on the held-out half. This matches step 4's estimate that the GNN has at
  most a few hundredths of embedding cosine to gain over the mean.
- **joint (reader + GNN fine-tuned together) beats both** on book-specific content: topic
  vocabulary (+34% relative content recall), identifiability (top-10: 29.7% vs 23.0%), names and
  format. Not on grade or age level. The oracle (the book's own embedding) shows the same
  pattern, larger, so these are the aspects the vector carries.
- **Confound:** joint also had 3 × 15,000 extra steps of LM training. Without a joint ←
  refs-mean arm, part of its gain may be the extra reader training rather than the GNN.
- **The text arm** (frozen Qwen reading the neighbours as text) still beats the joint arm on
  content and format. The GNN arms' apparent lead on proper nouns over the text arm was
  review boilerplate ("School Library Journal", copyright lines). Without it, names are a tie
  (joint 0.0245, text 0.0240).

**Prompt changes.** Accuracy, coherence and language stay, on 1–10, for comparison with the
abstracts. Four aspect scores, also 1–10, are added as new columns with anchored levels:

- `llm_subject`: the specific topic and premise;
- `llm_names`: characters, author, illustrator, series, places;
- `llm_format`: the kind of book and the audience age;
- `llm_identify`: could the book be picked out from a thousand others.

The judge is told to ignore review boilerplate when judging content, since that is the
artefact behind the false proper-noun lead. Grade level is not its own score: it showed no
difference. Its age half sits inside `format`.

**Still to do:** a judge pilot on the dev half (about 100 books × 4 arms), then the full run.
Report an aspect only if its difference holds on the held-out half.

## Book judge, revision 3: the paper prompt's shape

The seven-score prompt of revision 2 was too long. The book prompt now matches the paper prompt
in style and length: a one-line role, the same three 1–10 scores (`llm_accuracy`,
`llm_coherence`, `llm_language`), and one or two sentences per rubric. The four aspect columns
are gone.

What revision 2's analysis found is now inside **accuracy**. It rewards the specifics where the
GNN+LLM arms differ from the baselines: the subject and premise; the names of characters,
author, illustrator or series; and the kind of book. Generic text scores low. Review
boilerplate is to be ignored, since that is what gave the GNN arms a false lead on names. Like
the paper rubric, it penalises explicit contradictions and leaves plausible creative details
alone. Grade level is not mentioned: it showed no difference.

The pilot files (`pilot_dev100.csv`, 100 dev-half books × 4 arms) still apply.

## Fix: runs stopping partway and restarting from zero

**Symptom:** a book run stopped at row 160. The restart then stopped at row 137, having started
again from the beginning.

**Faults in `evaluate_results_scores_only.py`:**

1. **Nothing was saved until the end.** The output CSV was written once, after the last row, so
   a stopped run left no file. The resume code, which reads cached scores from the output, had
   nothing to read, and every restart began at row 0.
2. **No request timeout.** The OpenAI client used the SDK defaults: 600 s per request and 2
   silent retries. A stalled connection could hang one row for up to 30 minutes, which looks
   like the run has stopped.
3. **Transient errors were not retried, and error rows were cached for good.** Only
   `RateLimitError` was retried. A timeout, connection error or 5xx wrote `ERROR: ...` into the
   row, and a resumed run treated that string as a score and never judged the row again.

**Fix:**

- The partial results are written every 10 judged rows.
- `timeout=60, max_retries=0`, with the script's own exponential backoff (6 attempts) now also
  covering `APITimeoutError`, `APIConnectionError` and `InternalServerError`.
- Only numeric scores are cached on resume.

The papers' scoring logic is unchanged.

**Checked** with 12 rows and an invalid key, which makes no billable calls. The run finished and
wrote the file. A rerun with 5 rows set to real scores loaded exactly those 5 and judged the
other 7 again.
