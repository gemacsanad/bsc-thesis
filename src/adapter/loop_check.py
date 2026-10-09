"""
Run 11 — does a decoding fix stop sentence loops without costing identifiability?

Reads generations.csv files written on the same papers (generate.py --subset) and prints, per
file: the share of outputs that repeat a sentence of 5+ words, identifiability (identify.py,
ranked against the golds of the same papers), and mean length. No judge: the loop fix is a
decoding setting, and these are the two things it can change.

Run 10a measured what loops cost: 17.6% of oracle outputs looped, scoring judge accuracy 2.87
and coherence 3.8 against 3.52 and 7.7 for the rest.

`--choose N` takes exactly two files, without the fix and with --no-repeat-ngram N, and prints
the setting to use as its last line: N if the fix lowers the loop rate and costs at most
MAX_TOP1_DROP of top-1, 0 otherwise. run11_all.sh reads it, so no one has to between phases.

Usage:
    python loop_check.py <generations.csv> [<generations.csv> ...]
    python loop_check.py --choose 6 <ngram0/generations.csv> <ngram6/generations.csv>
"""

import re
import sys

import pandas as pd

from identify import ranks, summary

# Top-1 on 200 papers moves in steps of 0.5 points; two points is four papers.
MAX_TOP1_DROP = 0.02


def loops(text):
    """True if any sentence of 5+ words occurs twice."""
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", str(text))]
    long = [s for s in sentences if len(s.split()) >= 5]
    return len(long) != len(set(long))


def measure(path):
    """-> {"loops", "top1"} for one generations.csv, after printing its line."""
    df = pd.read_csv(path).sort_values("node_id")
    top = summary(ranks(df["generated_abstract"].astype(str).tolist(),
                        df["original_abstract"].astype(str).tolist()))
    rate = df["generated_abstract"].map(loops).mean()
    print(f"{path}\n  n {len(df)}  loops {rate:.1%}  "
          f"top1 {top['top1']:.1%}  mrr {top['mrr']:.3f}  "
          f"words {df['generated_abstract'].astype(str).str.split().str.len().mean():.0f}")
    return {"loops": rate, "top1": top["top1"]}


def main():
    args = sys.argv[1:]
    if args[0] != "--choose":
        for path in args:
            measure(path)
        return
    ngram, off, on = int(args[1]), measure(args[2]), measure(args[3])
    keep = on["loops"] < off["loops"] and on["top1"] >= off["top1"] - MAX_TOP1_DROP
    print(f"{'keeping' if keep else 'dropping'} --no-repeat-ngram {ngram}: loops "
          f"{off['loops']:.1%} -> {on['loops']:.1%}, top1 {off['top1']:.1%} -> {on['top1']:.1%}")
    print(ngram if keep else 0)


if __name__ == "__main__":
    main()
