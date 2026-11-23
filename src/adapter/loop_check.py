"""Compare generations with and without n-gram repetition blocking: sentence loops and
identification.

Usage:
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
