"""What the graph prefix actually changes — the analysis the four rubrics were written from.

Compares the era-1 baseline and adapter arms over all 945 capped eval nodes, without any API
call, asking what the adapter does differently rather than how much better it scores. Four
questions:

1. **Is the adapter more complete, or more precise?** Recall and precision of the gold's
   distinctive terms. Terms appearing in more than 10% of the 945 golds are dropped as
   carrying no topical signal.
2. **Does the advantage come from neighbourhood the prompt could not show?** The centroid cap
   shows at most 15 references while the GNN reads the whole neighbourhood, so if the prefix
   is an information channel the gap should grow with the number of references the cap hid.
3. **Where in the score range does the gap live?** Concentrated in the tail or spread evenly.
4. **Does the arm describe one cited work instead of the citing paper?** Bigram overlap with
   the single closest reference — the failure the qualitative examples are full of.

Building the shown-reference lists needs the Vicuna tokenizer over 945 nodes, a few minutes;
the result is cached next to this file as `shown_refs.pkl`.

Usage:
    conda activate DyTAG
    python experiments/rubric_sensitivity/analyse_arms.py
"""

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "adapter"))
sys.path.insert(0, str(HERE))

import config                                                               # noqa: E402
from text import Corpus, capped_example                                     # noqa: E402
from runs import generations                                                # noqa: E402

CACHE = HERE / "shown_refs.pkl"

# Function words plus the register boilerplate every generated abstract carries; leaving them
# in would measure how many abstracts say "we propose".
STOP = set("""the a an and or of for to in on with by is are was were be been this that these
those we our us it its from as at using use used based new novel propose proposed present
presents show shows results result method methods approach approaches paper work study can may
also which such more most than then when where while both each other between into over under
two three first second have has had do does not no there only very much many some any all
well might could would should will shall must if else""".split())


def terms(text):
    return {w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", str(text).lower()) if w not in STOP}


def bigrams(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return set(zip(words, words[1:]))


def shown_refs(corpus):
    """-> {node_id: [reference node ids the era-1 prompt actually showed]}, cached."""
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    from transformers import AutoTokenizer
    print("Building shown-reference lists (Vicuna tokenizer, 945 nodes) …")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, use_fast=True)
    embeddings = np.load(Path(config.JOINT_DIR) / "node_embeddings.npy", mmap_mode="r")
    with open(ROOT / "data" / "frozen" / "eval_nodes_capped.json") as f:
        nodes = json.load(f)
    out = {}
    for i, node_id in enumerate(nodes, start=1):
        out[node_id] = [int(r) for r in capped_example(tokenizer, corpus, embeddings, node_id)[2]]
        if i % 200 == 0:
            print(f"  {i}/{len(nodes)}", flush=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def main():
    print("Loading corpus …")
    corpus = Corpus()
    shown = shown_refs(corpus)

    gold_terms = {n: terms(corpus.abstract[n]) for n in shown}
    frequency = Counter(w for s in gold_terms.values() for w in s)
    common = {w for w, c in frequency.items() if c > 0.10 * len(gold_terms)}
    print(f"{len(frequency):,} gold term types, {len(common)} dropped as common")

    arms = {k: pd.read_csv(generations(k)).set_index("node_id") for k in ("baseline", "adapter")}
    rows = []
    for node_id in shown:
        gold = gold_terms[node_id] - common
        ref_ids = shown[node_id]
        ref_terms = set().union(*[terms(f"{corpus.title[r]} {corpus.abstract[r]}")
                                  for r in ref_ids]) if ref_ids else set()
        ref_bigrams = [bigrams(f"{corpus.title[r]} {corpus.abstract[r]}") for r in ref_ids]
        gold_bigrams = bigrams(corpus.abstract[node_id])
        row = {"node_id": node_id, "out_degree": corpus.out_degree(node_id),
               "shown": len(ref_ids), "dropped": corpus.out_degree(node_id) - len(ref_ids)}
        for arm, frame in arms.items():
            generated = frame.loc[node_id, "generated_abstract"]
            gen, gen_bg = terms(generated), bigrams(generated)
            row[f"acc_{arm}"] = pd.to_numeric(frame.loc[node_id, "llm_accuracy"], errors="coerce")
            row[f"recall_{arm}"] = len(gen & gold) / max(len(gold), 1)
            row[f"precision_{arm}"] = len(gen & gold) / max(len(gen), 1)
            row[f"nterms_{arm}"] = len(gen)
            row[f"goldbg_{arm}"] = len(gen_bg & gold_bigrams) / max(len(gen_bg), 1)
            row[f"top1ref_{arm}"] = max((len(gen_bg & b) / max(len(gen_bg), 1)
                                         for b in ref_bigrams), default=0.0)
            # Everything the arm says that is in some reference but not in the gold: the
            # size of the "wrote about the bibliography" failure.
            row[f"refonly_{arm}"] = len((gen & ref_terms) - gold) / max(len(gen), 1)
        rows.append(row)
    d = pd.DataFrame(rows)
    d["gap"] = d.acc_adapter - d.acc_baseline
    d.to_csv(HERE / "arm_analysis.csv", index=False)

    print(f"\n=== 1. complete, or precise? (n={len(d)}) ===")
    print(f"{'':24s} {'baseline':>10s} {'adapter':>10s}")
    for label, key in [("gold-term recall", "recall"), ("gold-term precision", "precision"),
                       ("distinct terms used", "nterms"), ("bigram overlap w/ gold", "goldbg"),
                       ("reference-only terms", "refonly")]:
        print(f"  {label:22s} {d[f'{key}_baseline'].mean():10.4f} {d[f'{key}_adapter'].mean():10.4f}")

    print("\n=== 2. does the gap come from references the cap hid? ===")
    d["bin"] = pd.cut(d.dropped, [-1, 0, 5, 15, 40, 200], labels=["0", "1-5", "6-15", "16-40", ">40"])
    print(d.groupby("bin", observed=True).agg(n=("gap", "size"), baseline=("acc_baseline", "mean"),
                                              adapter=("acc_adapter", "mean"),
                                              gap=("gap", "mean")).round(3).to_string())
    for column in ("dropped", "out_degree"):
        r = stats.spearmanr(d.gap, d[column])
        print(f"  spearman(gap, {column:10s}) = {r.statistic:+.3f}  p={r.pvalue:.3f}")

    print("\n=== 3. where in the range does the gap live? ===")
    for arm in ("baseline", "adapter"):
        counts = [int((d[f"acc_{arm}"] == k).sum()) for k in range(1, 11)]
        print(f"  {arm:9s} {counts}  mean {d[f'acc_{arm}'].mean():.3f}")
    for arm in ("baseline", "adapter"):
        print(f"  {arm:9s} <=2: {(d[f'acc_{arm}'] <= 2).mean():5.1%}   "
              f">=5: {(d[f'acc_{arm}'] >= 5).mean():5.1%}")

    print("\n=== 4. captured by a single cited work? ===")
    for arm in ("baseline", "adapter"):
        print(f"  {arm:9s} mean top-1 reference overlap {d[f'top1ref_{arm}'].mean():.4f}   "
              f"share over 0.30: {(d[f'top1ref_{arm}'] > 0.30).mean():5.1%}")
    print("  (the mean is confounded by generation length — the adapter writes longer text "
          "with fewer distinct terms — so the share is the readable figure of the two)")

    print(f"\nWrote {(HERE / 'arm_analysis.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
