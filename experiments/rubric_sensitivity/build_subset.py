"""Freeze the node subsets every rubric is scored on.

Writes `subsets.json` with two lists:

    main     300 nodes, the baseline-vs-adapter comparison
    control  150 nodes (a subset of `main`), the judge-noise retest and the negative-
             control pair

Both are drawn from the 945 capped eval nodes, stratified by out-degree so the subset keeps
the shape of the full set rather than the shape of whatever the sampler happened to hit. Out-
degree is the axis the eval set was itself built on, and the one every earlier analysis in
`docs/adapter.md` bins by.

Selection is on out-degree alone and never on a score, so no arm's outcome influences which
nodes it is measured on — the mistake round 2 of `../sentiment_analysis/` recorded about its
own sample.

Usage:
    conda activate DyTAG
    python experiments/rubric_sensitivity/build_subset.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "adapter"))

from text import Corpus                                                     # noqa: E402

MAIN_N, CONTROL_N, SEED = 300, 150, 42


def main():
    with open(ROOT / "data" / "frozen" / "eval_nodes_capped.json") as f:
        eval_nodes = json.load(f)
    corpus = Corpus()
    degree = pd.Series({n: corpus.out_degree(n) for n in eval_nodes}, name="out_degree")

    # Ten equal-mass out-degree strata; take each one's share of MAIN_N.
    strata = pd.qcut(degree, 10, labels=False, duplicates="drop")
    rng = random.Random(SEED)
    main = []
    for s in sorted(set(strata)):
        pool = sorted(degree.index[strata == s])
        want = round(MAIN_N * len(pool) / len(degree))
        main += rng.sample(pool, min(want, len(pool)))
    # qcut rounding leaves the total a few off; settle it against the unused remainder.
    rest = sorted(set(degree.index) - set(main))
    while len(main) < MAIN_N:
        main.append(rest.pop(rng.randrange(len(rest))))
    main = sorted(main[:MAIN_N])
    control = sorted(rng.sample(main, CONTROL_N))

    out = {"main": [int(n) for n in main], "control": [int(n) for n in control],
           "seed": SEED, "source": "data/frozen/eval_nodes_capped.json"}
    with open(HERE / "subsets.json", "w") as f:
        json.dump(out, f, indent=1)

    d = degree.loc[main]
    print(f"main    {len(main)} nodes, out-degree {d.min()}-{d.max()}, "
          f"mean {d.mean():.1f}, median {d.median():.0f}")
    print(f"        full 945: mean {degree.mean():.1f}, median {degree.median():.0f}")
    print(f"control {len(control)} nodes (subset of main)")
    print(f"\nWrote {(HERE / 'subsets.json').relative_to(ROOT)}")
    print(f"\njudge calls: main 4 rubrics x 2 arms x {MAIN_N} = {4 * 2 * MAIN_N:,}")
    print(f"             retest 4 x {CONTROL_N} = {4 * CONTROL_N:,}, "
          f"negative control 4 x 2 x {CONTROL_N} = {4 * 2 * CONTROL_N:,}")
    print(f"             total {4 * 2 * MAIN_N + 4 * CONTROL_N + 4 * 2 * CONTROL_N:,}")


if __name__ == "__main__":
    main()
