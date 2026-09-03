"""Freeze the node set every rubric is scored on.

Writes `subsets.json` with one list of 150 nodes, used for every cell: the baseline-vs-adapter
comparison and the negative-control pair alike.

The nodes are drawn from the 945 capped eval nodes, stratified by out-degree so the subset
keeps the shape of the full set rather than the shape of whatever the sampler happened to hit.
Out-degree is the axis the eval set was itself built on, and the one every earlier analysis in
`docs/adapter.md` bins by.

The draw is deliberately in two stages — 300 stratified, then 150 of those — rather than 150
stratified directly. It is the same two-stage draw the earlier 300/150 split used, at the same
seed, so these 150 nodes are exactly the ones the earlier `control` list held. That keeps
`judge_comparison/`, which was scored on the 300, valid at the new size without re-scoring a
single row.

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

POOL_N, NODES_N, SEED = 300, 150, 42


def main():
    with open(ROOT / "data" / "frozen" / "eval_nodes_capped.json") as f:
        eval_nodes = json.load(f)
    corpus = Corpus()
    degree = pd.Series({n: corpus.out_degree(n) for n in eval_nodes}, name="out_degree")

    # Ten equal-mass out-degree strata; take each one's share of POOL_N.
    strata = pd.qcut(degree, 10, labels=False, duplicates="drop")
    rng = random.Random(SEED)
    pool = []
    for s in sorted(set(strata)):
        stratum = sorted(degree.index[strata == s])
        want = round(POOL_N * len(stratum) / len(degree))
        pool += rng.sample(stratum, min(want, len(stratum)))
    # qcut rounding leaves the total a few off; settle it against the unused remainder.
    rest = sorted(set(degree.index) - set(pool))
    while len(pool) < POOL_N:
        pool.append(rest.pop(rng.randrange(len(rest))))
    pool = sorted(pool[:POOL_N])
    nodes = sorted(rng.sample(pool, NODES_N))

    out = {"nodes": [int(n) for n in nodes], "seed": SEED,
           "source": "data/frozen/eval_nodes_capped.json",
           "draw": f"{POOL_N} stratified by out-degree, then {NODES_N} of those"}
    with open(HERE / "subsets.json", "w") as f:
        json.dump(out, f, indent=1)

    d = degree.loc[nodes]
    print(f"{len(nodes)} nodes, out-degree {d.min()}-{d.max()}, "
          f"mean {d.mean():.1f}, median {d.median():.0f}")
    print(f"full 945:   mean {degree.mean():.1f}, median {degree.median():.0f}")
    print(f"\nWrote {(HERE / 'subsets.json').relative_to(ROOT)}")
    # R1 is not scored: it is the project's own rubric and judge, so the arms' stored
    # llm_accuracy already is the R1 score for every node.
    print(f"\njudge calls: 3 rubrics x 4 arms x {NODES_N} = {3 * 4 * NODES_N:,}")


if __name__ == "__main__":
    main()
