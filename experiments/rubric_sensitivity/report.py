"""Read the scored cells and rank the four rubrics against their controls.

For each rubric, three numbers matter and only the first is the headline:

    graph gap    adapter - baseline on the 300 main nodes. What the rubric was written to
                 change.
    noise floor  adapter - adapter_retest on the 150 control nodes. The same text scored
                 twice under the same rubric, so this is the judge's own wobble. A graph gap
                 is only readable to the extent it exceeds this.
    control gap  neg_a - neg_b on the same 150. Two era-4 adapter runs differing only in
                 repetition penalty. Nothing about the graph separates them, so a rubric that
                 widens this one is spreading every comparison, not detecting the prefix.

A rubric earns the headline only if the graph gap grows *and* the other two stay put. Ranking
on the graph gap alone would be choosing the instrument by the answer it gives.

Usage:
    conda activate DyTAG
    python experiments/rubric_sensitivity/report.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rubrics import ARMS, R3_COMPONENTS, RUBRICS                            # noqa: E402


def load(rubric, arm, column="accuracy"):
    path = HERE / ARMS[rubric] / f"{arm}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path).set_index("node_id")
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def paired(a, b):
    """-> (mean difference, n, p) over the nodes both arms scored."""
    if a is None or b is None:
        return None
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if j.empty:
        return None
    p = stats.wilcoxon(j.a, j.b).pvalue if (j.a - j.b).abs().sum() else 1.0
    return (j.a - j.b).mean(), len(j), p


def main():
    print(f"{'rubric':8s} {'baseline':>9s} {'adapter':>8s} {'graph gap':>20s} "
          f"{'noise floor':>12s} {'control gap':>12s}")
    for rubric in RUBRICS:
        base, adapt = load(rubric, "baseline"), load(rubric, "adapter")
        gap = paired(adapt, base)
        noise = paired(load(rubric, "adapter"), load(rubric, "adapter_retest"))
        control = paired(load(rubric, "neg_a"), load(rubric, "neg_b"))
        if gap is None:
            print(f"{rubric:8s} not scored yet")
            continue
        print(f"{rubric:8s} {base.mean():9.3f} {adapt.mean():8.3f} "
              f"{gap[0]:+8.3f} (n={gap[1]}, p={gap[2]:.1e}) "
              f"{(abs(noise[0]) if noise else float('nan')):12.3f} "
              f"{(abs(control[0]) if control else float('nan')):12.3f}")

    print("\nA rubric is only interesting if its graph gap grew while the noise floor and the "
          "control gap did not.\nR1 is the rubric every existing number in the project was "
          "produced with; it is the reference, not a candidate.")

    components = {c: (load("R3", "adapter", c), load("R3", "baseline", c))
                  for c in R3_COMPONENTS}
    if all(a is not None for a, _ in components.values()):
        print("\nR3 components — which dimension carries the effect:")
        for name, (a, b) in components.items():
            r = paired(a, b)
            print(f"  {name:9s} baseline {b.mean():.3f}  adapter {a.mean():.3f}  "
                  f"gap {r[0]:+.3f}  p={r[2]:.1e}")


if __name__ == "__main__":
    main()
