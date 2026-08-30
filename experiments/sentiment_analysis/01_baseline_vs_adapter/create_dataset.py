"""Build a side-by-side comparison dataset of baseline vs adapter generations.

Samples 3 nodes from each neighbour-count bin (5-10, 10-15, ..., 95-100) and
writes node_id, original_abstract, baseline_abstract, adapter_abstract.
"""

import sys
from pathlib import Path

import pandas as pd

EXPERIMENTS = Path(__file__).resolve().parents[2]
BASELINE = EXPERIMENTS / "2026-08-15-14207605_vicuna-baseline" / "generations.csv"
ADAPTER = EXPERIMENTS / "2026-08-15-14207605_adapter" / "generations.csv"
EVALUATED = EXPERIMENTS / "2026-08-15-14207605_adapter" / "evaluated_result.csv"
OUTPUT = Path(__file__).resolve().parent / "comparison_dataset.csv"

BIN_EDGES = list(range(5, 105, 5))  # [5, 10), [10, 15), ..., [95, 100]
PER_BIN = 3
SEED = 42


def _refuse_to_clobber(output):
    """Re-running a builder over a CSV the judge has written into destroys paid-for work."""
    if output.exists() and "comparison" in pd.read_csv(output, nrows=0).columns:
        if "--force" not in sys.argv:
            sys.exit(f"{output.name} already holds judge output; pass --force to overwrite.")


def main():
    _refuse_to_clobber(OUTPUT)
    baseline = pd.read_csv(BASELINE)
    adapter = pd.read_csv(ADAPTER)
    evaluated = pd.read_csv(EVALUATED)

    merged = baseline[["node_id", "original_abstract", "generated_abstract"]].merge(
        adapter[["node_id", "generated_abstract"]],
        on="node_id",
        suffixes=("_baseline", "_adapter"),
    )
    merged = merged.merge(evaluated[["node_id", "num_of_neighbors"]], on="node_id")

    # right=True on the last bin only, so nodes with exactly 100 neighbours land in [95, 100].
    merged["bin"] = pd.cut(
        merged["num_of_neighbors"], bins=BIN_EDGES, right=False, include_lowest=True
    )
    merged.loc[merged["num_of_neighbors"] == BIN_EDGES[-1], "bin"] = merged["bin"].cat.categories[-1]

    sampled = (
        merged.groupby("bin", observed=True)
        .apply(lambda g: g.sample(n=min(PER_BIN, len(g)), random_state=SEED), include_groups=False)
        .reset_index(drop=True)
    )

    sampled = sampled.rename(
        columns={
            "generated_abstract_baseline": "baseline_abstract",
            "generated_abstract_adapter": "adapter_abstract",
        }
    )
    sampled[
        ["node_id", "original_abstract", "baseline_abstract", "adapter_abstract"]
    ].to_csv(OUTPUT, index=False)

    print(f"Wrote {len(sampled)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
