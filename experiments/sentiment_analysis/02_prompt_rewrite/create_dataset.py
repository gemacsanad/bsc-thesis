"""Build the node sample for diagnosing what the new prompt changed.

The two runs differ by the prompt rewrite, the reference-count line, the repetition
penalty, the larger generation reserve, and a retrain. Overall mean accuracy barely
moved (3.288 -> 3.259), but that is the net of two large opposite effects: the floor
rose (old 1-2, +0.81) and the top collapsed (old 5-8, -2.00).

Sampling is by four contrast cells rather than by size of movement. Droppers alone
cannot explain the collapse — the top nodes that *survived* are what identify the
cause, and the floor nodes that stayed stuck are what bound the 4-5 floor target.
"""

import sys
from pathlib import Path

import pandas as pd

EXPERIMENTS = Path(__file__).resolve().parents[2]
OLD = EXPERIMENTS / "2026-08-15-14207605_adapter"
NEW = EXPERIMENTS / "2026-08-20-14228804_adapter-gnn2"
OUTPUT = Path(__file__).resolve().parent / "comparison_dataset.csv"

SEED = 42
# (label, old-score predicate, new-score predicate, how many to sample)
CELLS = [
    ("A_collapsed_top", lambda s: s >= 6, lambda s: s <= 3, 15),
    ("B_survived_top", lambda s: s >= 6, lambda s: s >= 6, 10),
    ("C_rescued_floor", lambda s: s <= 2, lambda s: s >= 4, 12),
    ("D_stuck_floor", lambda s: s <= 2, lambda s: s <= 2, 11),
]


def _refuse_to_clobber(output):
    """Re-running a builder over a CSV the judge has written into destroys paid-for work."""
    if output.exists() and "comparison" in pd.read_csv(output, nrows=0).columns:
        if "--force" not in sys.argv:
            sys.exit(f"{output.name} already holds judge output; pass --force to overwrite.")


def main():
    _refuse_to_clobber(OUTPUT)
    old_gen = pd.read_csv(OLD / "generations.csv").set_index("node_id")
    new_gen = pd.read_csv(NEW / "generations.csv").set_index("node_id")
    old_score = pd.to_numeric(
        pd.read_csv(OLD / "evaluated_result.csv").set_index("node_id")["llm_accuracy"],
        errors="coerce")
    new_score = pd.to_numeric(
        pd.read_csv(NEW / "evaluated_result.csv").set_index("node_id")["llm_accuracy"],
        errors="coerce")

    df = pd.DataFrame({
        "original_abstract": new_gen["original_abstract"],
        "old_prompt_abstract": old_gen["generated_abstract"],
        "new_prompt_abstract": new_gen["generated_abstract"],
        "old_score": old_score,
        "new_score": new_score,
    }).dropna()

    picked = []
    for label, old_ok, new_ok, wanted in CELLS:
        cell = df[df["old_score"].map(old_ok) & df["new_score"].map(new_ok)]
        sample = cell.sample(n=min(wanted, len(cell)), random_state=SEED)
        picked.append(sample.assign(group=label))
        print(f"{label:16s} {len(cell):>4} available -> {len(sample):>3} sampled")

    out = pd.concat(picked).reset_index()
    out[["node_id", "group", "old_score", "new_score",
         "original_abstract", "old_prompt_abstract", "new_prompt_abstract"]].to_csv(
        OUTPUT, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
