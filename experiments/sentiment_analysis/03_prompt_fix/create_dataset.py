"""Build the node sample for diagnosing whether the prompt fix recovered the lost ceiling.

Round 2 (`../02_prompt_rewrite/report.md`) found the 2026-08-19 rewrite was redistributive:
it raised the floor by +0.81 and destroyed the top by -2.00, for no net change. Two clauses
were blamed — the genre menu and the "departs from" framing — and both were removed for the
2026-08-21 run, alongside an explicit ban on inventing specifics, a hedging rule, and an
authorial-voice clause.

The mean did not move again (3.259 -> 3.214), so the question is which of the two halves the
fix touched: did the collapsed top come back, and at whose expense?

Sampling reuses **the same 48 nodes as round 2**, not a fresh draw. Two reasons:

  * those cells are exactly the groups the fix targets — cell A is the collapse it was written
    to undo, cell D is the floor it must not damage;
  * round 2's stated limitation was that its cells were defined using the same scores the
    analysis then explained. Here the sample was fixed before the 2026-08-21 run existed, so
    the third-prompt comparison is not selected on its own outcome.

The judge sees only the two generations and the gold. The scores are carried in the CSV for
the write-up, and `score_v1` (the 2026-08-15 original prompt) is included as context: it is
the level the fix was trying to get back to.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
ROUND2 = HERE.parent / "02_prompt_rewrite" / "comparison_dataset.csv"
OUTPUT = HERE / "comparison_dataset.csv"

RUNS = {
    "v1": EXPERIMENTS / "2026-08-15-14207605_adapter",           # original prompt
    "v2": EXPERIMENTS / "2026-08-20-14228804_adapter-gnn2",      # reconstruction rewrite
    "v3": EXPERIMENTS / "2026-08-21-14230198_adapter-gnn2",      # the fix
}


def accuracy(run_dir):
    return pd.to_numeric(
        pd.read_csv(run_dir / "evaluated_result.csv").set_index("node_id")["llm_accuracy"],
        errors="coerce")


def main():
    # Re-running a builder over a CSV the judge has already written into destroys the
    # judgements, which cost API calls to produce. Refuse unless asked twice.
    if OUTPUT.exists() and "comparison" in pd.read_csv(OUTPUT, nrows=0).columns:
        if "--force" not in sys.argv:
            sys.exit(f"{OUTPUT.name} already holds judge output; pass --force to overwrite.")

    sample = pd.read_csv(ROUND2, usecols=["node_id", "group"])
    score = {k: accuracy(d) for k, d in RUNS.items()}
    gen = {k: pd.read_csv(d / "generations.csv").set_index("node_id")
           for k, d in RUNS.items()}

    out = pd.DataFrame({
        "node_id": sample["node_id"],
        "group": sample["group"],
        "score_v1": sample["node_id"].map(score["v1"]),
        "score_v2": sample["node_id"].map(score["v2"]),
        "score_v3": sample["node_id"].map(score["v3"]),
        "original_abstract": sample["node_id"].map(gen["v3"]["original_abstract"]),
        "prompt_v2_abstract": sample["node_id"].map(gen["v2"]["generated_abstract"]),
        "prompt_v3_abstract": sample["node_id"].map(gen["v3"]["generated_abstract"]),
    })
    assert out.notna().all().all(), "a sampled node is missing from one of the three runs"

    print(f"{'cell':16s} {'n':>3}  {'v1':>5} {'v2':>5} {'v3':>5}   v2->v3")
    for label, cell in out.groupby("group"):
        m = cell[["score_v1", "score_v2", "score_v3"]].mean()
        print(f"{label:16s} {len(cell):>3}  {m.score_v1:5.2f} {m.score_v2:5.2f} "
              f"{m.score_v3:5.2f}   {m.score_v3 - m.score_v2:+.2f}")

    out.to_csv(OUTPUT, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
