"""Score one arm's generations under one rubric.

Reads the arm's existing `evaluated_result.csv` for its `original_abstract` and
`generated_abstract` — the generations are fixed, only the judge prompt changes — and writes
`0N_RN/<arm>.csv` with the rubric's scores.

Held identical to `src/evaluation/evaluate_results.py`: the judge model (`gpt-5.4-mini`),
`temperature=0.0`, `response_format=json_object`, the user message layout
(`ORIGINAL ABSTRACT: … GENERATED ABSTRACT: …`), and the retry policy. Only the system prompt
differs, which is the variable under study — so every number here sits beside the figures in
`docs/adapter.md`.

**R1 is never sent to the API.** R1 is the project's current rubric verbatim, scored by the
project's own judge at the same temperature, so the arms' stored `llm_accuracy`,
`llm_coherence` and `llm_language` already are its scores — for all 945 nodes, paid for when
the runs were evaluated. `--rubric R1` copies those out instead of buying them again. (No
`reason` column: the main evaluator consumes it and does not persist it.)

ROUGE and BERTScore are not recomputed: they do not depend on the judge, and the arms already
carry them.

Usage:
    conda activate DyTAG
    python experiments/rubric_sensitivity/score.py --rubric R2 --arm baseline
    python experiments/rubric_sensitivity/score.py --all          # every cell, in order

Resumable: a rerun appends only the node_ids missing from the output CSV.

`--all` covers the four arms under R2/R3/R4 plus the free R1 copy. It does **not** include
`adapter_retest`, the same-text-twice cell that measures a rubric's own noise floor: at
`temperature=0` the judge reproduces itself closely enough that the cell was judged not worth
its calls. It is still reachable as `--rubric R4 --arm adapter_retest` if a rubric's gap comes
back small enough that the floor matters — R4 is the one to spend it on, since its cap at 3 is
a categorical judgement call and the likeliest of the four to wobble.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from rubrics import ARMS, R3_COMPONENTS, RUBRICS                            # noqa: E402
from runs import MAIN, NEGATIVE_CONTROL, RETEST, generations                # noqa: E402

load_dotenv(ROOT / ".env_local")

# Every arm is scored on the same 150 nodes: the two that answer the question, and the two
# that guard it. `adapter_retest` re-scores the adapter a second time under an unchanged
# rubric — identical text, so any gap between it and `adapter` is the judge's own noise. It is
# off by default (see the module docstring) but stays addressable by name.
CELLS = list(MAIN) + list(NEGATIVE_CONTROL)
OPTIONAL_CELLS = ["adapter_retest"]
SOURCE = {"adapter_retest": RETEST}

# R1 is the project's own rubric and judge, so its scores are already in the arms' CSVs.
STORED = {"accuracy": "llm_accuracy", "coherence": "llm_coherence", "language": "llm_language"}


def nodes():
    with open(HERE / "subsets.json") as f:
        return json.load(f)["nodes"]


def clean(result):
    """Flatten a judge response to CSV-safe scalars.

    NUL bytes break pandas on read-back rather than on write, and a stray dict or list under
    a key would otherwise reach the writer as a repr. Everything is coerced to str except
    the numeric scores, which stay numeric so `report.py` does not have to parse them back.
    """
    out = {}
    for key, value in result.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
        else:
            out[key] = str(value).replace("\x00", "")
    return out


def call_judge(client, args, system, original, generated, retries=6):
    user = f"ORIGINAL ABSTRACT:\n{original}\n\nGENERATED ABSTRACT:\n{generated}"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=args.temperature,
            )
            return json.loads(resp.choices[0].message.content)
        except RateLimitError:
            wait = 2 ** attempt
            print(f"    rate limit — retrying in {wait}s …", flush=True)
            time.sleep(wait)
        except Exception as e:
            return {k: f"ERROR: {e}" for k in ("accuracy", "coherence", "language", "reason")}
    return {k: "ERROR: max retries exceeded"
            for k in ("accuracy", "coherence", "language", "reason")}


def copy_stored(arm):
    """Write the R1 cell from the arm's already-paid-for scores. No API call."""
    if arm == "adapter_retest":
        # Copying the same stored column twice would report a noise floor of exactly zero by
        # construction. An R1 retest has to be bought from the API like any other.
        sys.exit("R1 comes from stored scores, so it cannot be retested by copying; "
                 "score a retest under R2/R3/R4, or re-run the main evaluator")
    source = pd.read_csv(generations(SOURCE.get(arm, arm))).set_index("node_id")
    out_dir = HERE / ARMS["R1"]
    out_dir.mkdir(exist_ok=True)
    frame = source.loc[nodes(), list(STORED.values())]
    frame.columns = list(STORED)
    frame.to_csv(out_dir / f"{arm}.csv", quoting=csv.QUOTE_ALL)
    print(f"R1 / {arm}: {len(frame)} nodes copied from stored scores (no API call)")


def run_cell(client, args, rubric, arm):
    source = pd.read_csv(generations(SOURCE.get(arm, arm))).set_index("node_id")
    out_dir = HERE / ARMS[rubric]
    out_dir.mkdir(exist_ok=True)
    output = out_dir / f"{arm}.csv"

    fields = ["node_id", "accuracy", "coherence", "language", "reason"]
    if rubric == "R3":
        fields = fields[:2] + R3_COMPONENTS + fields[2:]

    done, write_header = set(), True
    if output.exists():
        done = set(pd.read_csv(output)["node_id"].tolist())
        write_header = False

    todo = [n for n in nodes() if n not in done]
    print(f"{rubric} / {arm}: {len(todo)} to score")
    if not todo:
        return

    with open(output, "a", newline="", encoding="utf-8") as f:
        # QUOTE_ALL, not the default QUOTE_MINIMAL: a judge's free-text `reason` is arbitrary
        # model output, and under QUOTE_MINIMAL a rare character combination raises
        # "need to escape, but no escapechar set" and takes the whole run down with it —
        # which is what happened 72 rows into R1/adapter_retest. Quoting every field removes
        # the escape path entirely, and stays readable by a plain pd.read_csv.
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore",
                                quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        for i, node_id in enumerate(todo, start=1):
            row = source.loc[node_id]
            # Generation errors carry no text to judge; skipped exactly as the main
            # evaluator skips them, so the arms stay row-comparable.
            if str(row["generated_abstract"]).startswith("ERROR:"):
                result = {k: "skipped (generation error)" for k in fields[1:]}
            else:
                result = call_judge(client, args, RUBRICS[rubric],
                                    row["original_abstract"], row["generated_abstract"])
            try:
                writer.writerow({"node_id": node_id, **clean(result)})
            except (csv.Error, TypeError, ValueError) as e:
                # One unwritable row must not discard the rows already paid for. The node is
                # left out of the CSV, so a later rerun picks it up rather than resuming past
                # a silent hole.
                print(f"  node {node_id}: row not written ({type(e).__name__}: {e}) "
                      f"— left for a rerun", flush=True)
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rubric", choices=list(RUBRICS))
    ap.add_argument("--arm", choices=CELLS + OPTIONAL_CELLS)
    ap.add_argument("--all", action="store_true", help="every rubric x every cell")
    ap.add_argument("--model", default="gpt-5.4-mini",
                    help="judge model; the project's own evaluator, kept so results here sit "
                         "beside every figure in docs/adapter.md")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if not (HERE / "subsets.json").exists():
        sys.exit("subsets.json missing — run build_subset.py first")

    if args.all:
        cells = [(r, a) for r in RUBRICS for a in CELLS]
    elif args.rubric and args.arm:
        cells = [(args.rubric, args.arm)]
    else:
        sys.exit("pass --all, or both --rubric and --arm")

    client = None
    for rubric, arm in cells:
        if rubric == "R1":
            copy_stored(arm)
            continue
        client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        run_cell(client, args, rubric, arm)


if __name__ == "__main__":
    main()
