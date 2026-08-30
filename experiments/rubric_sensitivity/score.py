"""Score one arm's generations under one rubric.

Reads the arm's existing `evaluated_result.csv` for its `original_abstract` and
`generated_abstract` — the generations are fixed, only the judge changes — and writes
`0N_RN/<arm>.csv` with the rubric's scores.

Held identical to `src/evaluation/evaluate_results.py`: `response_format=json_object`, the
user message layout (`ORIGINAL ABSTRACT: … GENERATED ABSTRACT: …`), and the retry policy.
Only the system prompt differs, which is the variable under study.

The judge, however, is **not** the main evaluator's. It is `gpt-5.6-luna`, chosen for cost.
Two consequences, both real:

  * `temperature` cannot be sent — the 5.6 models accept only the default of 1 and reject
    `temperature=0.0` with a 400. The judge is therefore **not deterministic**: measured
    per-item SD of the accuracy score is 0.44 against gpt-5.4-mini's 0.00, so a rerun does
    not reproduce a previous run row for row. Over 300 nodes that noise averages to a
    standard error of about 0.026 on the mean, which is far below any gap under study, but
    it does mean no number here is exactly reproducible and per-node claims are noisier than
    the project's other CSVs. The noise is worst at the top of the range, which is where the
    adapter's advantage sits, so the adapter arm is the noisier of the two.
  * The levels are not the main evaluator's levels. Nothing scored here can be compared with
    a figure in `docs/adapter.md`; only the cells in this folder compare with each other.

`--model gpt-5.4-mini --temperature 0` reverts both properties if a run needs to be tied back
to the project's existing numbers.

ROUGE and BERTScore are not recomputed: they do not depend on the judge, and the arms already
carry them.

Usage:
    conda activate DyTAG
    python experiments/rubric_sensitivity/score.py --rubric R1 --arm baseline
    python experiments/rubric_sensitivity/score.py --all          # every cell, in order

Resumable: a rerun appends only the node_ids missing from the output CSV.
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

# Which arms each rubric is scored over, and on which subset. `adapter_retest` re-scores the
# adapter a second time under an unchanged rubric: identical text, so whatever gap appears
# between it and `adapter` is the judge's own noise, and no measured effect smaller than that
# means anything.
CELLS = [(a, "main") for a in MAIN] + \
        [("adapter_retest", "control")] + \
        [(a, "control") for a in NEGATIVE_CONTROL]
SOURCE = {"adapter_retest": RETEST}


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
    # Reasoning models take reasoning_effort and reject temperature; the older ones the
    # reverse. Sending the wrong one is a 400, not a silently different judge.
    tuning = ({"reasoning_effort": args.reasoning_effort} if args.temperature is None
              else {"temperature": args.temperature})
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                **tuning,
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


def run_cell(client, args, rubric, arm, subset_key):
    with open(HERE / "subsets.json") as f:
        nodes = json.load(f)[subset_key]

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

    todo = [n for n in nodes if n not in done]
    print(f"{rubric} / {arm} ({subset_key}, {len(nodes)} nodes): {len(todo)} to score")
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
    ap.add_argument("--arm")
    ap.add_argument("--all", action="store_true", help="every rubric x every cell")
    ap.add_argument("--model", default="gpt-5.4-mini",
                    help="judge model; the project's own evaluator, kept so results here sit "
                         "beside every figure in docs/adapter.md")
    ap.add_argument("--reasoning-effort", default="low",
                    choices=["none", "low", "medium", "high"],
                    help="reasoning models only (--model gpt-5.6-*); ignored otherwise")
    ap.add_argument("--temperature", type=lambda s: None if s == "" else float(s), default=0.0,
                    help="non-reasoning models only; the 5.6 models reject it, so pass "
                         "--temperature '' with those")
    args = ap.parse_args()
    if args.model.startswith("gpt-5.6"):
        if args.temperature is not None:
            sys.exit(f"{args.model} accepts only the default temperature — pass "
                     f"--temperature '' to use --reasoning-effort instead")
    elif args.temperature is None:
        sys.exit(f"{args.model} needs a --temperature; 0 is what the main evaluator uses")

    if not (HERE / "subsets.json").exists():
        sys.exit("subsets.json missing — run build_subset.py first")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if args.all:
        cells = [(r, a, s) for r in RUBRICS for a, s in CELLS]
    elif args.rubric and args.arm:
        subset = dict((a, s) for a, s in CELLS)[args.arm]
        cells = [(args.rubric, args.arm, subset)]
    else:
        sys.exit("pass --all, or both --rubric and --arm")

    for rubric, arm, subset in cells:
        run_cell(client, args, rubric, arm, subset)


if __name__ == "__main__":
    main()
