"""Judge scores of one configuration on the thesis scale (Ch. 6 §6.4).

For every row of <run>/generations.csv, two calls to the judge (gpt-5.4-mini, temperature 0):
  content — the unchanged evaluate_detailed.compare call: subfield, problem, task (0-3);
            the call's other fields (claims, 1-10 coherence and language) are not kept;
  quality — QUALITY_SYSTEM of experiments/2026-11-16_judge-repeat-test/run.py, imported so the
            prompt is the one the repeat test measured: coherence, language (0-3).

With --reuse-content, subfield/problem/task are taken from <run>/evaluated_result_detailed.csv
(the same compare call, judge and temperature, run 2026-10-13) and only the quality call is made.

Writes <run>/judge_scores.csv (node_id, subfield, problem, task, coherence, language).
Resumable: rows already in the file are skipped; a failed call leaves the row out, so a rerun
retries it. Progress is printed every 50 papers.

Usage (from the repository root, so .env_local is found):
    python src/evaluation/judge_scores.py experiments/2026-10-09-14541449_run11-joint
"""

import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/evaluation"))
sys.path.insert(0, str(ROOT / "experiments/2026-11-16_judge-repeat-test"))
import evaluate_detailed as ed  # noqa: E402  (also loads .env_local from the working directory)
import run as repeat_test  # noqa: E402

JUDGE = repeat_test.JUDGE
FIELDS = ["node_id", "subfield", "problem", "task", "coherence", "language"]


def score(row, reused=None):
    content = reused if reused is not None else ed.compare(row.original_abstract,
                                                           row.generated_abstract, JUDGE)
    quality = repeat_test.quality(row.generated_abstract)
    if content is None or quality is None:
        return None
    return {"node_id": row.node_id, **{k: content[k] for k in ("subfield", "problem", "task")},
            **quality}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reuse-content", action="store_true",
                    help="take subfield/problem/task from evaluated_result_detailed.csv")
    args = ap.parse_args()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    ed.client = client

    gen = pd.read_csv(args.run_dir / "generations.csv")
    out = args.run_dir / "judge_scores.csv"
    done = set(pd.read_csv(out)["node_id"]) if out.exists() else set()
    todo = [r for r in gen.itertuples() if r.node_id not in done]
    reused = {}
    if args.reuse_content:
        det = pd.read_csv(args.run_dir / "evaluated_result_detailed.csv")
        reused = {r.node_id: {k: getattr(r, k) for k in ("subfield", "problem", "task")}
                  for r in det.itertuples()}
    print(f"{args.run_dir.name}: {len(todo)} of {len(gen)} papers to score", flush=True)

    with open(out, "a", newline="") as f, ThreadPoolExecutor(ed.WORKERS) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        lock, failed = threading.Lock(), 0
        for k, fut in enumerate(as_completed([pool.submit(score, r, reused.get(r.node_id)) for r in todo]), 1):
            result = fut.result()
            with lock:
                if result is None:
                    failed += 1
                else:
                    writer.writerow(result)
                    f.flush()
            if k % 250 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} scored, {failed} failed", flush=True)

    scores = pd.read_csv(out)
    print(f"{len(scores)} of {len(gen)} papers in {out.name}")
    print(scores[FIELDS[1:]].apply(pd.to_numeric, errors="coerce").mean().round(3).to_string())


if __name__ == "__main__":
    main()
