"""Generate both conditions' abstracts with gpt-5.6-luna (a different model from the judge)."""

import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYSTEM = "You are a scientific writing assistant."
FIELDS = ["node_id", "mag_id", "original_abstract", "generated_abstract"]

load_dotenv(ROOT / ".env_local")


def complete(client, args, prompt):
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return f"ERROR: empty completion (finish_reason={response.choices[0].finish_reason})"
            return text
        except RateLimitError:
            time.sleep(2 ** attempt)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
    return "ERROR: max retries exceeded"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high"])
    ap.add_argument("--max-completion-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="send at most this many prompts")
    args = ap.parse_args()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    df = pd.read_csv(HERE / "test_set.csv")
    writers, files, done, lock = {}, [], set(), threading.Lock()
    for condition in df["condition"].unique():
        out = HERE / condition / "generations.csv"
        out.parent.mkdir(exist_ok=True)
        if out.exists():
            done |= {(condition, n) for n in pd.read_csv(out)["node_id"]}
        f = open(out, "a", newline="", encoding="utf-8")
        files.append(f)
        writers[condition] = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writers[condition].writeheader()

    todo = [r for r in df.itertuples() if (r.condition, r.node_id) not in done][:args.limit]
    print(f"model={args.model} reasoning_effort={args.reasoning_effort}; "
          f"{len(todo)} of {len(df)} prompts to send", flush=True)

    def job(r):
        return r, complete(client, args, r.prompt)

    with ThreadPoolExecutor(args.workers) as pool:
        for k, fut in enumerate(as_completed([pool.submit(job, r) for r in todo]), 1):
            r, text = fut.result()
            with lock:
                writers[r.condition].writerow({"node_id": r.node_id, "mag_id": r.mag_id,
                                               "original_abstract": r.original_abstract,
                                               "generated_abstract": text})
                for f in files:
                    f.flush()
            print(f"[{k}/{len(todo)}] {r.condition} node_id={r.node_id}: {text[:90]}", flush=True)

    for f in files:
        f.close()
    for condition in df["condition"].unique():
        g = pd.read_csv(HERE / condition / "generations.csv")
        errors = g["generated_abstract"].astype(str).str.startswith("ERROR:").sum()
        print(f"{condition}: {len(g)} rows, {errors} errors")


if __name__ == "__main__":
    main()
