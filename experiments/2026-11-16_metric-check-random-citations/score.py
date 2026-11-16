"""Score both conditions and compare them paper by paper.

Instruments, all against the paper's original abstract:
  rougeL_f, bertscore_f1 — as in src/evaluation/evaluate_detailed.py (stemming; lang="en");
  tfidf_rank            — identify.ranks within the 200 papers of this check (chance top-1 0.5%);
  subfield, problem, task — the content-agreement call of evaluate_detailed.compare (0-3),
                          judge gpt-5.4-mini; the call's other fields are kept but not used;
  content               — the single content-agreement score: mean of subfield, problem, task.

Writes <condition>/scored.csv and summary.csv (paired real - control, for each control
condition `random` and `same_category`, on papers valid in both).

Usage (from the repository root, so .env_local is found):
    python experiments/2026-11-16_metric-check-random-citations/score.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src/evaluation"))
import evaluate_detailed as ed  # noqa: E402  (also loads .env_local from the working directory)

JUDGE = "gpt-5.4-mini"
METRICS = ["rougeL_f", "bertscore_f1", "tfidf_top1", "tfidf_rr", "content", "subfield", "problem", "task"]


def score(condition):
    out = HERE / condition / "scored.csv"
    if out.exists():
        return pd.read_csv(out)
    df = pd.read_csv(HERE / condition / "generations.csv")
    df = df[~df["generated_abstract"].astype(str).str.startswith("ERROR:")]
    df = df.sort_values("node_id").reset_index(drop=True)
    df = df.join(ed.rouge_bertscore(df)[["rougeL_f", "bertscore_f1"]])
    rank = np.asarray(ed.ranks(df["generated_abstract"].astype(str).tolist(),
                               df["original_abstract"].astype(str).tolist()))
    df["tfidf_rank"], df["tfidf_top1"], df["tfidf_rr"] = rank, (rank == 1).astype(int), 1 / rank
    with ThreadPoolExecutor(ed.WORKERS) as pool:
        judged = list(pool.map(lambda r: ed.compare(r.original_abstract, r.generated_abstract, JUDGE)
                               or {}, df.itertuples()))
    df = df.join(pd.DataFrame(judged, index=df.index))
    df["content"] = df[["subfield", "problem", "task"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=False)
    df.to_csv(out, index=False)
    return df


def main():
    ed.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    real = score("real").set_index("node_id")
    rows = []
    for control in ("random", "same_category"):
        other = score(control).set_index("node_id")
        for m in METRICS:
            both = pd.concat([real[m], other[m]], axis=1, keys=["real", "control"]).apply(
                pd.to_numeric, errors="coerce").dropna()
            d = both["real"] - both["control"]
            rows.append({"control": control, "metric": m, "papers": len(d),
                         "real": both["real"].mean(), "control_mean": both["control"].mean(),
                         "difference": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)),
                         "relative_drop": d.mean() / both["real"].mean(),
                         "real_better": (d > 0).mean(), "equal": (d == 0).mean(),
                         "control_better": (d < 0).mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(HERE / "summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
