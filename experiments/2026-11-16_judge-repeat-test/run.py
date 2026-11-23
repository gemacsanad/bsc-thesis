"""
Repeat test of the LLM judge: score the same outputs three times and measure the variation.
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
GENERATIONS = ROOT / "experiments/2026-10-09-14541449_run11-reader-gnn/generations.csv"
N_PAPERS, REPEATS, SEED = 100, 3, 0
ASPECTS = ["subfield", "problem", "task", "coherence", "language"]

QUALITY_SYSTEM = """\
You are a rigorous scientific peer reviewer. Rate the writing of the abstract below, without \
judging whether its content is correct. Return valid JSON with exactly this structure:
{
  "coherence": <int 0-3>,
  "language":  <int 0-3>
}

- coherence: does the abstract form one logical argument?
  0 = no recognisable argument; 1 = a list of loosely connected statements; \
2 = a clear argument with gaps or jumps; 3 = a logical progression from the problem to the \
approach to the results.
- language: does the text follow the norms of scientific writing?
  0 = not scientific prose; 1 = frequent generic or conversational phrasing; \
2 = scientific register with some vague or imprecise terms; 3 = precise domain terminology \
throughout, with no generic filler."""


def quality(gen):
    r = ed.call(QUALITY_SYSTEM, f"ABSTRACT:\n{gen}", JUDGE)
    return None if r is None else {"coherence": r.get("coherence"), "language": r.get("language")}


def score(row):
    content = ed.compare(row.original_abstract, row.generated_abstract, JUDGE)
    qual = quality(row.generated_abstract)
    if content is None or qual is None:
        return None
    return {"node_id": row.node_id, "repeat": row.repeat,
            **{k: content[k] for k in ("subfield", "problem", "task")}, **qual}


def main():
    ed.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    g = pd.read_csv(GENERATIONS)
    sample = sorted(np.random.default_rng(SEED).choice(g["node_id"], N_PAPERS, replace=False))
    g = g.set_index("node_id").loc[sample].reset_index()
    jobs = pd.concat([g.assign(repeat=k) for k in range(1, REPEATS + 1)], ignore_index=True)

    out = HERE / "scores.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["node_id", "repeat"])
    keys = set(zip(done["node_id"], done["repeat"]))
    todo = [r for r in jobs.itertuples() if (r.node_id, r.repeat) not in keys]
    print(f"{len(todo)} of {len(jobs)} scorings to run", flush=True)
    with ThreadPoolExecutor(ed.WORKERS) as pool:
        new = [r for r in pool.map(score, todo) if r is not None]
    scores = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
    scores[ASPECTS] = scores[ASPECTS].apply(pd.to_numeric, errors="coerce")
    scores.sort_values(["node_id", "repeat"]).to_csv(out, index=False)

    rows = []
    for a in ASPECTS:
        wide = scores.pivot(index="node_id", columns="repeat", values=a).dropna()
        rows.append({"aspect": a, "papers": len(wide),
                     **{f"mean_repeat{k}": wide[k].mean() for k in wide.columns},
                     "mean_per_paper_sd": wide.std(axis=1, ddof=1).mean(),
                     "identical_all_repeats": (wide.nunique(axis=1) == 1).mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(HERE / "summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
