"""LLM comparison of adapter vs baseline generations against the gold abstract.

Reads comparison_dataset.csv, asks the judge to compare Generation B (adapter)
to Generation A (baseline) with the original abstract as reference, and writes
the free-text answer back into a `comparison` column of the same CSV.

Requires OPENAI_API_KEY in .env_local (or environment).

Usage:
    python compare_abstracts.py [--csv comparison_dataset.csv] [--model gpt-5.4-mini]
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env_local")

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=str(Path(__file__).resolve().parent / "comparison_dataset.csv"))
parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model for the comparison judge")
args = parser.parse_args()

_SYSTEM = """\
You are analyzing outputs from an abstract-generation system by comparing them against the real \
(gold) abstract. You will be shown the gold abstract and two candidate generations, both produced \
from the same citation context. Your job is to identify concretely what Generation B did \
differently from Generation A relative to the gold abstract — not to grade either one in isolation.\
"""

_USER_TEMPLATE = """\
Gold abstract (ground truth):
{gold_abstract}

Generation A:
{generation_a}

Generation B:
{generation_b}

Compare Generation B to Generation A, using the gold abstract as the reference point. In 3-5 \
sentences, answer:
- What did Generation B get closer to the gold abstract on, compared to Generation A? Be specific \
— e.g. correctly identifying the task/problem being solved, naming the right methodology or \
technique family, correctly framing the motivating gap or limitation, using more accurate domain \
terminology, or something else. Quote or closely paraphrase the specific phrases that show this.
- What did Generation B still get wrong or miss, that the gold abstract contains? Note this even \
if Generation B is otherwise an improvement.
- If Generation B is not actually closer to gold than Generation A (same quality, worse, or just \
different wording with no real gain), say so plainly instead of manufacturing an improvement.

Ground every claim in specific text from the abstracts. Do not speculate about why the two \
generations differ (e.g. do not guess what internal information one system had access to) — only \
describe what changed and whether it moved toward or away from the gold abstract's actual content.\
"""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_judge(gold: str, generation_a: str, generation_b: str, retries: int = 6) -> str:
    user_msg = _USER_TEMPLATE.format(
        gold_abstract=gold, generation_a=generation_a, generation_b=generation_b
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            wait = 2 ** attempt
            print(f"    Rate limit — retrying in {wait}s …", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"    LLM error: {e}", flush=True)
            return f"ERROR: {e}"
    return "ERROR: max retries exceeded"


df = pd.read_csv(args.csv)
if "comparison" not in df.columns:
    df["comparison"] = pd.NA

todo = df["comparison"].isna() | df["comparison"].astype(str).str.startswith("ERROR:")
print(f"Rows: {len(df)} total | {todo.sum()} to judge | {(~todo).sum()} already done")

done = 0
for idx in df.index[todo]:
    done += 1
    print(f"  [{done}/{todo.sum()}] node_id={df.at[idx, 'node_id']} …", end=" ", flush=True)
    df.at[idx, "comparison"] = call_judge(
        str(df.at[idx, "original_abstract"]),
        str(df.at[idx, "baseline_abstract"]),
        str(df.at[idx, "adapter_abstract"]),
    )
    df.to_csv(args.csv, index=False)  # save after each row so interrupts don't lose work
    print("ok", flush=True)

print(f"\nWrote comparisons to {args.csv}")
