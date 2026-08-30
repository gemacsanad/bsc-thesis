"""LLM diagnosis of what the prompt fix did to the adapter's output.

Same design as round 2 (`../02_prompt_rewrite/compare_abstracts.py`): the judge sees the gold
abstract and two generations of the same system, is blind to the accuracy scores and to the
sampling cells, and is asked symmetric questions so it cannot infer which version is meant to
be better. Only the four configuration fingerprints differ, because the change under test is a
different one — the 2026-08-20 instruction rewrite, applied on top of the run round 2 judged.

Version 1 is the 2026-08-20 run (the rewrite that collapsed the top), Version 2 the 2026-08-21
run (the fix). Answers go into a `comparison` column of the same CSV.

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

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env_local")

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=str(Path(__file__).resolve().parent
                                         / "comparison_dataset.csv"))
parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model for the comparison judge")
args = parser.parse_args()

_SYSTEM = """\
You are diagnosing a configuration change to a scientific abstract-generation system. The system \
is given the titles and abstracts of the papers a target paper cites — never the target paper \
itself — and must write that paper's abstract. You will see the target paper's real (gold) \
abstract and two generations of it, produced by the same system before and after a configuration \
change.

Neither version is known to be better. On some papers the change helped, on others it hurt, on \
others it did nothing, and your job is to say which of those happened here and why. Do not assume \
the later version is an improvement, and do not manufacture a difference where the two are \
equivalent. Judge only against the gold abstract's actual content, and ground every claim in \
specific words from the texts you are shown.\
"""

_USER_TEMPLATE = """\
GOLD ABSTRACT (the target paper's real abstract):
{gold_abstract}

VERSION 1:
{generation_a}

VERSION 2:
{generation_b}

The two versions were produced by the same model, decoded identically, from prompts that differ \
only in the instruction. Version 1's instruction asked the model to reconstruct the missing \
abstract, to say how the paper "builds on and departs from" the work it cites, and to infer what \
kind of paper it is from a list of options (a new method, an application, an empirical study, or \
a survey). Version 2 made four changes to that instruction. Each leaves a different trace:
  (a) GENRE MENU REMOVED — Version 2 is no longer offered a list of paper kinds to choose from. \
Trace: a changed guess at what kind of paper this is — one version framing it as a survey or \
review ("we review", "a comprehensive investigation of") where the other frames it as a single \
piece of work, or the reverse.
  (b) INVENTION FORBIDDEN — "departs from" was dropped and Version 2 was told not to name a \
method, model, system, or dataset that does not appear in the cited works. Trace: a named \
contribution, acronym, system or benchmark that appears in one version and is supported neither \
by the gold abstract nor by anything the cited works would contain; claims of novelty or of \
superseding the cited work.
  (c) HEDGING RULE — Version 2 was told that where the sources do not determine the paper's \
specific contribution it should state what they do determine about the problem and approach and \
stop, because a correct general statement is worth more than a specific guess. Trace: one version \
describing the shared problem area in general terms where the other commits to a particular \
technique or result.
  (d) AUTHORIAL VOICE — Version 2 was told to write as the paper's own authors and never to \
mention the cited works as a collection, how many there are, or "the authors". Trace: phrases \
like "the authors", "these works", "the papers reviewed here", or references to the size of the \
bibliography, present in one version and absent in the other.

Answer in exactly these labelled fields, nothing else:

CLOSER: <V1 | V2 | EQUIVALENT> — which version is closer to the gold abstract on what the paper is \
actually about: its problem, its method family, and its specific contribution. Weigh those three \
above fluency, length or polish. Say EQUIVALENT when the difference is only wording.
GAINED: what Version 2 gets right about the gold that Version 1 does not. Quote the phrases. \
Write "nothing" if there is no gain.
LOST: what Version 1 gets right about the gold that Version 2 does not. Quote the phrases. Write \
"nothing" if there is no loss.
CAUSE: which of (a), (b), (c), (d) the difference is visible evidence of — one or more letters \
with a one-clause justification each, quoting the text that shows it. Write "none" if the two \
versions differ in ways none of these explain, or if they do not meaningfully differ.
INVENTED: <V1 | V2 | BOTH | NEITHER> — in which version, if either, the model names a specific \
method, model, system or dataset that the gold abstract does not support. Name it. This tests \
whether change (b) took effect at all.
SPECIFICITY: <V1 | V2 | EQUIVALENT> — which version commits harder to a specific contribution, \
regardless of whether it is right. This separates a fix that made the model hedge from one that \
made it accurate.
FIX: the single most useful change to the instruction that would keep what Version 2 gained while \
recovering what it lost. One sentence. Write "none needed" if Version 2 lost nothing.\
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
        str(df.at[idx, "prompt_v2_abstract"]),
        str(df.at[idx, "prompt_v3_abstract"]),
    )
    df.to_csv(args.csv, index=False)  # save after each row so interrupts don't lose work
    print("ok", flush=True)

print(f"\nWrote comparisons to {args.csv}")
