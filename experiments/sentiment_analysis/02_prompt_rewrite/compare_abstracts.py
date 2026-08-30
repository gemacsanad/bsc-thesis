"""LLM diagnosis of what the prompt/decoding change did to the adapter's output.

Reads comparison_dataset.csv and, for each node, shows the judge the gold
abstract plus the two generations of the same system — old configuration as Version 1,
new as Version 2 — and asks what changed, which configuration change explains it, and
what to do about it. The answer goes into a `comparison` column of the same CSV.

The judge is blind to the accuracy scores and to the four sampling cells, so it cannot
infer which version is supposed to be better; the questions are symmetric for the same
reason. It is told what the configuration difference *is*, because attribution is the
point — but it is instructed to report "none" rather than manufacture an explanation.

Answers are returned in labelled fields so 48 of them can be counted, not just read.

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

Version 2 was produced after four changes to the system. Each leaves a different trace in the text:
  (a) INSTRUCTION REFRAMED — Version 1 was asked to write "a new application paper that builds on \
and combines these works"; Version 2 was asked to "reconstruct" the citing paper's abstract, to \
say how it "builds on and departs from" its sources, and to infer the paper's kind (method, \
application, empirical study, survey) rather than assume it. Trace: claims of novelty or of \
departing from the cited work that the sources do not support; a changed guess at what kind of \
paper this is.
  (b) BIBLIOGRAPHY SIZE DISCLOSED — Version 2's prompt states how many works the paper cites in \
total and says the shown references are "the most representative" of them. Trace: broader, more \
survey-like or more generic framing; describing the field rather than one paper.
  (c) REPETITION PENALTY — Version 2 was decoded with a penalty on repeated tokens. Trace: shorter \
output; the paper's central term named once and then avoided or replaced by vaguer wording.
  (d) LONGER GENERATION BUDGET — Version 1 was often cut off mid-sentence; Version 2 usually \
finishes. Trace: a concluding or results sentence present in Version 2 and absent in Version 1.

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
FIX: the single most useful change to the instruction that would keep what Version 2 gained while \
recovering what it lost. One sentence. Write "none needed" if Version 2 lost nothing.
RECOVERABLE: <YES | NO> — whether the gold abstract's core contribution could plausibly be \
inferred from a reference list at all, or whether it depends on the paper's own results and is \
therefore out of reach for any version.\
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
        str(df.at[idx, "old_prompt_abstract"]),
        str(df.at[idx, "new_prompt_abstract"]),
    )
    df.to_csv(args.csv, index=False)  # save after each row so interrupts don't lose work
    print("ok", flush=True)

print(f"\nWrote comparisons to {args.csv}")
