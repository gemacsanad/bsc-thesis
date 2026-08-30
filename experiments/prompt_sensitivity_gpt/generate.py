"""Generate one arm's abstracts with gpt-5.6-luna.

Reads `0N_eraN/test_set.csv`, sends each row's finished prompt, and writes
`0N_eraN/generations.csv` with the four columns every run directory in this repository uses
(`node_id, mag_id, original_abstract, generated_abstract`), so
`src/evaluation/evaluate_results.py` scores it unchanged.

Held fixed across the four arms, so the prompt is the only difference:

  * the model, `gpt-5.6-luna`;
  * the decoding settings, whatever `--reasoning-effort` and `--max-completion-tokens` are
    set to on the command line — pass the same values to all four;
  * the Vicuna chat wrapper, split into a system and a user message exactly as
    `src/generation/generate_gpt.py` splits it for the archived gpt-5.4-mini baseline;
  * the node set and the references, both frozen by `build_test_set.py`.

Temperature and top_p are not sent. The archived gpt-5.4-mini baseline set them to 0.35 and
0.90, but the 5.6 models reject both, and leaving them at their defaults is closer to the
7B runs' greedy decoding than any value that would be accepted.

Resumable: a rerun appends only the node_ids missing from the output CSV. A row that errors
is written as `generated_abstract = "ERROR: ..."` and skipped by the evaluator, matching how
`src/adapter/generate.py` handles a failure.

Usage:
    conda activate DyTAG
    python experiments/prompt_sensitivity_gpt/generate.py --era 1
    python experiments/prompt_sensitivity_gpt/generate.py --era all
"""

import argparse
import csv
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = {1: "01_era1", 2: "02_era2", 3: "03_era3", 4: "04_era4"}

load_dotenv(ROOT / ".env_local")

# The Vicuna wrapper the prompts carry. Split here rather than rebuilt, so the string that
# reaches the API is provably the string in test_set.csv.
SYSTEM_PREFIX = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's "
                 "questions.")
USER_MARK, ASSISTANT_MARK = " USER: ", " ASSISTANT:"


def split_prompt(prompt):
    """-> (system, user) from a stored Vicuna-format prompt."""
    assert prompt.startswith(SYSTEM_PREFIX + USER_MARK), "prompt lost its Vicuna wrapper"
    assert prompt.endswith(ASSISTANT_MARK), "prompt lost its assistant marker"
    return SYSTEM_PREFIX, prompt[len(SYSTEM_PREFIX) + len(USER_MARK):-len(ASSISTANT_MARK)]


def complete(client, args, system, user):
    """-> the generated abstract, or an `ERROR: ...` string."""
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                # A reasoning model that spends the whole budget thinking returns an empty
                # string with finish_reason="length". Silently writing "" would look like a
                # model that had nothing to say.
                return (f"ERROR: empty completion "
                        f"(finish_reason={response.choices[0].finish_reason})")
            return text
        except RateLimitError:
            wait = 2 ** attempt
            print(f"  rate limit — retrying in {wait}s …", flush=True)
            time.sleep(wait)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
    return "ERROR: max retries exceeded"


def run_arm(client, args, era):
    arm = HERE / ARMS[era]
    test_set = arm / "test_set.csv"
    if not test_set.exists():
        print(f"era {era}: no {test_set.relative_to(HERE)} — run build_test_set.py first")
        return

    df = pd.read_csv(test_set)
    output = arm / "generations.csv"

    done, write_header = set(), True
    if output.exists():
        done = set(pd.read_csv(output)["node_id"].tolist())
        write_header = False
        print(f"era {era}: resuming — {len(done)} of {len(df)} rows already written")

    with open(output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "mag_id", "original_abstract",
                                               "generated_abstract"])
        if write_header:
            writer.writeheader()

        for i, row in df.iterrows():
            node_id = int(row["node_id"])
            if node_id in done:
                continue
            system, user = split_prompt(row["prompt"])
            print(f"[era {era}] [{i + 1}/{len(df)}] node_id={node_id} "
                  f"refs={row['refs_shown']}/{row['out_degree']} "
                  f"prompt_tokens={row['prompt_tokens']}", flush=True)

            generated = complete(client, args, system, user)
            print(f"  >> {generated[:160]}…", flush=True)

            writer.writerow({"node_id": node_id,
                             "mag_id": row["mag_id"],
                             "original_abstract": row["original_abstract"],
                             "generated_abstract": generated})
            f.flush()

    written = pd.read_csv(output)
    errors = written["generated_abstract"].astype(str).str.startswith("ERROR:").sum()
    print(f"era {era}: {len(written)} rows in {output.relative_to(ROOT)} ({errors} errors)\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--era", default="all", help="1, 2, 3, 4, or all")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--max-completion-tokens", type=int, default=2048,
                    help="covers the reasoning tokens as well as the abstract; the 150-180 "
                         "words eras 2-4 ask for are ~250 of these")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["none", "low", "medium", "high"])
    args = ap.parse_args()

    eras = [1, 2, 3, 4] if args.era == "all" else [int(args.era)]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(f"model={args.model} reasoning_effort={args.reasoning_effort} "
          f"max_completion_tokens={args.max_completion_tokens}\n")
    for era in eras:
        run_arm(client, args, era)


if __name__ == "__main__":
    main()
