"""Generate one arm's abstracts with the frozen Qwen3-14B, on the GPU.

The Qwen twin of `../prompt_sensitivity_gpt/generate.py`. Reads `0N_eraN/test_set.csv` — the
same file, byte for byte — and writes `0N_eraN/generations.csv` with the four columns every
run directory in this repository uses, so `src/evaluation/evaluate_results.py` scores it
unchanged.

What is held fixed, and against what
------------------------------------
Two comparisons have to survive this script, and they pull in different directions.

  1. *Against GPT and Vicuna*: the **experimental content** — the instruction and the
     reference block — must be the string the other two models read. It is not rebuilt here;
     it is lifted out of the stored `prompt` column by `split_prompt`, so the bytes that reach
     the tokenizer are provably the bytes in `test_set.csv`.

  2. *Against the 945-node Qwen baseline* (`experiments/<date>_qwen14b-baseline/`): the
     **calling convention** must be the one that run used. That is Qwen3's own ChatML, via
     `apply_chat_template(..., enable_thinking=False)`, and the Vicuna `SYSTEM … USER: …
     ASSISTANT:` markers the stored prompt carries are discarded.

Keeping the text and swapping the delimiters is the same split `src/adapter/text.py` makes
(`docs/qwen_migration.md`, step 5), and it is not cosmetic: a frozen Qwen3 handed non-ChatML
input is off-distribution precisely on instruction following, which is the faculty this
experiment measures. `enable_thinking=False` does not remove the reasoning block — it emits an
empty, already-closed `<think>\n\n</think>`, which is how Qwen3 is told to answer directly.

Everything else comes from `src/adapter/config.py` rather than being written down again, so
this arm cannot drift from the baseline it is compared against:

  * `BASE_MODEL` = Qwen/Qwen3-14B, bfloat16;
  * greedy decoding (`do_sample=False`) with `REPETITION_PENALTY`;
  * `MAX_NEW_TOKENS` = 320;
  * batch size 1.

There is no graph prefix here. This is the no-graph condition, the same one the baseline arm
generates, so `NUM_PREFIX_TOKENS` is *not* charged against the context.

Resumable: a rerun appends only the node_ids missing from the output CSV. A row that errors is
written as `generated_abstract = "ERROR: ..."` and skipped by the evaluator, matching
`src/adapter/generate.py`.

Usage:
    python generate.py --era 1
    python generate.py --era all
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "adapter"))          # flat imports, as in src/adapter

from config import (BASE_MODEL, MAX_LENGTH, MAX_NEW_TOKENS,  # noqa: E402
                    REPETITION_PENALTY)

ARMS = {1: "01_era1", 2: "02_era2", 3: "03_era3", 4: "04_era4"}

# The Vicuna wrapper the stored prompts carry. Split here rather than rebuilt, so the content
# that reaches the API is provably the content in test_set.csv.
SYSTEM_PREFIX = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's "
                 "questions.")
USER_MARK, ASSISTANT_MARK = " USER: ", " ASSISTANT:"


def split_prompt(prompt):
    """-> (system, user) from a stored Vicuna-format prompt."""
    assert prompt.startswith(SYSTEM_PREFIX + USER_MARK), "prompt lost its Vicuna wrapper"
    assert prompt.endswith(ASSISTANT_MARK), "prompt lost its assistant marker"
    return SYSTEM_PREFIX, prompt[len(SYSTEM_PREFIX) + len(USER_MARK):-len(ASSISTANT_MARK)]


def chatml(tokenizer, system, user):
    """-> the same two messages in Qwen3's own chat format.

    Mirrors `src/adapter/text.build_prompt`. The system message is Vicuna's, unchanged: it is
    part of what the other two models were shown, so it stays in the content even though the
    delimiters around it are now Qwen's.
    """
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def complete(llm, tokenizer, prompt, device, budget):
    """-> (generated abstract or `ERROR: ...`, prompt token count)."""
    input_ids = torch.tensor([tokenizer(prompt).input_ids], device=device)
    n_tokens = input_ids.size(1)
    if n_tokens > budget:
        return f"ERROR: prompt too long ({n_tokens} tokens > {budget} limit)", n_tokens
    try:
        out = llm.generate(input_ids,
                           max_new_tokens=MAX_NEW_TOKENS,
                           do_sample=False,
                           repetition_penalty=REPETITION_PENALTY,
                           pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0, n_tokens:], skip_special_tokens=True).strip()
        return (text or "ERROR: empty completion"), n_tokens
    except torch.cuda.OutOfMemoryError as e:
        return f"ERROR: CUDA out of memory — {e}", n_tokens
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}", n_tokens


def run_arm(llm, tokenizer, era, device, budget):
    arm = HERE / ARMS[era]
    test_set = arm / "test_set.csv"
    if not test_set.exists():
        print(f"era {era}: no {test_set.relative_to(HERE)}")
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
            prompt = chatml(tokenizer, *split_prompt(row["prompt"]))
            generated, n_tokens = complete(llm, tokenizer, prompt, device, budget)

            print(f"[era {era}] [{i + 1}/{len(df)}] node_id={node_id} "
                  f"refs={row['refs_shown']}/{row['out_degree']} "
                  f"vicuna_tokens={row['prompt_tokens']} qwen_tokens={n_tokens}", flush=True)
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
    ap.add_argument("--model", default=BASE_MODEL)
    args = ap.parse_args()

    eras = [1, 2, 3, 4] if args.era == "all" else [int(args.era)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model} in {dtype} …", flush=True)
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    llm.config.use_cache = True
    llm.eval()

    # No prefix in this condition, so nothing is held back for one — only the abstract.
    budget = MAX_LENGTH - MAX_NEW_TOKENS
    print(f"greedy, repetition_penalty={REPETITION_PENALTY}, "
          f"max_new_tokens={MAX_NEW_TOKENS}, budget={budget}\n", flush=True)

    for era in eras:
        run_arm(llm, tokenizer, era, device, budget)


if __name__ == "__main__":
    main()
