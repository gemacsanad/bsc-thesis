"""CPU pre-flight for the four Qwen prompt-sensitivity arms. Run it before the GPU hours.

Every failure this guards against is silent — a prompt that lost its wrapper, an arm that
reads a different bibliography, a `<think>` block that was never suppressed — and all of them
produce a fluent abstract rather than an error. None would be visible in `generations.csv`.

Five checks:

1. **The stored prompts still split.** All four arms carry the Vicuna `SYSTEM … USER: …
   ASSISTANT:` wrapper `generate.py` strips before re-wrapping in ChatML.

2. **Era 1 is the 945-node baseline's prompt.** Era 1's instruction and block framing are the
   ones in the working tree's `src/adapter/text.py`, and its reference lists are identical to
   `data/frozen/prompt_refs_capped.json` on all 48 nodes — so the era-1 user message rebuilt
   through `text.user_message` must equal the one stored in `01_era1/test_set.csv`. This is
   what makes the era-1 arm a *reproduction* of `experiments/<date>_qwen14b-baseline/` on
   these 48 nodes rather than a fifth condition, and it is the only tie between this folder
   and the run it is compared against.

3. **The ChatML wrapper is Qwen3's, with thinking off.** The re-wrapped prompt must open with
   `<|im_start|>system`, end with an empty, already-closed `<think>\n\n</think>`, and carry no
   BOS — Qwen3 has none.

4. **Nothing overflows.** Under the Qwen tokenizer every re-wrapped prompt must fit
   `MAX_LENGTH - MAX_NEW_TOKENS`. The stored prompts were capped under the *Vicuna* tokenizer
   against a budget that also held back 10 prefix tokens; Qwen's BPE is more efficient and
   this condition has no prefix, so there should be room to spare. The margin is reported
   rather than assumed.

5. **The 48 nodes are in the frozen capped eval set**, so every one of them has a counterpart
   row in the baseline CSV.

Usage:
    python verify_prompts.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "adapter"))          # flat imports, as in src/adapter

from config import BASE_MODEL, FROZEN_DIR, MAX_LENGTH, MAX_NEW_TOKENS  # noqa: E402
import text                                                            # noqa: E402

sys.path.insert(0, str(HERE))
from generate import ARMS, chatml, split_prompt                        # noqa: E402


def main():
    failures = []

    print(f"Loading tokenizer {BASE_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    budget = MAX_LENGTH - MAX_NEW_TOKENS

    arms = {}
    for era, name in ARMS.items():
        df = pd.read_csv(HERE / name / "test_set.csv")
        arms[era] = df
        print(f"era {era}: {len(df)} nodes")

    # --- 1. the stored prompts still split ---------------------------------
    for era, df in arms.items():
        for _, row in df.iterrows():
            try:
                split_prompt(row["prompt"])
            except AssertionError as e:
                failures.append(f"era {era} node {row['node_id']}: {e}")
    print(f"\n[1] Vicuna wrapper splits on all {sum(len(d) for d in arms.values())} prompts: "
          f"{'ok' if not failures else 'FAILED'}")

    # --- 2. era 1 is the 945-node baseline's prompt -------------------------
    print("\n[2] rebuilding the era-1 user message through src/adapter/text.py …")
    corpus = text.Corpus()
    frozen = text.frozen_refs()
    mismatched = []
    for _, row in arms[1].iterrows():
        node_id = int(row["node_id"])
        _, body = split_prompt(row["prompt"])
        rebuilt = text.user_message([(corpus.title[n], corpus.abstract[n])
                                     for n in frozen[node_id]])
        if body != rebuilt:
            mismatched.append(node_id)
    if mismatched:
        failures.append(f"era 1 differs from text.user_message on {len(mismatched)} nodes: "
                        f"{mismatched[:5]}")
    print(f"    era 1 == text.user_message(prompt_refs_capped) on "
          f"{len(arms[1]) - len(mismatched)}/{len(arms[1])} nodes: "
          f"{'ok' if not mismatched else 'FAILED'}")

    # The other three arms are *expected* to differ — a longer instruction buys fewer
    # references — so the drift is reported, not asserted. It is the run's main caveat.
    for era in (2, 3, 4):
        same = sum([int(x) for x in str(row["ref_node_ids"]).split()] == frozen[int(row["node_id"])]
                   for _, row in arms[era].iterrows())
        print(f"    era {era}: {same}/{len(arms[era])} nodes share the frozen bibliography "
              f"({len(arms[era]) - same} trimmed further by the longer instruction)")

    # --- 3. the ChatML wrapper ---------------------------------------------
    probe = chatml(tokenizer, *split_prompt(arms[1]["prompt"].iloc[0]))
    checks = {
        "opens with <|im_start|>system": probe.startswith("<|im_start|>system\n"),
        "empty closed <think> block": probe.endswith("<think>\n\n</think>\n\n"),
        "assistant turn opened": "<|im_start|>assistant\n" in probe,
        "no BOS token": tokenizer.bos_token is None,
        "eos is <|im_end|>": tokenizer.eos_token == "<|im_end|>",
    }
    print("\n[3] ChatML wrapper:")
    for label, passed in checks.items():
        print(f"    {'ok  ' if passed else 'FAIL'} {label}")
        if not passed:
            failures.append(f"ChatML: {label}")

    # --- 4. nothing overflows ----------------------------------------------
    print(f"\n[4] token budget {MAX_LENGTH} - {MAX_NEW_TOKENS} = {budget} "
          f"(no prefix in this condition):")
    for era, df in arms.items():
        lengths = [len(tokenizer(chatml(tokenizer, *split_prompt(p))).input_ids)
                   for p in df["prompt"]]
        longest = max(lengths)
        over = sum(n > budget for n in lengths)
        print(f"    era {era}: qwen tokens mean {sum(lengths) / len(lengths):,.0f}, "
              f"max {longest:,} ({budget - longest:,} to spare), "
              f"vicuna mean {df['prompt_tokens'].mean():,.0f}")
        if over:
            failures.append(f"era {era}: {over} prompts over the {budget}-token budget")

    # --- 5. the nodes are in the frozen capped eval set ---------------------
    with open(Path(FROZEN_DIR) / "eval_nodes_capped.json") as f:
        eval_nodes = set(json.load(f))
    missing = [int(n) for n in arms[1]["node_id"] if int(n) not in eval_nodes]
    if missing:
        failures.append(f"{len(missing)} nodes are not in eval_nodes_capped.json: {missing[:5]}")
    print(f"\n[5] {len(arms[1]) - len(missing)}/{len(arms[1])} nodes in "
          f"eval_nodes_capped.json: {'ok' if not missing else 'FAILED'}")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
