"""Build the 48-node test set and the four era-specific prompt sets.

Writes:

    test_set.csv          the node selection, with the Vicuna scores it is traceable to
    0N_eraN/test_set.csv  the same 48 nodes, each row carrying that era's finished prompt

Node selection
--------------
The 48 nodes are copied from `../sentiment_analysis/03_prompt_fix/comparison_dataset.csv`,
the sample rounds 2 and 3 both judged. They are not a random draw: they were stratified on
how each node's era-2 and era-3 accuracy moved (cells A-D), so the group column is carried
through and the mean over these 48 is not an estimate of the mean over the 945. What they
buy is a per-node trace — every one of them already has three Vicuna generations and three
judge scores, so a GPT output can be laid directly against the 7B's on the same paper under
the same prompt.

Prompts
-------
Built through the same centroid cap the Vicuna runs were evaluated under:
`neighbour_cap.fit_refs` keeps the PROMPT_MAX_REFS references closest to the node's own
reference centroid, then drops the farthest one at a time until the prompt fits

    MAX_LENGTH - NUM_PREFIX_TOKENS - max(GENERATION_RESERVE, len(gold) + 1)

measured with the **Vicuna tokenizer**. None of those three terms describe gpt-5.6-luna,
which has neither a 4,096-token context nor a 10-token graph prefix. They are kept because
the point of the run is to change the model and nothing else: GPT has to see the same
references, in the same order, in the same words that the 7B saw.

One consequence is worth stating rather than hiding: the budget is shared between the
instruction and the references, so a longer instruction buys fewer references. The eras do
**not** all show the same number of sources, and `refs_shown` records what each one got.
That is not a flaw introduced here — it is how the 7B runs behaved too.

Usage:
    conda activate DyTAG
    python experiments/prompt_sensitivity_gpt/build_test_set.py            # all four eras
    python experiments/prompt_sensitivity_gpt/build_test_set.py --era 4    # just one
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "adapter"))          # flat imports, as in src/adapter

from config import (BASE_MODEL, GENERATION_RESERVE, JOINT_DIR, MAX_LENGTH,  # noqa: E402
                    NUM_PREFIX_TOKENS)
from neighbour_cap import fit_refs                                          # noqa: E402
from text import Corpus                                                     # noqa: E402

import prompts                                                              # noqa: E402

SOURCE = ROOT / "experiments" / "sentiment_analysis" / "03_prompt_fix" / "comparison_dataset.csv"
ARMS = {1: "01_era1", 2: "02_era2", 3: "03_era3", 4: "04_era4"}


def capped_prompt(era, tokenizer, corpus, embeddings, node_id):
    """-> (prompt, kept_refs) for one node under one era.

    A transcription of `text.capped_example` with the era's `build_prompt` substituted for
    the one that happens to be in the working tree. The budget arithmetic is copied exactly;
    if it drifts, GPT reads a different bibliography than Vicuna did and the comparison
    quietly stops being about the model.
    """
    target_ids = tokenizer(corpus.abstract[node_id], add_special_tokens=False).input_ids
    reserve = max(GENERATION_RESERVE, len(target_ids) + 1)
    budget = MAX_LENGTH - NUM_PREFIX_TOKENS - reserve

    all_refs = corpus.refs.get(node_id, [])
    total = len(all_refs)

    def texts(refs):
        return [(corpus.title[n], corpus.abstract[n]) for n in refs]

    def measure(kept):
        return len(tokenizer(prompts.build_prompt(era, texts(kept), total)).input_ids)

    refs = fit_refs(all_refs, embeddings, measure, budget)
    return prompts.build_prompt(era, texts(refs), total), refs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--era", type=int, action="append", choices=[1, 2, 3, 4],
                    help="build only this era; repeatable. Default: all four.")
    args = ap.parse_args()
    eras = sorted(set(args.era)) if args.era else [1, 2, 3, 4]

    from transformers import AutoTokenizer

    sample = pd.read_csv(SOURCE, usecols=["node_id", "group", "score_v1", "score_v2",
                                          "score_v3", "original_abstract"])
    print(f"{len(sample)} nodes from {SOURCE.relative_to(ROOT)}")
    print(sample["group"].value_counts().to_string())

    print(f"\nLoading tokenizer {BASE_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    print("Loading corpus …")
    corpus = Corpus()
    mag_id = pd.read_csv(Path(JOINT_DIR) / "nodes.csv",
                         usecols=["node_id", "mag_id"]).set_index("node_id")["mag_id"]
    embeddings = np.load(Path(JOINT_DIR) / "node_embeddings.npy", mmap_mode="r")

    # The gold in the source CSV came from a generations.csv; re-read it from the corpus so
    # every arm scores against the same text the prompt budget was computed from.
    selection = pd.DataFrame({
        "node_id": sample["node_id"],
        "mag_id": sample["node_id"].map(mag_id),
        "group": sample["group"],
        "out_degree": [corpus.out_degree(n) for n in sample["node_id"]],
        "score_v1": sample["score_v1"],
        "score_v2": sample["score_v2"],
        "score_v3": sample["score_v3"],
        "original_abstract": [corpus.abstract[n] for n in sample["node_id"]],
    })
    assert selection.notna().all().all(), "a sampled node is missing from the joint corpus"
    selection.to_csv(HERE / "test_set.csv", index=False)
    print(f"\nWrote {HERE.name}/test_set.csv — {len(selection)} nodes, "
          f"out-degree {selection.out_degree.min()}-{selection.out_degree.max()}")

    for era in eras:
        if prompts.ERAS[era] is None:
            print(f"\nera {era}: no prompt text in prompts.py — skipped")
            continue

        rows = []
        for node_id in selection["node_id"]:
            prompt, refs = capped_prompt(era, tokenizer, corpus, embeddings, node_id)
            rows.append({
                "refs_shown": len(refs),
                "ref_node_ids": " ".join(str(r) for r in refs),
                "prompt_tokens": len(tokenizer(prompt).input_ids),
                "prompt": prompt,
            })

        arm = selection[["node_id", "mag_id", "out_degree", "original_abstract"]].copy()
        arm = pd.concat([arm.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        path = HERE / ARMS[era] / "test_set.csv"
        arm.to_csv(path, index=False)

        budget = MAX_LENGTH - NUM_PREFIX_TOKENS - GENERATION_RESERVE
        print(f"\nera {era} -> {ARMS[era]}/test_set.csv")
        print(f"  instruction {len(tokenizer(prompts.ERAS[era]).input_ids)} tokens")
        print(f"  refs shown    mean {arm.refs_shown.mean():.2f}  "
              f"min {arm.refs_shown.min()}  max {arm.refs_shown.max()}  "
              f"(of out-degree mean {arm.out_degree.mean():.1f})")
        print(f"  prompt tokens mean {arm.prompt_tokens.mean():,.0f}  "
              f"max {arm.prompt_tokens.max():,}  (budget <= {budget:,})")
        assert arm.prompt_tokens.max() <= budget, \
            "a prompt is over the budget the capped sets were frozen under"


if __name__ == "__main__":
    main()
