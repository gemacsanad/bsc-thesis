"""Build the two prompt sets of the metric check: real citations vs random citations.

200 of the 945 evaluation papers (fixed seed). For each paper:
  real   — the titles and abstracts of its recorded references (out-edges in the joint graph);
  random — the same number of papers drawn at random from the joint graph, excluding the paper
           itself and its real references;
  same_category — the same number of ogbn-arxiv papers drawn at random from the paper's own
           arXiv category (ogbn-arxiv label), excluding the paper and its real references. Drawn
           with a separate generator (SEED_CATEGORY), so the `random` condition is unchanged.
The prompt is identical in both conditions; only the listed publications differ. Every
reference is shown, with no cap: the generator's context is large enough.

Writes test_set.csv (node_id, mag_id, condition, n_refs, approx_words, prompt, original_abstract).

Usage (from the repository root):
    python experiments/2026-11-16_metric-check-random-citations/build_inputs.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
N_PAPERS = 200
SEED = 0
SEED_CATEGORY = 1

INSTRUCTION = (
    "The publications listed below are cited by one paper, whose own abstract is missing. "
    "Write the abstract of that paper. Write 150-200 words of plain prose. "
    "Output only the abstract text, without any heading or explanation."
)


def reference_block(ids, nodes):
    parts = []
    for k, nid in enumerate(ids, 1):
        title = str(nodes.at[nid, "title"])
        abstract = nodes.at[nid, "abstract"]
        entry = f"[{k}] Title: {title}"
        if isinstance(abstract, str) and abstract.strip():
            entry += f"\nAbstract: {abstract.strip()}"
        parts.append(entry)
    return "\n\n".join(parts)


def main():
    nodes = pd.read_csv(ROOT / "data/processed/joint/nodes.csv",
                        usecols=["node_id", "mag_id", "title", "abstract"]).set_index("node_id")
    edges = pd.read_csv(ROOT / "data/processed/joint/edges.csv")  # src cites dst
    edges = edges[edges["src_node_id"] != edges["dst_node_id"]]
    references = edges.groupby("src_node_id")["dst_node_id"].agg(lambda s: sorted(set(s))).to_dict()

    eval_ids = sorted(json.load(open(ROOT / "data/frozen/eval_nodes_capped.json")))
    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(eval_ids, size=N_PAPERS, replace=False).tolist())

    pool = nodes.index[nodes["title"].notna()].to_numpy()
    raw = pd.read_csv(ROOT / "data/raw/ogbn_arxiv/nodes.csv", usecols=["node_id", "mag_id", "label"])
    raw = raw.set_index("node_id")
    shared = raw.index.intersection(nodes.index)
    assert (raw.loc[shared, "mag_id"] == nodes.loc[shared, "mag_id"]).all(), "node ids differ"
    rng_category = np.random.default_rng(SEED_CATEGORY)
    rows = []
    for nid in sample:
        real = references[nid]
        excluded = set(real) | {nid}
        candidates = pool[~np.isin(pool, list(excluded))]
        random = sorted(rng.choice(candidates, size=len(real), replace=False).tolist())
        category = raw.index[raw["label"] == raw.at[nid, "label"]].to_numpy()
        category = category[~np.isin(category, list(excluded))]
        assert len(category) >= len(real), f"category of {nid} too small"
        same = sorted(rng_category.choice(category, size=len(real), replace=False).tolist())
        for condition, ids in (("real", real), ("random", random), ("same_category", same)):
            prompt = f"{INSTRUCTION}\n\nCited publications:\n\n{reference_block(ids, nodes)}"
            rows.append({"node_id": nid, "mag_id": nodes.at[nid, "mag_id"],
                         "condition": condition, "n_refs": len(ids),
                         "approx_words": len(prompt.split()), "prompt": prompt,
                         "original_abstract": nodes.at[nid, "abstract"]})

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "test_set.csv", index=False)
    words = df.groupby("condition")["approx_words"]
    print(f"{len(sample)} papers, {len(df)} prompts")
    print(f"references per paper: median {df['n_refs'].median():.0f}, max {df['n_refs'].max()}")
    print(f"prompt words: total {df['approx_words'].sum():,}; per condition median "
          f"{words.median().to_dict()}, max {words.max().to_dict()}")


if __name__ == "__main__":
    main()
