"""Reproduce the Results section of README.md: era separation, and the recall check.

Two questions, both answered without any API call.

**Did the prompts separate?** Paired Wilcoxon over the 48 nodes, era against era, plus the
comparison against the Vicuna scores the same nodes carry in `test_set.csv`.

**Are the high scores recalled abstracts rather than reconstructions?** `gpt-5.6-luna` may
have these arXiv papers in its training data, in which case a high score would measure
retrieval, not the prompt. The score distribution cannot answer this — a mixture of
recognised and inferred papers is bimodal, which is what the distribution looks like either
way. Three lexical measures answer it instead:

  * **gold-only names** — an acronym, CamelCase name or alphanumeric identifier that appears
    in the target's own abstract and in *none* of the references the prompt showed. It cannot
    have come from the prompt, so it came from the model. This is the direct signature of
    recall, and it is why generic terms need no stoplist: CNN and LSTM are in the references,
    so subtracting the references removes them.
  * **bigram overlap with the gold**, against `docs/adapter.md` Stage 6c's 0.027 for Vicuna.
  * **bigram overlap with the nearest single cited reference**, and the >0.5 copy rate that
    Stage 6c puts at 10.4% for Vicuna.

The test is indirect: a model that recalled the target's content while renaming everything
would pass it. Combined with a gold overlap that does not move, that is unlikely — but the
direct test (ask the model to name the paper citing this reference list) is a different
script that has not been written.

Usage:
    conda activate DyTAG
    python experiments/prompt_sensitivity_gpt/diagnostics.py
    python experiments/prompt_sensitivity_gpt/diagnostics.py --csv per_node.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "adapter"))

from text import Corpus                                                     # noqa: E402

ARMS = {1: "01_era1", 2: "02_era2", 3: "03_era3", 4: "04_era4"}

# Acronyms (CNN, M3SDA), CamelCase names (TextBoxes, ResNet), and alphanumeric identifiers
# (VGG-16). A heuristic, but applied identically to the generation, the gold and the
# references, so the *difference* between them is what carries the argument.
NAME = re.compile(r"\b(?:[A-Z]{2,}[0-9]*"
                  r"|[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+"
                  r"|[A-Za-z]{2,}-?[0-9]{1,4})\b")


def bigrams(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return set(zip(words, words[1:]))


def overlap(generated, other):
    """Share of the generation's bigrams that `other` also has. Asymmetric on purpose: the
    question is how much of the output came from somewhere, not how similar two texts are."""
    return len(generated & other) / max(len(generated), 1)


def per_node(corpus):
    rows = []
    for era, arm in ARMS.items():
        test_set = pd.read_csv(HERE / arm / "test_set.csv").set_index("node_id")
        scored = pd.read_csv(HERE / arm / "evaluated_result.csv").set_index("node_id")
        for node_id in test_set.index:
            generated = scored.loc[node_id, "generated_abstract"]
            gold = corpus.abstract[node_id]
            refs = [int(r) for r in str(test_set.loc[node_id, "ref_node_ids"]).split()]
            ref_texts = [f"{corpus.title[r]} {corpus.abstract[r]}" for r in refs]

            gen_bg = bigrams(generated)
            leaked = ((set(NAME.findall(str(generated))) & set(NAME.findall(str(gold))))
                      - set(NAME.findall(" ".join(ref_texts))))
            rows.append({
                "era": era,
                "node_id": node_id,
                "accuracy": pd.to_numeric(scored.loc[node_id, "llm_accuracy"], errors="coerce"),
                "overlap_gold": overlap(gen_bg, bigrams(gold)),
                "overlap_nearest_ref": max((overlap(gen_bg, bigrams(t)) for t in ref_texts),
                                           default=0.0),
                "n_gold_only_names": len(leaked),
                "gold_only_names": " ".join(sorted(leaked)),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="also write the per-node table here")
    args = ap.parse_args()

    print("Loading corpus …")
    d = per_node(Corpus())
    selection = pd.read_csv(HERE / "test_set.csv").set_index("node_id")
    wide = d.pivot(index="node_id", columns="era", values="accuracy").join(selection)

    print("\n=== era separation ===")
    print("era   mean accuracy")
    for era, g in d.groupby("era"):
        print(f"  {era}   {g.accuracy.mean():.3f}")

    print("\npaired Wilcoxon, n=48")
    for a, b in [(1, 2), (2, 3), (3, 4), (1, 4)]:
        diff = wide[b] - wide[a]
        print(f"  era{a} -> era{b}: {diff.mean():+.2f}  "
              f"{(diff > 0).sum()}W/{(diff == 0).sum()}T/{(diff < 0).sum()}L  "
              f"p={stats.wilcoxon(wide[a], wide[b]).pvalue:.4f}")
    print(f"  GPT era4 vs Vicuna era3: {(wide[4] - wide.score_v3).mean():+.2f}  "
          f"p={stats.wilcoxon(wide[4], wide.score_v3).pvalue:.2e}")

    print("\n=== recall check ===")
    print("era  gold-only names   overlap w/ gold   overlap w/ nearest ref   copies >0.5")
    for era, g in d.groupby("era"):
        print(f"  {era}      {(g.n_gold_only_names > 0).sum():>2}/48            "
              f"{g.overlap_gold.mean():.3f}             {g.overlap_nearest_ref.mean():.3f}"
              f"              {(g.overlap_nearest_ref > 0.5).sum():>2}/48")
    print(f"\n  largest single overlap with a gold anywhere: {d.overlap_gold.max():.3f}")
    leaks = d[d.n_gold_only_names > 0]
    print(f"  every gold-only name in the run ({len(leaks)} node{'s' * (len(leaks) != 1)}): "
          f"{'; '.join(f'era{r.era} node {r.node_id}: {r.gold_only_names}' for _, r in leaks.iterrows()) or 'none'}")

    print("\n=== is the shape the sample? ===")
    cells = wide.groupby("group").agg(n=("score_v3", "size"), vicuna_v3=("score_v3", "mean"),
                                      gpt_era4=(4, "mean"))
    print(cells.round(2).to_string())
    for column in ("score_v3", "out_degree"):
        r = stats.spearmanr(wide[4], wide[column])
        print(f"  spearman(GPT era4, {column}) = {r.statistic:+.2f}  p={r.pvalue:.3f}")

    if args.csv:
        d.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
