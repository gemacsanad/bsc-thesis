"""Score constructed pairs of short scientific statements with ROUGE-L and BERTScore.

Illustration for thesis §6.2, not data from the experiment: each pair was written for this
purpose. Three pairs share almost all their words but state clearly different things (a
reversed finding, swapped roles, a different application); one pair states the same thing in
different words. Scored with the same settings as the experiment (ROUGE-L F1 with stemming;
BERTScore F1, lang="en", i.e. roberta-large, no rescaling), text A as the reference.

Writes example_pairs.csv.

Usage (from the repository root):
    python experiments/2026-11-16_metric-check-random-citations/example_pairs.py
"""

from pathlib import Path

import pandas as pd
from bert_score import score as bs_fn
from rouge_score import rouge_scorer

HERE = Path(__file__).resolve().parent

PAIRS = [
    ("reversed finding",
     "Adding residual connections improves the accuracy of deep graph neural networks on node classification.",
     "Adding residual connections reduces the accuracy of deep graph neural networks on node classification."),
    ("swapped roles",
     "We use a large language model to generate training data for a graph neural network.",
     "We use a graph neural network to generate training data for a large language model."),
    ("different application",
     "We propose a transformer model for detecting tumours in medical images.",
     "We propose a transformer model for detecting fraud in financial transactions."),
    ("same meaning, different words",
     "We propose a transformer model for detecting tumours in medical images.",
     "A transformer network is presented that locates cancerous growths in clinical scans."),
]


def main():
    df = pd.DataFrame(PAIRS, columns=["kind", "text_a", "text_b"])
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    df["rougeL_f"] = [rouge.score(a, b)["rougeL"].fmeasure for a, b in zip(df.text_a, df.text_b)]
    _, _, f1 = bs_fn(df.text_b.tolist(), df.text_a.tolist(), lang="en")
    df["bertscore_f1"] = f1.numpy()
    df.to_csv(HERE / "example_pairs.csv", index=False)
    print(df[["kind", "rougeL_f", "bertscore_f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
