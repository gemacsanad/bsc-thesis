"""
Run 9 — identifiability: does a generated abstract pick out its own paper?

Every generation is ranked against the gold abstracts of the same set of papers by TF-IDF
cosine; rank 1 means the text is closer to its own gold than to any other. This is the probe
every soft-prompt analysis since step 2 has used (src/analysis/probe.py), moved into training so
that model selection and the stop rules read generated text rather than LM loss — the number
that has misled this project three times.

The vectoriser is fitted on the golds only, exactly as the analysis probe does, so the scores
in training and in the evaluation write-ups come from one instrument.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def ranks(generated, golds):
    """-> (n,) rank of each generation's own gold among all `golds`; 1 is best.

    `generated[i]` is scored against `golds[i]`. Ties count AGAINST the generation: an empty
    text has similarity 0 to every gold, and with ties in its favour it would rank first on
    every row — a collapsed decoder would look perfect. Counting `>=` puts it last instead.
    """
    assert len(generated) == len(golds), "one generation per gold"
    vectoriser = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(golds)
    similarity = (vectoriser.transform(generated) @ vectoriser.transform(golds).T).toarray()
    own = similarity.diagonal()
    return (similarity >= own[:, None]).sum(axis=1)


def summary(rank):
    """-> {"top1": share at rank 1, "mrr": mean reciprocal rank}."""
    rank = np.asarray(rank)
    return {"top1": float(np.mean(rank == 1)), "mrr": float(np.mean(1.0 / rank))}
