"""Identification: rank each generated abstract against the original abstracts of all papers in
the set by TF-IDF cosine. Rank 1 means the text is closest to its own paper's abstract.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def ranks(generated, golds):
    """-> (n,) rank of each generation's own gold among all `golds`; 1 is best."""
    assert len(generated) == len(golds), "one generation per gold"
    vectoriser = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(golds)
    similarity = (vectoriser.transform(generated) @ vectoriser.transform(golds).T).toarray()
    own = similarity.diagonal()
    return (similarity >= own[:, None]).sum(axis=1)


def summary(rank):
    """-> {"top1": share at rank 1, "mrr": mean reciprocal rank}."""
    rank = np.asarray(rank)
    return {"top1": float(np.mean(rank == 1)), "mrr": float(np.mean(1.0 / rank))}
