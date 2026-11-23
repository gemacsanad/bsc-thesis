"""Select which references of a paper fit into its neighbourhood, by distance from their centroid.
"""

import os

import numpy as np

from config import PROMPT_MAX_REFS


def centroid_distances(embeddings):
    """-> (N,) cosine distance of each row to the centroid direction of the set."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms == 0, 1.0, norms)

    centroid = normed.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm != 0:
        centroid = centroid / centroid_norm

    sims = normed @ centroid
    return np.abs((1.0 - sims) / 2.0)


def rank_refs(refs, embeddings, max_refs=PROMPT_MAX_REFS):
    """-> the <= max_refs closest references, closest to the centroid first."""
    refs = list(refs)
    if len(refs) <= 1:
        return refs

    distances = centroid_distances(np.asarray(embeddings[refs], dtype=np.float32))
    return [refs[i] for i in np.argsort(distances, kind="stable")[:max_refs]]


def fit_refs(refs, embeddings, measure, budget, max_refs=PROMPT_MAX_REFS):
    """-> the references that go in the prompt, in edges.csv order."""
    refs = list(refs)
    ranked = rank_refs(refs, embeddings, max_refs)

    for k in range(len(ranked), 0, -1):
        keep = set(ranked[:k])
        kept = [r for r in refs if r in keep]
        if measure(kept) <= budget:
            return kept
    return []


def main():
    """Reproduce the frozen centroid_cosine_distance column of the ogbn subset."""
    import pandas as pd

    from config import ROOT

    subset = os.path.join(ROOT, "data", "processed", "ogbn_subset")
    nodes = pd.read_csv(os.path.join(subset, "nodes_subset.csv"))
    edges = pd.read_csv(os.path.join(subset, "edges_subset.csv"))
    features = np.load(os.path.join(subset, "node_subset_features.npy"))
    row = {int(n): i for i, n in enumerate(nodes["node_id"])}

    worst, checked = 0.0, 0
    for test_id in nodes.loc[nodes["is_test"] == 1, "node_id"].astype(int):
        mine = edges["src_node_id"] == test_id
        refs = edges.loc[mine, "dst_node_id"].astype(int).tolist()
        if not refs:
            continue
        got = centroid_distances(features[[row[r] for r in refs]])
        want = edges.loc[mine, "centroid_cosine_distance"].to_numpy()
        worst = max(worst, float(np.abs(got - want).max()))
        checked += len(refs)

    print(f"{checked:,} edges checked against centroid_cosine_distance.py")
    print(f"  largest absolute difference: {worst:.2e}")
    assert worst < 1e-5, "distances disagree with the frozen filter column"

    # Trimming is only the ordering of that same score, cut to a budget.
    refs = list(range(40))
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(nodes), 1024)).astype(np.float32)
    ranked = rank_refs(refs, emb, PROMPT_MAX_REFS)
    d = centroid_distances(emb[refs])
    assert set(ranked) == set(np.argsort(d)[:PROMPT_MAX_REFS]), "ranked the wrong references"

    # 100 tokens per reference: the budget alone decides how many survive.
    for budget, expect in ((10_000, PROMPT_MAX_REFS), (450, 4), (60, 0)):
        kept = fit_refs(refs, emb, lambda k: 100 * len(k), budget, PROMPT_MAX_REFS)
        assert len(kept) == expect, f"budget {budget}: kept {len(kept)}, expected {expect}"
        assert kept == sorted(kept), "kept references left edges.csv order"
        assert set(kept) == set(ranked[:len(kept)]), "dropped the wrong references"
    print(f"  cap: 40 references -> {PROMPT_MAX_REFS} closest to the centroid, "
          f"then trimmed to the token budget, order preserved")


if __name__ == "__main__":
    main()
