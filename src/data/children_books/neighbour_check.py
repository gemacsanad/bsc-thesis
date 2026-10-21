"""
Step 4: does a learned aggregator have room to beat the plain neighbour mean? CPU only.

For every evaluation node, how close can its neighbours' voyage embeddings get to its own?
The rows of docs/thesis_context/06_results.md §6.2, plus a learned linear map:

  mean           normalised mean of the neighbours (run 11's refs-mean input)
  best single    the one neighbour closest to the answer (chosen with the answer)
  closest-3 mean mean of the 3 neighbours closest to the answer (chosen with the answer)
  NNLS           the best non-negative weighting of the neighbours, fitted to the answer
  linear map     ridge regression mean -> own embedding, fitted on training nodes, scored on
                 the held-out nodes: the "mean + linear head" baseline, and the room for
                 *translation* (neighbours that point at the node without resembling it)

The rows chosen or fitted with the answer measure room for *selection* (a few neighbours are
informative and attention could find them). Every such fit improves on any target, so each
is repeated against an unrelated node's embedding (another evaluation node, not a neighbour):
the control. Room = how much more the fit gains for the real target than for the control.

Top-1 is retrieval among the evaluation nodes by cosine, as in run 11's GNN table.

Usage:
    python neighbour_check.py --dataset children
    python neighbour_check.py --dataset papers
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from build_children import DATA_DIR, ROOT

PATHS = {
    "children": dict(
        edges=os.path.join(DATA_DIR, "edges.csv"),
        embeddings=os.path.join(DATA_DIR, "node_embeddings.npy"),
        sets=os.path.join(ROOT, "data", "frozen", "children_books", "training_sets.json"),
        eval=os.path.join(ROOT, "data", "frozen", "children_books", "eval_nodes.json")),
    "papers": dict(
        edges=os.path.join(ROOT, "data", "processed", "joint", "edges.csv"),
        embeddings=os.path.join(ROOT, "data", "processed", "joint", "node_embeddings.npy"),
        sets=os.path.join(ROOT, "data", "frozen", "training_sets_run9.json"),
        eval=os.path.join(ROOT, "data", "frozen", "eval_nodes_capped.json")),
}
SEED = 42
RIDGE = [1e-4, 1e-3, 1e-2, 1e-1, 1]   # x the mean diagonal of the Gram matrix; chosen on validation


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def top1(estimates, targets):
    return float(((unit(estimates) @ targets.T).argmax(1) == np.arange(len(targets))).mean())


def neighbour_means(x, nbrs, nodes):
    return unit(np.stack([x[nbrs[n]].mean(0) for n in nodes]))


def fits(x, nbrs, node, target):
    """-> (mean, best single, closest-3 mean, NNLS) cosines of node's neighbours to target."""
    n = np.asarray(x[nbrs[node]], dtype=np.float64)
    cos = n @ target
    best3 = unit(n[np.argsort(-cos)[:3]].mean(0))
    w, _ = nnls(n.T, target)
    fitted = n.T @ w
    nnls_cos = float(fitted @ target / np.linalg.norm(fitted)) if w.any() else 0.0
    return float(unit(n.mean(0)) @ target), float(cos.max()), float(best3 @ target), nnls_cos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(PATHS), required=True)
    args = ap.parse_args()
    p = PATHS[args.dataset]

    x = np.load(p["embeddings"], mmap_mode="r")
    edges = pd.read_csv(p["edges"])
    edges = edges[edges.src_node_id != edges.dst_node_id]
    nbrs = edges.groupby("src_node_id")["dst_node_id"].apply(np.asarray)   # papers: references
    with open(p["sets"]) as f:
        sets = json.load(f)
    with open(p["eval"]) as f:
        eval_nodes = json.load(f)
    has = set(nbrs.index)
    train = [n for n in sets["train"] if n in has]
    valid = [n for n in sets["valid"] if n in has]
    eval_nodes = [n for n in eval_nodes if n in has]
    print(f"{args.dataset}: {len(train):,} train / {len(valid):,} valid / {len(eval_nodes):,} eval "
          f"nodes with a neighbour; eval degree median "
          f"{np.median([len(nbrs[n]) for n in eval_nodes]):.0f}")

    # The unrelated control: each eval node is fitted to another eval node's embedding,
    # never one of its own neighbours.
    rng = np.random.default_rng(SEED)
    other = {}
    for n in eval_nodes:
        while True:
            m = eval_nodes[rng.integers(len(eval_nodes))]
            if m != n and m not in set(nbrs[n]):
                other[n] = m
                break

    targets = np.asarray(x[eval_nodes], dtype=np.float64)
    real = np.array([fits(x, nbrs, n, t) for n, t in zip(eval_nodes, targets)])
    ctrl = np.array([fits(x, nbrs, n, np.asarray(x[other[n]], dtype=np.float64))
                     for n in eval_nodes])

    # mean + linear head: ridge with a bias, the regulariser chosen on validation.
    m_tr = neighbour_means(x, nbrs, train)
    y_tr = np.asarray(x[train], dtype=np.float32)
    mu_m, mu_y = m_tr.mean(0), y_tr.mean(0)
    a, b = m_tr - mu_m, y_tr - mu_y
    gram, cross = a.T @ a, a.T @ b
    m_va, y_va = neighbour_means(x, nbrs, valid), np.asarray(x[valid], dtype=np.float32)
    best = None
    for lam in RIDGE:
        w = np.linalg.solve(gram + lam * np.trace(gram) / len(gram) * np.eye(len(gram)), cross)
        c = float((unit((m_va - mu_m) @ w + mu_y) * y_va).sum(1).mean())
        print(f"  ridge lambda {lam:>5}: validation cosine {c:.4f}")
        if best is None or c > best[0]:
            best = (c, lam, w)
    _, lam, w = best
    m_ev = neighbour_means(x, nbrs, eval_nodes)
    mapped = unit((m_ev - mu_m) @ w + mu_y)
    lin = (mapped * targets).sum(1)

    print(f"\n{'estimate (945 eval)':38s} {'real':>7s} {'control':>8s} {'top-1':>7s}")
    print(f"{'mean':38s} {real[:, 0].mean():7.3f} {ctrl[:, 0].mean():8.3f} "
          f"{top1(m_ev, targets.astype(np.float32)):7.1%}")
    for i, name in [(1, "best single (with the answer)"), (2, "closest-3 mean (with the answer)"),
                    (3, "NNLS (fitted to the answer)")]:
        print(f"{name:38s} {real[:, i].mean():7.3f} {ctrl[:, i].mean():8.3f}")
    print(f"{f'linear map (ridge {lam}, fitted on train)':38s} {lin.mean():7.3f} {'':8s} "
          f"{top1(mapped, targets.astype(np.float32)):7.1%}")
    gain, gain_c = real[:, 3].mean() - real[:, 0].mean(), ctrl[:, 3].mean() - ctrl[:, 0].mean()
    room = (real[:, 3] - real[:, 0]) - (ctrl[:, 3] - ctrl[:, 0])
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    print(f"\nselection room: NNLS gain {gain:+.3f} real vs {gain_c:+.3f} control "
          f"-> {room.mean():+.3f} (SE {se(room):.3f})")
    print(f"translation room: linear map - mean {(lin - real[:, 0]).mean():+.3f} "
          f"(SE {se(lin - real[:, 0]):.3f})")
    deg = np.array([len(nbrs[n]) for n in eval_nodes])
    q = np.quantile(real[:, 0], [.2, .8])
    print(f"mean cosine: p20 {q[0]:.3f}, p80 {q[1]:.3f}; "
          f"corr(mean cosine, log degree) {np.corrcoef(real[:, 0], np.log(deg))[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
