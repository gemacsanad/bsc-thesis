"""
Run 8 — ZCA-whiten the voyage node features, once, before anything reads them.

Why
---
The features are anisotropic before the GNN touches them. Over the 25,539 nodes reachable
from the eval set the participation ratio is 92.6 of 1024 dimensions, and the mean of 25k
unit vectors has norm 0.5975 where isotropy would give ~0.006. Mean-pooling a reference set
concentrates that further: the pooled vectors the adapter consumes sit at pairwise cosine
0.6602, so two thirds of their energy is one direction shared by every node.

`PrefixAdapter.trunk` ends in a LayerNorm, which normalises per sample across features and
therefore cannot remove a component shared *across* nodes. A near-constant input produced a
near-constant prefix, and greedy decoding emitted one string for 945 papers. Full evidence in
docs/step8_representation_plan.md.

ZCA is the classical correction (Mu & Viswanath, ICLR 2018; Su et al., 2021):

    W = V diag(w + floor)^-0.5 V^T,   floor = WHITEN_EPS * w.max()

applied to the centred features, which are then re-normalised to unit length because that is
what every downstream consumer — the GNN, oracle_vector, the retrieval metrics — assumes.

Where it is placed, and why not in the adapter
-----------------------------------------------
This is preprocessing, not a model component. The anisotropy belongs to the embedding model's
output, so it is corrected where the features are made; every arm — GNN, neighbour tokens,
oracle — then receives the same treatment and the comparisons between them stay honest. A
fixed 1024x1024 matrix bolted onto the adapter would correct one arm and not the others.

Leakage
-------
The transform is fitted on the nodes NOT in the evaluation or validation sets. That excludes
the seeds only; their references are shared with training nodes and stay in the fit, which is
the same situation the GNN is already in and is not avoidable without partitioning the graph.

Usage:
    python whiten_features.py                    # fit, verify, write
    python whiten_features.py --dry-run          # measure the gate, write nothing
"""

import argparse
import json
import os

import numpy as np

from config import (FROZEN_DIR, JOINT_DIR, NODE_EMBEDDINGS, RAW_NODE_EMBEDDINGS, SEED,
                    WHITEN_COSINE_GATE, WHITEN_EPS)

CHUNK = 32768      # rows held in RAM at once; 32k x 1024 float32 = 134 MB


def held_out_nodes():
    """-> the seed node ids the transform must not be fitted on."""
    with open(os.path.join(FROZEN_DIR, "eval_nodes_capped.json")) as f:
        ev = json.load(f)
    ev = ev["nodes"] if isinstance(ev, dict) else ev
    with open(os.path.join(FROZEN_DIR, "training_sets_capped.json")) as f:
        sets = json.load(f)
    return set(ev) | set(sets["valid"]), ev


def fit_zca(x, rows, eps):
    """-> (mean, W) over `rows` of the memmap `x`, accumulated in two streaming passes.

    Streamed rather than loaded: the full matrix is 384,756 x 1024 float32 = 1.5 GB and this
    has to run on a laptop. The accumulators are 1024 and 1024x1024, so memory is set by CHUNK
    and not by the number of nodes.
    """
    dim = x.shape[1]
    total = np.zeros(dim, dtype=np.float64)
    for i in range(0, len(rows), CHUNK):
        total += np.asarray(x[rows[i:i + CHUNK]], dtype=np.float64).sum(axis=0)
    mean = total / len(rows)

    cov = np.zeros((dim, dim), dtype=np.float64)
    for i in range(0, len(rows), CHUNK):
        block = np.asarray(x[rows[i:i + CHUNK]], dtype=np.float64) - mean
        cov += block.T @ block
    cov /= len(rows) - 1

    w, v = np.linalg.eigh(cov)
    floor = eps * w.max()
    scale = 1.0 / np.sqrt(np.maximum(w, floor))
    print(f"eigenvalues: max {w.max():.3e}  min {w.min():.3e}  floor {floor:.3e}  "
          f"{int((w < floor).sum())} of {dim} shrunk")
    return mean.astype(np.float32), (v * scale) @ v.T


def apply_zca(block, mean, transform):
    """Centre, whiten, re-normalise to unit length — the order every consumer assumes."""
    out = (np.asarray(block, dtype=np.float32) - mean) @ transform.astype(np.float32)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def spectrum(vectors):
    """Participation ratio and the mean-direction norm, the two collapse numbers."""
    unit = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    centred = unit - unit.mean(axis=0)
    w = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    w = np.clip(w, 0, None)
    return {"participation_ratio": float(w.sum() ** 2 / (w ** 2).sum()),
            "mean_direction_norm": float(np.linalg.norm(unit.mean(axis=0)))}


def retrieval(x, eval_nodes, refs, mean=None, transform=None):
    """Top-1 of the right paper out of 945 from its mean-pooled references. The gate.

    The same quantity src/analysis/refsignal.py reports as 64.34%, so the two are directly
    comparable — which is the only reason this measurement is worth running here.
    """
    need = sorted(set(eval_nodes) | {r for n in eval_nodes for r in refs.get(n, ())})
    row = {n: i for i, n in enumerate(need)}
    e = np.asarray(x[need], dtype=np.float32)
    e = (apply_zca(e, mean, transform) if transform is not None
         else e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))

    targets = e[[row[n] for n in eval_nodes]]
    pooled = np.stack([e[[row[r] for r in refs[n]]].mean(axis=0) if refs.get(n)
                       else np.zeros(e.shape[1], np.float32) for n in eval_nodes])
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9

    scores = pooled @ targets.T
    hits = scores.argmax(axis=1) == np.arange(len(eval_nodes))
    cosines = pooled @ pooled.T
    n = len(pooled)
    return (float(hits.mean()),
            float((cosines.sum() - np.trace(cosines)) / (n * (n - 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps", type=float, default=WHITEN_EPS)
    ap.add_argument("--gate-cosine", type=float, default=WHITEN_COSINE_GATE,
                    help="pooled-reference pairwise cosine the whitened features must get "
                         "BELOW. This is the gate that matters: step 7 had 64.34%% of the "
                         "information retrievable and delivered 0.11%%, because the prefix did "
                         "not vary between nodes. The oracle is fed vectors at 0.378")
    ap.add_argument("--raw", default=RAW_NODE_EMBEDDINGS)
    ap.add_argument("--out", default=NODE_EMBEDDINGS)
    ap.add_argument("--dry-run", action="store_true",
                    help="measure the gate and write nothing")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    x = np.load(args.raw, mmap_mode="r")
    print(f"{args.raw}: {x.shape}")

    import pandas as pd
    edges = pd.read_csv(os.path.join(JOINT_DIR, "edges.csv"))
    refs = edges.groupby("src_node_id")["dst_node_id"].apply(list).to_dict()

    excluded, eval_nodes = held_out_nodes()
    rows = np.array([i for i in range(x.shape[0]) if i not in excluded])
    print(f"fitting on {len(rows):,} nodes ({len(excluded):,} eval/valid seeds held out)")

    mean, transform = fit_zca(x, rows, args.eps)

    before, cos_before = retrieval(x, eval_nodes, refs)
    after, cos_after = retrieval(x, eval_nodes, refs, mean, transform)
    sample = np.asarray(x[rng.choice(x.shape[0], 20000, replace=False)], dtype=np.float32)
    print(f"\npooled-reference retrieval over {len(eval_nodes)} eval nodes "
          f"(chance {1 / len(eval_nodes):.3%}):")
    print(f"  raw       top-1 {before:7.2%}   pooled pairwise cosine {cos_before:.4f}")
    print(f"  whitened  top-1 {after:7.2%}   pooled pairwise cosine {cos_after:.4f}")
    print(f"  raw       {spectrum(sample)}")
    print(f"  whitened  {spectrum(apply_zca(sample, mean, transform))}")

    # The cheap half of the run-8 hypothesis. Failing here costs minutes; discovering it after
    # Stage B costs a day of A100 time and another evaluation job.
    if cos_after > args.gate_cosine:
        print(f"\nFAILED: pooled pairwise cosine {cos_after:.4f} is above the "
              f"{args.gate_cosine} gate — the features are still near-constant between nodes, "
              f"which is exactly what step 7 shipped. Do not spend GPU time on this.")
        raise SystemExit(1)
    if after < before:
        print(f"\nFAILED: retrieval fell {before:.2%} -> {after:.2%}. Whitening is meant to "
              f"spread the representation without losing information, and this lost some; "
              f"raise --eps and re-measure before spending GPU time.")
        raise SystemExit(1)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    # Written through a memmap for the same reason the fit is streamed: the output is another
    # 1.5 GB and materialising it alongside the input would not fit.
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32, shape=x.shape)
    for i in range(0, x.shape[0], CHUNK):
        out[i:i + CHUNK] = apply_zca(x[i:i + CHUNK], mean, transform)
    out.flush()
    np.savez(os.path.splitext(args.out)[0] + "_transform.npz", mean=mean,
             transform=transform.astype(np.float32), eps=args.eps,
             eval_top1=after, eval_cosine=cos_after, fitted_nodes=len(rows))
    print(f"\nwrote {args.out} and its transform; eval top-1 {after:.2%}")


if __name__ == "__main__":
    main()
