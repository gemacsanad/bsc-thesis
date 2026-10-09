"""Run 9's no-LLM floor: answer each eval paper with the nearest TRAINING-split abstract.

The query is the normalised mean of the paper's references' embeddings — exactly the vector the
run-9 refs-mean arm feeds the decoder — and the answer is the real abstract of the closest
train-split paper (pre-2018; the eval papers are 2019). No model, no training. Any soft-prompt
arm that cannot beat this is not using the graph better than a lookup table.

Writes a generations.csv in the format evaluate_results.py scores, so the floor gets a judge
score on the same rubric as every run. Prints TF-IDF identifiability (the probe.py instrument)
for raw and whitened space. CPU, ~2 min, ~1 GB (memmap + 90k x 1024 rows).

    python3 src/analysis/retrieval_baseline.py      # from the repo root, DyTAG env
"""
import csv, json, os, sys
import numpy as np, pandas as pd

B = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(B, "src", "adapter"))
from whiten_features import apply_zca, fit_zca, held_out_nodes   # noqa: E402
from config import RAW_NODE_EMBEDDINGS, WHITEN_EPS                 # noqa: E402
from identify import ranks, summary                               # noqa: E402

OUT = os.path.join(B, "experiments", "2026-10-07_retrieval-baseline", "generations.csv")

splits = json.load(open(os.path.join(B, "data/processed/joint/splits.json")))
ev = json.load(open(os.path.join(B, "data/frozen/eval_nodes_capped.json")))
held, _ = held_out_nodes()
pool = np.array(sorted(set(splits["train"]) - held))
e = pd.read_csv(os.path.join(B, "data/processed/joint/edges.csv"))      # (citing, cited)
refs = e[e.src_node_id.isin(set(ev))].groupby("src_node_id").dst_node_id.apply(list)
nodes = pd.read_csv(os.path.join(B, "data/processed/joint/nodes.csv"),
                    usecols=["node_id", "mag_id", "abstract"]).set_index("node_id")
gold = nodes.abstract.reindex(ev).fillna("").tolist()

X = np.load(RAW_NODE_EMBEDDINGS, mmap_mode="r")
mean, W = fit_zca(X, np.array([i for i in range(X.shape[0]) if i not in held]), WHITEN_EPS)
unit = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)      # noqa: E731
spaces = {"raw": unit, "whitened": lambda a: apply_zca(a, mean, W)}

answer = None
for name, f in spaces.items():
    P = f(np.asarray(X[pool], dtype=np.float32))
    Q = unit(np.stack([f(np.asarray(X[sorted(refs[n])], dtype=np.float32)).mean(0) for n in ev]))
    nearest = pool[np.argmax(Q @ P.T, axis=1)]
    s = summary(ranks(nodes.abstract.reindex(nearest).fillna("").tolist(), gold))
    print(f"{name:9s} nearest train abstract: top-1 {s['top1']:.2%}  MRR {s['mrr']:.3f}  "
          f"distinct {len(set(nearest))}")
    answer = nearest                                   # whitened, the space run 9 uses

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["node_id", "mag_id", "original_abstract", "generated_abstract"])
    w.writeheader()
    for n, a in zip(ev, answer):
        w.writerow({"node_id": n, "mag_id": nodes.mag_id[n], "original_abstract": nodes.abstract[n],
                    "generated_abstract": nodes.abstract[a]})
print(f"wrote {OUT} (whitened answers) — score it with src/evaluation/evaluate_results.py")
