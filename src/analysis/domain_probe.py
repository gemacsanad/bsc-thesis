"""Is the DOMAIN recovered? arXiv-category agreement between each run's text and its gold paper.

A TF-IDF + logistic-regression classifier over ogbn-arxiv's 40 CS categories, trained on the
train-split abstracts, labels every generated abstract; agreement with the gold paper's label
is the run's domain recovery. Its own accuracy on the 945 GOLD abstracts is the instrument's
ceiling (72.4% when written). Also prints what the graph alone gets with no LLM: the majority
label of a paper's references. Pass run directories under experiments/ as arguments.

    python3 src/analysis/domain_probe.py 2026-09-16-14393649_adapter-qwen14b-graphonly-gnn2 ...
"""
import collections, json, os, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

B = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lab = pd.read_csv(os.path.join(B, "data/raw/ogbn_arxiv/nodes.csv"),
                  usecols=["node_id", "abstract", "label"]).set_index("node_id")
splits = json.load(open(os.path.join(B, "data/processed/joint/splits.json")))
ev = json.load(open(os.path.join(B, "data/frozen/eval_nodes_capped.json")))
train = [n for n in splits["train"] if n not in set(ev)]

vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=3, max_features=200000)
clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=20, random_state=0).fit(
    vec.fit_transform(lab.abstract.reindex(train).fillna("")), lab.label.reindex(train))
y = lab.label.reindex(ev).values
predict = lambda texts: clf.predict(vec.transform(texts))                   # noqa: E731
print(f"instrument ceiling (gold abstracts): {np.mean(predict(lab.abstract.reindex(ev).fillna('')) == y):.3f}"
      f"   majority class: {pd.Series(y).value_counts(normalize=True).iloc[0]:.3f}")

e = pd.read_csv(os.path.join(B, "data/processed/joint/edges.csv"))
refs = e[e.src_node_id.isin(set(ev))].groupby("src_node_id").dst_node_id.apply(list)
votes = [collections.Counter(lab.label.reindex([r for r in refs.get(n, []) if r in lab.index])
                             .dropna().astype(int)).most_common(1) for n in ev]
print(f"graph alone, majority label of the references: "
      f"{np.mean([v[0][0] == t if v else False for v, t in zip(votes, y)]):.3f}")

for run in sys.argv[1:]:
    d = pd.read_csv(os.path.join(B, "experiments", run, "evaluated_result.csv")).set_index("node_id").reindex(ev)
    p = predict(d.generated_abstract.fillna("").astype(str))
    acc = pd.to_numeric(d.llm_accuracy, errors="coerce")
    print(f"{run:60s} domain {np.mean(p == y):.3f}   judge | right {acc[p == y].mean():.2f}"
          f"  | wrong {acc[p != y].mean():.2f}")
