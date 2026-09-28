"""TF-IDF identifiability: can a generated abstract retrieve its own gold out of 945?"""
import csv, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def probe(path):
    rows=list(csv.DictReader(open(path)))
    gen=[r['generated_abstract'] for r in rows]
    gold=[r['original_abstract'] for r in rows]
    v=TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2)
    v.fit(gold+gen)
    G=v.transform(gold); P=v.transform(gen)
    S=(P@G.T).toarray()                      # rows = generated, cols = gold
    n=len(rows)
    order=np.argsort(-S,axis=1)
    rank=np.array([int(np.where(order[i]==i)[0][0])+1 for i in range(n)])
    return dict(n=n, top1=(rank==1).mean(), top5=(rank<=5).mean(),
                top10=(rank<=10).mean(), med=float(np.median(rank)),
                mrr=float(np.mean(1/rank)))

runs=[("baseline-text","2026-09-10-14364743_qwen14b-baseline"),
      ("graphonly-kv192(step2)","2026-09-17-14395431_adapter-qwen14b-graphonly-kv192-gnn2"),
      ("oracle-kv192-lora(step4)","2026-09-22-14422070_oracle-qwen14b-kv192-lora"),
      ("graphonly-lora(step5)","2026-09-24-14437531_graphonly-qwen14b-kv192-lora"),
      ("ntok64-gatv2-lora64(step6)","2026-09-25-14450409_ntok64-gatv2-lora64")]
base="/Users/gemacsanad/egyetem/DyTAG/baseline/experiments/"
print(f"{'run':28s} {'top1':>7s} {'top5':>7s} {'top10':>7s} {'medrank':>8s} {'MRR':>7s}")
print("-"*70)
for name,d in runs:
    r=probe(base+d+"/evaluated_result.csv")
    print(f"{name:28s} {r['top1']*100:6.2f}% {r['top5']*100:6.2f}% {r['top10']*100:6.2f}% {r['med']:8.0f} {r['mrr']:7.4f}")
print(f"\nchance top-1 = {100/945:.3f}%  (n=945)")
