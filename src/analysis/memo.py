"""Are the generated abstracts memorised training abstracts, or a generic template?"""
import csv, json, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
csv.field_size_limit(10**9)
B="/Users/gemacsanad/egyetem/DyTAG/baseline/"
tr=set(json.load(open(B+"data/frozen/training_sets_capped.json"))['train'])
train_abs=[]
with open(B+"data/processed/joint/nodes.csv") as f:
    for r in csv.DictReader(f):
        if int(r['node_id']) in tr:
            train_abs.append(r['abstract'])
print("train abstracts loaded:", len(train_abs))
rows=list(csv.DictReader(open(B+"experiments/2026-09-25-14450409_ntok64-gatv2-lora64/evaluated_result.csv")))
gen=[r['generated_abstract'] for r in rows]; gold=[r['original_abstract'] for r in rows]
v=TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2, max_features=60000)
v.fit(train_abs+gen)
A=v.transform(train_abs); G=v.transform(gen); Gl=v.transform(gold)
import sklearn.preprocessing as pp
A=pp.normalize(A); G=pp.normalize(G); Gl=pp.normalize(Gl)
S=(G@A.T)
mx=np.asarray(S.max(axis=1).todense()).ravel()
print("\ngenerated vs NEAREST TRAINING abstract (TF-IDF cosine):")
print("  mean %.3f  median %.3f  p90 %.3f  max %.3f"%(mx.mean(),np.median(mx),np.percentile(mx,90),mx.max()))
print("  >0.6 (near-copy): %d   >0.8: %d"%((mx>0.6).sum(),(mx>0.8).sum()))
Sg=(Gl@A.T); mg=np.asarray(Sg.max(axis=1).todense()).ravel()
print("\ncontrol - real GOLD abstracts vs nearest training abstract:")
print("  mean %.3f  median %.3f  p90 %.3f"%(mg.mean(),np.median(mg),np.percentile(mg,90)))
# and generated vs its own gold
own=np.asarray((G.multiply(Gl)).sum(axis=1)).ravel()
print("\ngenerated vs its OWN gold: mean %.3f  median %.3f"%(own.mean(),np.median(own)))
