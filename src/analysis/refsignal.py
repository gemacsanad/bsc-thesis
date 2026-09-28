"""How much does the reference set determine the target, in the representation we feed?

Pure retrieval in Voyage space. No model, no GPU. mmap so only the needed rows are read.
"""
import csv, json, numpy as np, collections
B="/Users/gemacsanad/egyetem/DyTAG/baseline/"
ev=[int(r['node_id']) for r in csv.DictReader(open(B+"experiments/2026-09-25-14450409_ntok64-gatv2-lora64/evaluated_result.csv"))]
evs=set(ev)

# edge_index is stored (cited, citing); a node's references are its in-edges.
refs=collections.defaultdict(list)
import pandas as pd
e=pd.read_csv(B+"data/processed/joint/edges.csv")
print("edge cols:",list(e.columns), len(e))
c0,c1=e.columns[0],e.columns[1]
m=e[e[c0].isin(evs)]
for a,b in zip(m[c0].values, m[c1].values): refs[int(a)].append(int(b))
print("eval nodes with refs:", sum(1 for n in ev if refs[n]), "/", len(ev))
print("mean refs:", np.mean([len(refs[n]) for n in ev]))

X=np.load(B+"data/processed/joint/node_embeddings.npy", mmap_mode='r')
print("embeddings:", X.shape, X.dtype)

need=sorted({n for n in ev} | {r for n in ev for r in refs[n]})
idx={n:i for i,n in enumerate(need)}
print("rows to read:", len(need), f"~{len(need)*X.shape[1]*4/1e6:.0f} MB")
E=np.asarray(X[need], dtype=np.float32)
E/= (np.linalg.norm(E,axis=1,keepdims=True)+1e-9)

T=np.stack([E[idx[n]] for n in ev])                       # target embeddings (945,1024)
def pooled(fn):
    out=[]
    for n in ev:
        r=refs[n]
        out.append(fn(E[[idx[x] for x in r]]) if r else np.zeros(E.shape[1],np.float32))
    P=np.stack(out); P/=(np.linalg.norm(P,axis=1,keepdims=True)+1e-9); return P

def rank_stats(P,T,label):
    S=P@T.T; n=len(T)
    order=np.argsort(-S,axis=1)
    rk=np.array([int(np.where(order[i]==i)[0][0])+1 for i in range(n)])
    print(f"{label:34s} top1={np.mean(rk==1)*100:6.2f}%  top5={np.mean(rk<=5)*100:6.2f}%  "
          f"top10={np.mean(rk<=10)*100:6.2f}%  med={np.median(rk):5.0f}  MRR={np.mean(1/rk):.4f}")

print()
rank_stats(pooled(lambda A:A.mean(0)), T, "mean-pooled refs -> target")
rank_stats(pooled(lambda A:A[:64].mean(0)), T, "mean-pooled first 64 refs")
rank_stats(pooled(lambda A:A.max(0)),  T, "max-pooled refs -> target")
# best single reference (an oracle over the set: what the BEST neighbour could do)
best=[]
for n in ev:
    r=refs[n]
    if not r: best.append(999); continue
    A=E[[idx[x] for x in r]]
    best.append(A)
S=np.stack([ (E[[idx[x] for x in refs[n]]] @ T.T).max(0) if refs[n] else np.zeros(len(ev),np.float32) for n in ev])
order=np.argsort(-S,axis=1)
rk=np.array([int(np.where(order[i]==i)[0][0])+1 for i in range(len(ev))])
print(f"{'best-matching single ref (oracle)':34s} top1={np.mean(rk==1)*100:6.2f}%  top5={np.mean(rk<=5)*100:6.2f}%  "
      f"top10={np.mean(rk<=10)*100:6.2f}%  med={np.median(rk):5.0f}  MRR={np.mean(1/rk):.4f}")
print(f"\nchance top-1 = {100/len(ev):.3f}%")
