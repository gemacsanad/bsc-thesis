"""Does 'closest to centroid first' keep the informative references?"""
import csv, numpy as np, collections, pandas as pd
B="/Users/gemacsanad/egyetem/DyTAG/baseline/"
rows=list(csv.DictReader(open(B+"experiments/2026-09-25-14450409_ntok64-gatv2-lora64/evaluated_result.csv")))
ev=[int(r['node_id']) for r in rows]; evs=set(ev)
e=pd.read_csv(B+"data/processed/joint/edges.csv"); c0,c1=e.columns[:2]
refs=collections.defaultdict(list)
m=e[e[c0].isin(evs)]
for a,b in zip(m[c0].values,m[c1].values): refs[int(a)].append(int(b))
X=np.load(B+"data/processed/joint/node_embeddings.npy", mmap_mode='r')
need=sorted(evs|{r for n in ev for r in refs[n]}); idx={n:i for i,n in enumerate(need)}
E=np.asarray(X[need],dtype=np.float32); E/=(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
T=np.stack([E[idx[n]] for n in ev])

def centroid_dist(A):
    c=A.mean(0); c/=np.linalg.norm(c)+1e-9
    return (1-A@c)/2

def stats(P,label):
    P=P/(np.linalg.norm(P,axis=1,keepdims=True)+1e-9)
    S=P@T.T; n=len(T); order=np.argsort(-S,axis=1)
    rk=np.array([int(np.where(order[i]==i)[0][0])+1 for i in range(n)])
    print(f"  {label:32s} top1={np.mean(rk==1)*100:6.2f}%  med={np.median(rk):4.0f}  MRR={np.mean(1/rk):.4f}")

hi=[n for n in ev if len(refs[n])>64]
print(f"nodes with >64 refs: {len(hi)}/{len(ev)}  (the only ones the cap touches)\n")
rng=np.random.default_rng(0)
for label, sel in [("ALL refs (no cap)", lambda A:A),
                   ("64 NEAREST centroid (current)", lambda A:A[np.argsort(centroid_dist(A))[:64]]),
                   ("64 FARTHEST centroid", lambda A:A[np.argsort(-centroid_dist(A))[:64]]),
                   ("64 RANDOM", lambda A:A[rng.permutation(len(A))[:64]])]:
    P=[]
    for n in ev:
        A=E[[idx[x] for x in refs[n]]]
        P.append(sel(A).mean(0) if len(A) else np.zeros(E.shape[1],np.float32))
    stats(np.stack(P), label)

print("\nrestricted to the %d nodes where the cap actually bites:"%len(hi))
Th=np.stack([E[idx[n]] for n in hi])
def stats2(P,label):
    P=P/(np.linalg.norm(P,axis=1,keepdims=True)+1e-9)
    S=P@Th.T; n=len(Th); order=np.argsort(-S,axis=1)
    rk=np.array([int(np.where(order[i]==i)[0][0])+1 for i in range(n)])
    print(f"  {label:32s} top1={np.mean(rk==1)*100:6.2f}%  med={np.median(rk):4.0f}  MRR={np.mean(1/rk):.4f}")
for label, sel in [("ALL refs (no cap)", lambda A:A),
                   ("64 NEAREST centroid (current)", lambda A:A[np.argsort(centroid_dist(A))[:64]]),
                   ("64 FARTHEST centroid", lambda A:A[np.argsort(-centroid_dist(A))[:64]]),
                   ("64 RANDOM", lambda A:A[rng.permutation(len(A))[:64]])]:
    P=[sel(E[[idx[x] for x in refs[n]]]).mean(0) for n in hi]
    stats2(np.stack(P), label)
