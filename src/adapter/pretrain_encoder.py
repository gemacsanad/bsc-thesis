"""Train the GNN estimator: predict a paper's own text embedding from its references.

Usage (the final run trains eight variants this way, see hpc/komondor/jobs/run11_all.sh):
    python pretrain_encoder.py --sets run9 --features raw --lambda-cos 1 --lambda-nce 0 \
        --vicreg-var 0 --vicreg-cov 0 --cosine-gate 1 --out ../../checkpoints/run11-gnn/gatv2-l2-cos.pt
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import (CHECKPOINT_DIR, ENCODER_COSINE_GATE, FEATURES, FROZEN_DIR, GNN_CONV,
                    GNN_LAYERS, OUT_CHANNELS, PRETRAIN_BATCH, PRETRAIN_EPOCHS, PRETRAIN_LR,
                    PRETRAIN_NEGATIVES, PRETRAIN_TAU, SEED, VICREG_COV, VICREG_GAMMA,
                    VICREG_VAR, WANDB_PROJECT, check_sets)
from graph import load_graph, make_loader, with_references
from model import GraphEncoder, oracle_vector, refs_mean_vector

SETS = {"capped": "training_sets_capped.json", "legacy": "training_sets.json",
        "run9": "training_sets_run9.json", "children": "training_sets.json"}
# Every paper set is checked against the same 945 eval papers; the books have their own.
EVAL = {"children": "eval_nodes.json"}


def info_nce(z, positives, negatives, positive_ids, negative_ids, tau):
    """-> (loss, top-1 accuracy over the batch)."""
    z = F.normalize(z, dim=-1)
    candidates = F.normalize(torch.cat([positives, negatives]), dim=-1)
    logits = z @ candidates.T / tau

    ids = torch.cat([positive_ids, negative_ids])
    collision = ids.unsqueeze(0) == positive_ids.unsqueeze(1)      # (B, B+N)
    collision[torch.arange(z.size(0)), torch.arange(z.size(0))] = False   # keep the true one
    logits = logits.masked_fill(collision, float("-inf"))

    target = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, target), (logits.argmax(dim=1) == target).float().mean()


def vicreg(z, gamma=VICREG_GAMMA):
    """-> (variance hinge, off-diagonal covariance) over the batch. Bardes, Ponce & LeCun 2021."""
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    variance = F.relu(gamma - std).mean()
    cov = (z.T @ z) / (z.size(0) - 1)
    off_diagonal = cov.pow(2).sum() - cov.pow(2).diagonal().sum()
    return variance, off_diagonal / z.size(1)


@torch.no_grad()
def retrieval(encoder, head, data, node_ids, device, gnn_layers, batch_size, refs_mean=False):
    """Top-1/top-5/median rank of each node's own embedding among the held-out set."""
    encoder.eval()
    head.eval()
    anchors, targets = [], []
    for batch in make_loader(data, node_ids, batch_size, shuffle=False,
                             gnn_layers=gnn_layers, hops=1 if refs_mean else None):
        batch = batch.to(device)
        if refs_mean:
            anchors.append(refs_mean_vector(batch).float().cpu())
        else:
            h = encoder(batch.x, batch.edge_index, batch.batch_size)[:batch.batch_size]
            anchors.append(F.normalize(head(h), dim=-1).float().cpu())
        targets.append(F.normalize(oracle_vector(batch), dim=-1).float().cpu())
    encoder.train()
    head.train()

    a, t = torch.cat(anchors), torch.cat(targets)
    scores = a @ t.T
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == torch.arange(len(a)).unsqueeze(1)).float().argmax(dim=1) + 1
    pairwise = a @ a.T
    n = len(a)
    return {"top1": (ranks == 1).float().mean().item(),
            "top5": (ranks <= 5).float().mean().item(),
            "median_rank": ranks.median().item(),
            "mean_cosine": scores.diag().mean().item(),
            "output_cosine": ((pairwise.sum() - pairwise.diagonal().sum())
                              / (n * (n - 1))).item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gnn-layers", type=int, default=GNN_LAYERS)
    ap.add_argument("--gnn-conv", choices=["sage", "gatv2"], default=GNN_CONV,
                    help="this stage is cheap enough to settle the choice on evidence; step 6 "
                         "changed it alongside two other things and could attribute nothing")
    ap.add_argument("--sets", choices=sorted(SETS), default="capped")
    ap.add_argument("--features", choices=sorted(FEATURES), default="whitened",
                    help="node features: 'whitened' is run 8, 'raw' is run 11, whose reader "
                         "was trained on raw embeddings. Recorded in the checkpoint")
    ap.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=PRETRAIN_BATCH)
    ap.add_argument("--negatives", type=int, default=PRETRAIN_NEGATIVES)
    ap.add_argument("--tau", type=float, default=PRETRAIN_TAU)
    ap.add_argument("--lambda-nce", type=float, default=1.0,
                    help="weight on InfoNCE; 0 disables it (run 11's cosine-only variants)")
    ap.add_argument("--lambda-cos", type=float, default=0.0,
                    help="weight on 1 - cosine(head output, own embedding); run 11's main loss. "
                         "0 reproduces the run-8 objective")
    ap.add_argument("--vicreg-var", type=float, default=VICREG_VAR,
                    help="weight on the per-dimension variance hinge; 0 disables it and "
                         "reproduces the step-7 objective")
    ap.add_argument("--vicreg-cov", type=float, default=VICREG_COV,
                    help="weight on off-diagonal covariance")
    ap.add_argument("--cosine-gate", type=float, default=ENCODER_COSINE_GATE,
                    help="an epoch whose output pairwise cosine exceeds this cannot be "
                         "selected, whatever its score. The oracle, which generates 944 "
                         "distinct abstracts, is fed vectors at 0.378. 1 disables it: run 10a's "
                         "refs-mean arm was fed 0.660 and still wrote distinct abstracts")
    ap.add_argument("--lr", type=float, default=PRETRAIN_LR)
    ap.add_argument("--out", default=os.path.join(CHECKPOINT_DIR, "encoder_pretrained.pt"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    check_sets(args.sets)
    with open(os.path.join(FROZEN_DIR, SETS[args.sets])) as f:
        sets = json.load(f)
    with open(os.path.join(FROZEN_DIR, EVAL.get(args.sets, "eval_nodes_capped.json"))) as f:
        eval_nodes = json.load(f)
    eval_nodes = eval_nodes["nodes"] if isinstance(eval_nodes, dict) else eval_nodes

    data = load_graph(FEATURES[args.features])
    # Papers without references have only the [MASK] row to estimate from and are dropped.
    train_nodes = with_references(data, sets["train"])
    print(f"{len(train_nodes):,} train ({len(sets['train']) - len(train_nodes):,} without "
          f"references dropped) / {len(sets['valid']):,} valid / {len(eval_nodes):,} eval nodes; "
          f"graph {data.num_nodes:,} nodes, {args.features} features")

    encoder = GraphEncoder(num_layers=args.gnn_layers, conv=args.gnn_conv).to(device)
    head = nn.Linear(OUT_CHANNELS, OUT_CHANNELS).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    print(f"{args.gnn_layers}-layer {args.gnn_conv} encoder + linear head: "
          f"{sum(p.numel() for p in params):,} parameters; loss {args.lambda_cos} x cosine + "
          f"{args.lambda_nce} x InfoNCE + VICReg ({args.vicreg_var}, {args.vicreg_cov})")

    optimizer = AdamW(params, lr=args.lr)
    loader = make_loader(data, train_nodes, args.batch_size, shuffle=True,
                         gnn_layers=args.gnn_layers)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, name=args.wandb_run,
                         config={**vars(args), "stage": "pretrain-encoder", "seed": SEED})

    def measure(nodes, refs_mean=False):
        return retrieval(encoder, head, data, nodes, device, args.gnn_layers, args.batch_size,
                         refs_mean)

    # Selection reads the validation papers only; the eval set is touched once, at the end.
    reference = measure(sets["valid"], refs_mean=True)
    print(f"validation, plain reference mean: cosine {reference['mean_cosine']:.4f}  "
          f"top1 {reference['top1']:.2%}  output cosine {reference['output_cosine']:.3f}")
    if run is not None:
        run.log({f"pretrain/refs_mean_{k}": v for k, v in reference.items()})

    best, best_stats = -1.0, None
    for epoch in range(args.epochs):
        total, cos_total, correct, seen = 0.0, 0.0, 0.0, 0
        var_total, cov_total = 0.0, 0.0
        for batch in loader:
            batch = batch.to(device)
            b = batch.batch_size
            h = encoder(batch.x, batch.edge_index, b)[:b]
            z = head(h)

            positives = oracle_vector(batch)
            positive_ids = batch.n_id[:b]
            negative_ids = torch.randint(data.num_nodes, (args.negatives,))
            negatives = data.x[negative_ids].to(device)

            nce, acc = info_nce(z, positives, negatives, positive_ids,
                                negative_ids.to(device), args.tau)
            cosine = F.cosine_similarity(z, positives.float(), dim=-1).mean()
            variance, covariance = vicreg(z)
            loss = (args.lambda_nce * nce + args.lambda_cos * (1 - cosine)
                    + args.vicreg_var * variance + args.vicreg_cov * covariance)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += nce.item() * b
            cos_total += cosine.item() * b
            var_total += variance.item() * b
            cov_total += covariance.item() * b
            correct += acc.item() * b
            seen += b

        stats = measure(sets["valid"])
        print(f"epoch {epoch + 1:>3}  nce {total / seen:.4f}  train cos {cos_total / seen:.4f}  "
              f"var {var_total / seen:.4f}  cov {cov_total / seen:.4f}  "
              f"batch-acc {correct / seen:.3f}  valid cos {stats['mean_cosine']:.4f} "
              f"(mean {reference['mean_cosine']:.4f})  top1 {stats['top1']:.2%} "
              f"(mean {reference['top1']:.2%})  output cos {stats['output_cosine']:.3f}",
              flush=True)
        if run is not None:
            run.log({"pretrain/loss": total / seen, "pretrain/train_cosine": cos_total / seen,
                     "pretrain/batch_accuracy": correct / seen,
                     "pretrain/variance": var_total / seen,
                     "pretrain/covariance": cov_total / seen,
                     **{f"pretrain/{k}": v for k, v in stats.items()}, "pretrain/epoch": epoch})

        if stats["output_cosine"] > args.cosine_gate:
            continue
        if stats["mean_cosine"] > best:
            best, best_stats = stats["mean_cosine"], stats
            torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(),
                        "gnn_layers": args.gnn_layers, "gnn_conv": args.gnn_conv,
                        "sets": args.sets, "epoch": epoch + 1,
                        "node_embeddings": os.path.basename(FEATURES[args.features]),
                        "lambda_cos": args.lambda_cos, "lambda_nce": args.lambda_nce,
                        "valid": stats, "valid_refs_mean": reference,
                        "output_cosine": stats["output_cosine"]},
                       args.out)

    if best < 0:
        if run is not None:
            run.finish()
        raise SystemExit(
            f"FAILED: no epoch produced an output pairwise cosine below {args.cosine_gate}. "
            f"The encoder ranks but does not spread, which is what step 7 shipped — a "
            f"near-constant input the adapter reads as a constant.")

    # The eval set, once, for the selected epoch: reported, never selected on.
    saved = torch.load(args.out, map_location=device)
    encoder.load_state_dict(saved["encoder"])
    head.load_state_dict(saved["head"])
    final, final_ref = measure(eval_nodes), measure(eval_nodes, refs_mean=True)
    saved["eval"], saved["eval_refs_mean"] = final, final_ref
    torch.save(saved, args.out)
    if run is not None:
        run.summary.update({**{f"best/valid_{k}": v for k, v in best_stats.items()},
                            **{f"best/eval_{k}": v for k, v in final.items()},
                            "best/epoch": saved["epoch"]})
        run.finish()

    print(f"\nbest epoch {saved['epoch']} -> {args.out}")
    print(f"  valid: cosine {best_stats['mean_cosine']:.4f} vs mean {reference['mean_cosine']:.4f}"
          f"   top1 {best_stats['top1']:.2%} vs mean {reference['top1']:.2%}")
    print(f"  eval:  cosine {final['mean_cosine']:.4f} vs mean {final_ref['mean_cosine']:.4f}"
          f"   top1 {final['top1']:.2%} vs mean {final_ref['top1']:.2%}")
    if best_stats["mean_cosine"] <= reference["mean_cosine"]:
        print("\nWARNING: no better than the plain reference mean on validation. The encoder is "
              "destroying signal rather than extracting it — investigate before Stage B.")


if __name__ == "__main__":
    main()
