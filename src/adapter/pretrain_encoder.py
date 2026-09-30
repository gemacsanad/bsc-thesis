"""
Stage 10a — pretrain the graph encoder with InfoNCE, before the LLM ever sees it.

The task: from a node's masked neighbourhood, pick that node out of a large candidate pool.
Anchor is the encoder's seed output, positive is the seed's own voyage embedding — which
`model.oracle_vector` already returns — and negatives are other nodes' embeddings.

Why this is worth a separate stage
-----------------------------------
The encoder's only learning signal has always travelled through the adapter, forty
transformer layers, and a token-level cross-entropy dominated by generic English. That path
is long, noisy, and points the wrong way: step 6 reached the project's lowest LM loss while
its graph channel was worth ~0.001 nats. Here the gradient path is encoder -> head -> loss.

Two things become possible without the 14B model on the card, and both matter:

  * **batches of hundreds** rather than 1, so InfoNCE has real negatives;
  * **every eligible node**, not just the 7,750 sampled for LM training — this stage needs a
    graph and embeddings, not prompts, targets or a context budget.

It is an initialisation and a diagnostic, never a result. Reporting its retrieval score as
an outcome of the method would be reporting how well voyage embeddings predict themselves.

What it is allowed to be capped by
-----------------------------------
Training the anchor toward `voyage(target)` means information about the target that is not
in `voyage(target)` gets no gradient. That is a real ceiling, and a learned head on the
target side would not remove it — anything applied to `voyage(abstract)` is bounded by
`voyage(abstract)`.

It is accepted here for one measured reason: a plain untrained mean of a node's reference
embeddings already retrieves that node at **64.34% top-1** out of 945, while the full
pipeline delivers 0.32%. The ceiling is roughly fifty times above where we are, so it cannot
bind yet. Stage B drops voyage entirely and ranks real abstracts instead (rank.py).

The number to beat is therefore 64.34%, and this stage is meaningless reported without it:
an encoder that cannot match a plain mean is destroying signal, and that is the finding.

Usage:
    python pretrain_encoder.py --out ../../checkpoints/encoder-gatv2.pt
    python pretrain_encoder.py --gnn-conv sage --out ../../checkpoints/encoder-sage.pt
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import (CHECKPOINT_DIR, FROZEN_DIR, GNN_CONV, GNN_LAYERS, OUT_CHANNELS,
                    PRETRAIN_BATCH, PRETRAIN_EPOCHS, PRETRAIN_LR, PRETRAIN_NEGATIVES,
                    PRETRAIN_TAU, SEED, WANDB_PROJECT)
from graph import load_graph, make_loader
from model import GraphEncoder, oracle_vector

SETS = {"capped": "training_sets_capped.json", "legacy": "training_sets.json"}


def info_nce(z, positives, negatives, positive_ids, negative_ids, tau):
    """-> (loss, top-1 accuracy over the batch).

    Candidates are the batch's own positives followed by the sampled negatives, so anchor i
    is contrasted against every other anchor's target for free. Everything is L2-normalised,
    which makes the dot product a cosine and the temperature meaningful.

    False negatives are masked rather than tolerated: a sampled negative can be another
    anchor's positive, and with 256 anchors drawn from a 7,750-node pool that is not rare
    enough to ignore. Scoring a node's own target as a negative teaches the encoder to push
    away the thing it is supposed to retrieve.
    """
    z = F.normalize(z, dim=-1)
    candidates = F.normalize(torch.cat([positives, negatives]), dim=-1)
    logits = z @ candidates.T / tau

    ids = torch.cat([positive_ids, negative_ids])
    collision = ids.unsqueeze(0) == positive_ids.unsqueeze(1)      # (B, B+N)
    collision[torch.arange(z.size(0)), torch.arange(z.size(0))] = False   # keep the true one
    logits = logits.masked_fill(collision, float("-inf"))

    target = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, target), (logits.argmax(dim=1) == target).float().mean()


@torch.no_grad()
def retrieval(encoder, head, data, node_ids, device, gnn_layers, batch_size):
    """Top-1/top-5/median rank of each node's own embedding among the held-out set.

    The same quantity `src/analysis/refsignal.py` computes for an untrained mean-pool, so the
    two are directly comparable — which is the whole point of running it.
    """
    encoder.eval()
    head.eval()
    anchors, targets = [], []
    for batch in make_loader(data, node_ids, batch_size, shuffle=False, gnn_layers=gnn_layers):
        batch = batch.to(device)
        h = encoder(batch.x, batch.edge_index, batch.batch_size)[:batch.batch_size]
        anchors.append(F.normalize(head(h), dim=-1).float().cpu())
        targets.append(F.normalize(oracle_vector(batch), dim=-1).float().cpu())
    encoder.train()
    head.train()

    a, t = torch.cat(anchors), torch.cat(targets)
    scores = a @ t.T
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == torch.arange(len(a)).unsqueeze(1)).float().argmax(dim=1) + 1
    return {"top1": (ranks == 1).float().mean().item(),
            "top5": (ranks <= 5).float().mean().item(),
            "median_rank": ranks.median().item(),
            "mean_cosine": scores.diag().mean().item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gnn-layers", type=int, default=GNN_LAYERS)
    ap.add_argument("--gnn-conv", choices=["sage", "gatv2"], default=GNN_CONV,
                    help="this stage is cheap enough to settle the choice on evidence; step 6 "
                         "changed it alongside two other things and could attribute nothing")
    ap.add_argument("--sets", choices=sorted(SETS), default="capped")
    ap.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=PRETRAIN_BATCH)
    ap.add_argument("--negatives", type=int, default=PRETRAIN_NEGATIVES)
    ap.add_argument("--tau", type=float, default=PRETRAIN_TAU)
    ap.add_argument("--lr", type=float, default=PRETRAIN_LR)
    ap.add_argument("--out", default=os.path.join(CHECKPOINT_DIR, "encoder_pretrained.pt"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(os.path.join(FROZEN_DIR, SETS[args.sets])) as f:
        sets = json.load(f)
    with open(os.path.join(FROZEN_DIR, "eval_nodes_capped.json")) as f:
        eval_nodes = json.load(f)
    eval_nodes = eval_nodes["nodes"] if isinstance(eval_nodes, dict) else eval_nodes

    data = load_graph()
    print(f"{len(sets['train']):,} train / {len(sets['valid']):,} valid / "
          f"{len(eval_nodes):,} eval nodes; graph {data.num_nodes:,} nodes")

    encoder = GraphEncoder(num_layers=args.gnn_layers, conv=args.gnn_conv).to(device)
    # A single linear on the anchor side, identity on the target side. That keeps the trained
    # vector in voyage space, so retrieval here is directly comparable to the 64.34% mean-pool
    # baseline — an MLP projection head would score a space no baseline exists for. It is also
    # the vector PrefixAdapter consumes, so the guarantee applies where it is needed.
    head = nn.Linear(OUT_CHANNELS, OUT_CHANNELS).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    print(f"{args.gnn_layers}-layer {args.gnn_conv} encoder + linear head: "
          f"{sum(p.numel() for p in params):,} parameters")

    optimizer = AdamW(params, lr=args.lr)
    loader = make_loader(data, sets["train"], args.batch_size, shuffle=True,
                         gnn_layers=args.gnn_layers)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, name=args.wandb_run,
                         config={**vars(args), "stage": "pretrain-encoder", "seed": SEED})

    baseline = retrieval(encoder, head, data, eval_nodes, device, args.gnn_layers,
                         args.batch_size)
    print(f"before training: top1 {baseline['top1']:.2%}  median rank "
          f"{baseline['median_rank']:.0f}   (untrained mean-pool reference: 64.34%)")

    best = -1.0
    for epoch in range(args.epochs):
        total, correct, seen = 0.0, 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            b = batch.batch_size
            h = encoder(batch.x, batch.edge_index, b)[:b]
            z = head(h)

            positives = oracle_vector(batch)
            positive_ids = batch.n_id[:b]
            # Negatives are rows of the embedding matrix — no encoder pass, so thousands of
            # them cost nothing. This asymmetry is what lets the same loss survive into
            # Stage B at micro-batch 1.
            negative_ids = torch.randint(data.num_nodes, (args.negatives,))
            negatives = data.x[negative_ids].to(device)

            loss, acc = info_nce(z, positives, negatives, positive_ids,
                                 negative_ids.to(device), args.tau)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += loss.item() * b
            correct += acc.item() * b
            seen += b

        stats = retrieval(encoder, head, data, eval_nodes, device, args.gnn_layers,
                          args.batch_size)
        print(f"epoch {epoch + 1:>3}  loss {total / seen:.4f}  batch-acc {correct / seen:.3f}  "
              f"eval top1 {stats['top1']:.2%}  top5 {stats['top5']:.2%}  "
              f"median rank {stats['median_rank']:.0f}", flush=True)
        if run is not None:
            run.log({"pretrain/loss": total / seen, "pretrain/batch_accuracy": correct / seen,
                     **{f"pretrain/{k}": v for k, v in stats.items()}, "pretrain/epoch": epoch})

        if stats["top1"] > best:
            best = stats["top1"]
            torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(),
                        "gnn_layers": args.gnn_layers, "gnn_conv": args.gnn_conv,
                        "sets": args.sets, "epoch": epoch + 1, "eval_top1": stats["top1"]},
                       args.out)

    print(f"\nbest eval top-1: {best:.2%}  ->  {args.out}")
    print("reference: an untrained mean of the same references scores 64.34% "
          "(src/analysis/refsignal.py)")
    if best < 0.40:
        print("\nWARNING: below the 40% gate. A plain mean scores 64.34% with no training at "
              "all, so an encoder under 40% is destroying signal rather than extracting it. "
              "That is the finding — investigate before spending GPU hours on Stage B.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
