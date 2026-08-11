"""
Stage 4 — train the GraphSAGE encoder and prefix adapter against a frozen Vicuna-7B.

The LLM is frozen and never receives a gradient. The only trainable channel is the prefix:
the graph neighbourhood is encoded, mapped to per-layer key/value prefixes, and injected as
`past_key_values`. Loss is cross-entropy on the target abstract's tokens.

The injection recipe below was verified in `verify_prefix_injection.ipynb`; each part of it
fails *silently* if dropped, so none of it is incidental:

  * `PrefixCache` instead of `DynamicCache` — gradient checkpointing recomputes each layer,
    and a mutating cache appends the body's keys a second time (10 -> 26 -> 42 positions).
  * `attention_mask` of length `NUM_PREFIX_TOKENS + L`, covering the prefix as well.
  * explicit `cache_position`, because `use_cache=False` leaves `past_seen_tokens` at 0 and
    the causal mask is then built as if the prefix were not there.
  * `use_reentrant=False`, because with a frozen LLM no tensor input to layer 0 requires a
    gradient and reentrant checkpointing refuses to build a graph.

Positions run `P .. P+L-1` (the default, `position_ids = cache_position`), which is the same
convention the patched `generate()` produces — train and generate must agree or the prefix is
read at an offset it was never trained for.

Memory on one A100 40GB, bf16 + checkpointing: ~13.5 GB weights, ~1 GB activations, leaving
room for batch 4-8. Without checkpointing the activations alone are 14-22 GB. Those figures
were measured at a ~2,300-token mean; the capped sets average ~3,400, so the headroom is
smaller than it looks.

Node sets
---------
`--sets capped` (default) trains on `training_sets_capped.json` — 7,750 train / 806 valid
nodes at out-degree 5-100 — with prompts trimmed to the centroid-closest references by
`text.capped_example`. `--sets legacy` reproduces the older uncapped 4,000/495 runs.

The choice is written into every checkpoint, and both `--resume` and `generate.py` refuse a
checkpoint trained on the other sets: the prompts differ, so such a pairing would train or
measure against a neighbourhood the adapter never saw, without erroring.

Early stopping
--------------
`--epochs` is an upper bound, not a target. Validation runs at the end of every epoch and the
run stops once `--patience` consecutive epochs fail to improve the graph-on validation loss.
The best epoch is copied to `checkpoints/best.pt`, because when early stopping fires the last
checkpoint is by definition worse than the best one, and the eval job picks by filename.

Best loss and the patience counter are stored in every checkpoint. A 10-epoch run does not fit
one job, so it has to survive the resubmit the walltime forces; without this the counter would
reset on each restart and the run could never stop early.

Usage:
    python adapter/train.py --epochs 10 --patience 2
    python adapter/train.py --epochs 10 --patience 2 --wandb    # track metrics online
    python adapter/train.py --patience 0                        # disable early stopping
    python adapter/train.py --sets legacy                       # the old 4,000/495 run
"""

import argparse
import json
import os
import shutil
import time

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

from config import (CHECKPOINT_DIR, ENV_FILE, FROZEN_DIR, GRAD_CLIP,
                    LEARNING_RATE, NUM_PREFIX_TOKENS, WARMUP_STEPS, WEIGHT_DECAY)
from graph import load_graph, make_loader
from model import GraphSAGEEncoder, PrefixAdapter
from text import Corpus, capped_example, encode

# Which frozen node sets to train on. The two are not interchangeable: the legacy sets were
# selected on the assumption that a node's whole reference list fits the context, the capped
# sets on the assumption that it is trimmed to its centroid (text.capped_example). Pairing
# one set with the other prompt builder silently trains on prompts nobody measured, so the
# choice is recorded in the checkpoint and generate.py refuses a mismatch.
SETS = {
    "capped": ("training_sets_capped.json", True),
    "legacy": ("training_sets.json", False),
}


def load_dotenv(path):
    """Same helper as archive/graphsage_classification/train_graphsage_full.py — WANDB_KEY lives in
    a gitignored .env rather than on the command line or in the sbatch script."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class PrefixCache(Cache):
    """A read-only cache: concatenates a fixed prefix and stores nothing.

    `DynamicCache` cannot be used here. Gradient checkpointing runs each layer twice, and
    `DynamicCache.update` appends on every call, so the second pass sees a cache that has
    already grown by the sequence length. Returning the concatenation without recording it
    makes the layer a pure function of its inputs, which is what recomputation requires.
    """

    def __init__(self, prefix):
        self.prefix = prefix                      # (batch, layers, 2, heads, tokens, head_dim)
        self.length = prefix.size(4)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        return (torch.cat([self.prefix[:, layer_idx, 0], key], dim=-2),
                torch.cat([self.prefix[:, layer_idx, 1], value], dim=-2))

    def get_seq_length(self, layer_idx=0):
        return self.length

    def get_max_length(self):
        return None

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.length


def collate(tokenizer, example, node_ids, device):
    """Tokenise one batch of nodes, right-padded. Labels are -100 on prompt and padding.

    `example` is the (prompt, target) builder for the chosen node sets — capped or legacy.
    """
    seqs = [encode(tokenizer, *example(n)) for n in node_ids]
    width = max(len(ids) for ids, _ in seqs)

    input_ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, (ids, lab) in enumerate(seqs):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        mask[i, :len(ids)] = 1

    return input_ids.to(device), labels.to(device), mask.to(device)


def step_loss(llm, encoder, adapter, batch, input_ids, labels, mask, dtype):
    """One forward pass. Returns (mean loss over this batch's target tokens, the prefix)."""
    graph_vector = encoder(batch.x, batch.edge_index, batch.batch_size)
    prefix = adapter(graph_vector).to(dtype)

    p, (b, length) = NUM_PREFIX_TOKENS, input_ids.shape
    prefix_mask = torch.ones(b, p, dtype=mask.dtype, device=mask.device)

    loss = llm(
        input_ids=input_ids,
        past_key_values=PrefixCache(prefix),
        attention_mask=torch.cat([prefix_mask, mask], dim=1),
        cache_position=torch.arange(p, p + length, device=input_ids.device),
        use_cache=False,
        labels=labels,
    ).loss
    return loss, prefix


def graph_fraction(adapter, prefix):
    """Mean ‖gate·delta‖ / ‖base‖ over the batch — how much of the prefix the graph supplies.

    `base` is the node-independent prefix, so `prefix - base` is exactly the graph's
    contribution. Zero at initialisation, by construction.
    """
    with torch.no_grad():
        delta = (prefix.float() - adapter.base).flatten(1).norm(dim=1)
        return (delta / adapter.base.norm()).mean().item()


@torch.no_grad()
def validate(llm, encoder, adapter, loader, tokenizer, example, device, dtype):
    """Token-weighted mean loss, with the graph on and with the gate forced to zero.

    The second number is the ablation the whole design turns on — same weights, graph
    disconnected — so it is measured every time rather than left to a separate run.
    """
    encoder.eval()
    adapter.eval()
    totals = {}

    for name, zero_gate in (("graph", False), ("gate=0", True)):
        saved = adapter.gate.detach().clone()
        if zero_gate:
            adapter.gate.zero_()

        loss_sum = token_sum = 0
        for batch in loader:
            node_ids = batch.n_id[:batch.batch_size].tolist()
            input_ids, labels, mask = collate(tokenizer, example, node_ids, device)
            n_tokens = int((labels != -100).sum())
            loss, _ = step_loss(llm, encoder, adapter, batch.to(device),
                                input_ids, labels, mask, dtype)
            loss_sum += loss.item() * n_tokens
            token_sum += n_tokens

        adapter.gate.copy_(saved)
        totals[name] = loss_sum / token_sum

    encoder.train()
    adapter.train()
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="lmsys/vicuna-7b-v1.5")
    ap.add_argument("--sets", choices=sorted(SETS), default="capped",
                    help="which frozen node sets to train on; 'legacy' reproduces the "
                         "uncapped 4,000/495 runs")
    ap.add_argument("--epochs", type=int, default=10, help="upper bound; early stopping "
                                                           "usually ends the run sooner")
    ap.add_argument("--patience", type=int, default=2,
                    help="stop after this many consecutive epochs without a better "
                         "validation loss; 0 disables early stopping")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--log-every", type=int, default=10, help="optimizer steps")
    ap.add_argument("--out-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-project", default="dytag-adapter")
    ap.add_argument("--wandb-run", default=None, help="run name; W&B generates one if unset")
    args = ap.parse_args()

    # authenticate before anything expensive — a missing key should not surface after the
    # 13 GB checkpoint has been loaded
    if args.wandb:
        import wandb
        load_dotenv(ENV_FILE)
        wandb.login(key=os.environ["WANDB_KEY"])

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)

    # --- data -------------------------------------------------------------
    sets_file, capped = SETS[args.sets]
    with open(os.path.join(FROZEN_DIR, sets_file)) as f:
        sets = json.load(f)
    print(f"{args.sets} sets ({sets_file}): "
          f"{len(sets['train']):,} train nodes, {len(sets['valid']):,} valid nodes")

    data = load_graph()                                   # stays on CPU; sampling runs there
    train_loader = make_loader(data, sets["train"], args.batch_size, shuffle=True)
    valid_loader = make_loader(data, sets["valid"], args.batch_size, shuffle=False)

    corpus = Corpus()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:                        # Llama ships without one
        tokenizer.pad_token = tokenizer.eos_token

    # data.x is already the voyage embedding matrix the GNN reads, and .numpy() is a view of
    # the same memory — so the centroid distances cost no extra load and no extra 1.6 GB.
    if capped:
        embeddings = data.x.numpy()
        def example(node_id):
            prompt, target, _ = capped_example(tokenizer, corpus, embeddings, node_id)
            return prompt, target
    else:
        example = corpus.example

    # --- models -----------------------------------------------------------
    print(f"Loading {args.model} in {dtype} …")
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    llm.config.use_cache = False
    for p in llm.parameters():
        p.requires_grad_(False)
    # reentrant checkpointing needs an input that requires grad; with a frozen LLM there is none
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()

    encoder = GraphSAGEEncoder().to(device)
    adapter = PrefixAdapter().to(device)
    trainable = list(encoder.parameters()) + list(adapter.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"trainable parameters: {n_trainable:,}")

    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / WARMUP_STEPS)
    )

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={**vars(args), "trainable_parameters": n_trainable,
                    "effective_batch_size": args.batch_size * args.grad_accum,
                    "train_nodes": len(sets["train"]), "valid_nodes": len(sets["valid"]),
                    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                    "warmup_steps": WARMUP_STEPS, "grad_clip": GRAD_CLIP,
                    "num_prefix_tokens": NUM_PREFIX_TOKENS},
        )

    # Early-stopping state lives in the checkpoint: a 10-epoch run does not fit one job, so
    # it has to survive the resubmit that the walltime forces. Without this the patience
    # counter would reset on every restart and the run could never stop early.
    start_epoch, opt_step = 0, 0
    best_loss, bad_epochs = float("inf"), 0
    best_path = os.path.join(args.out_dir, "best.pt")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # The HPC job script resumes from whatever checkpoint is newest, so a stale run on
        # the other node sets is a live hazard rather than a hypothetical one.
        was = ckpt.get("sets", "legacy")     # checkpoints predating --sets are all legacy
        assert was == args.sets, (
            f"{args.resume} was trained on the '{was}' node sets but --sets is "
            f"'{args.sets}'; the prompts differ, so this would not be a continuation")
        encoder.load_state_dict(ckpt["encoder"])
        adapter.load_state_dict(ckpt["adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, opt_step = ckpt["epoch"], ckpt["step"]
        best_loss = ckpt.get("best_loss", float("inf"))
        bad_epochs = ckpt.get("bad_epochs", 0)
        print(f"resumed from {args.resume} at epoch {start_epoch}, step {opt_step} "
              f"(best valid {best_loss:.4f}, {bad_epochs} epoch(s) without improvement)")

    # The job script resubmits blindly after a timeout, so it will eventually resubmit a run
    # that has nothing left to do. Say so and exit, rather than entering an empty epoch loop
    # and failing on the summary that follows it.
    if start_epoch >= args.epochs or (args.patience and bad_epochs >= args.patience):
        why = ("early stopping already fired" if args.patience and bad_epochs >= args.patience
               else f"--epochs is {args.epochs}")
        print(f"nothing to do: checkpoint is at epoch {start_epoch} and {why}")
        if run is not None:
            run.finish()
        return

    # --- train ------------------------------------------------------------
    encoder.train()
    adapter.train()
    running, running_n, tokens, clock = 0.0, 0, 0, time.time()

    for epoch in range(start_epoch, args.epochs):
        for i, batch in enumerate(train_loader):
            node_ids = batch.n_id[:batch.batch_size].tolist()
            input_ids, labels, mask = collate(tokenizer, example, node_ids, device)

            loss, prefix = step_loss(llm, encoder, adapter, batch.to(device),
                                     input_ids, labels, mask, dtype)
            (loss / args.grad_accum).backward()

            running += loss.item()
            running_n += 1
            tokens += int(mask.sum())

            if (i + 1) % args.grad_accum:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % args.log_every == 0:
                elapsed = time.time() - clock
                print(f"epoch {epoch} step {opt_step:>5}  loss {running / running_n:.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}  {tokens / elapsed:,.0f} tok/s",
                      flush=True)
                if run is not None:
                    gate = adapter.gate.detach().float().cpu()
                    run.log({
                        "train/loss": running / running_n,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/tokens_per_sec": tokens / elapsed,
                        "train/grad_norm": grad_norm.item(),   # pre-clip
                        "train/epoch": epoch,
                        # the gate is what turns the graph on; these say when and where
                        "gate/mean_abs": gate.abs().mean().item(),
                        "gate/max_abs": gate.abs().max().item(),
                        "gate/per_layer": wandb.Histogram(gate.numpy()),
                        # snapshot from the last micro-batch, not an average over the window
                        "prefix/graph_fraction": graph_fraction(adapter, prefix),
                    }, step=opt_step)
                running, running_n, tokens, clock = 0.0, 0, 0, time.time()

            if opt_step % args.eval_every == 0:
                losses = validate(llm, encoder, adapter, valid_loader, tokenizer,
                                  example, device, dtype)
                print(f"  valid @ step {opt_step}: graph {losses['graph']:.4f}  "
                      f"gate=0 {losses['gate=0']:.4f}  "
                      f"delta {losses['gate=0'] - losses['graph']:+.4f}", flush=True)
                if run is not None:
                    run.log({"valid/graph": losses["graph"],
                             "valid/gate0": losses["gate=0"],
                             "valid/delta": losses["gate=0"] - losses["graph"]}, step=opt_step)
                clock = time.time()

        # End-of-epoch validation is what early stopping reads. The periodic --eval-every
        # checks above exist for the curve; this one decides whether the run continues.
        losses = validate(llm, encoder, adapter, valid_loader, tokenizer, example,
                          device, dtype)
        improved = losses["graph"] < best_loss
        if improved:
            best_loss, bad_epochs = losses["graph"], 0
        else:
            bad_epochs += 1

        print(f"epoch {epoch + 1} valid: graph {losses['graph']:.4f}  "
              f"gate=0 {losses['gate=0']:.4f}  delta {losses['gate=0'] - losses['graph']:+.4f}"
              f"  {'best' if improved else f'no improvement {bad_epochs}/{args.patience}'}",
              flush=True)
        if run is not None:
            run.log({"valid/graph": losses["graph"], "valid/gate0": losses["gate=0"],
                     "valid/delta": losses["gate=0"] - losses["graph"],
                     "valid/best": best_loss, "valid/bad_epochs": bad_epochs}, step=opt_step)

        path = os.path.join(args.out_dir, f"epoch{epoch + 1}.pt")
        torch.save({"encoder": encoder.state_dict(), "adapter": adapter.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "step": opt_step, "sets": args.sets,
                    "valid_loss": losses["graph"], "best_loss": best_loss,
                    "bad_epochs": bad_epochs}, path)
        print(f"saved {path}")

        # The last epoch is by definition not the best one when early stopping fires, and
        # the eval job picks a checkpoint by name — so the best is copied out under a fixed
        # name rather than left to be identified from the W&B curve months later.
        if improved:
            shutil.copyfile(path, best_path)
            print(f"saved {best_path} (epoch {epoch + 1}, valid {best_loss:.4f})")

        if args.patience and bad_epochs >= args.patience:
            print(f"\nearly stop: {bad_epochs} epochs without improvement on the validation "
                  f"loss (best {best_loss:.4f}); {args.epochs - epoch - 1} epoch(s) skipped")
            break

        clock = time.time()

    print(f"\nbest valid graph loss: {best_loss:.4f}  ->  {best_path}")

    if run is not None:
        run.summary.update({"final/graph": losses["graph"],
                            "final/gate0": losses["gate=0"],
                            "final/delta": losses["gate=0"] - losses["graph"],
                            "best/graph": best_loss,
                            "best/epoch": epoch + 1 - bad_epochs})
        run.finish()


if __name__ == "__main__":
    main()
