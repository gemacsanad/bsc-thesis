"""

Usage:
    python adapter/train.py
    python adapter/train.py --wandb                             # track metrics online
    python adapter/train.py --sets legacy                       # the old 4,000/495 run
    python adapter/train.py --gnn-layers 3 --out-dir checkpoints/gnn3
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

from config import (BASE_MODEL, BATCH_SIZE, CHECKPOINT_DIR, DEFAULT_SETS, ENV_FILE,
                    EPOCHS, EVAL_EVERY, FROZEN_DIR, GNN_LAYERS, GRAD_ACCUM, GRAD_CLIP,
                    LEARNING_RATE, LOG_EVERY, LORA_R, LORA_TARGETS, NUM_PREFIX_TOKENS,
                    PATIENCE, SEED, WANDB_PROJECT, WARMUP_STEPS, WEIGHT_DECAY)
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

    def get_max_cache_shape(self):
        # `get_max_length` in transformers 4.39.3; renamed in the modern Cache API. None means
        # unbounded, which is what a cache that never stores anything is.
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
    # Only what varies between two submissions of the same experiment. Everything else —
    # epochs, patience, batch size, gradient accumulation, logging cadence, seed, model,
    # W&B project — is a constant in config.py, so the job script does not restate it.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnn-layers", type=int, default=GNN_LAYERS,
                    help="GraphSAGE encoder depth; also sets how many hops the loader "
                         "samples, so the two can never disagree. A flag rather than a "
                         "config edit because two depths get submitted minutes apart")
    ap.add_argument("--sets", choices=sorted(SETS), default=DEFAULT_SETS,
                    help="which frozen node sets to train on; 'legacy' reproduces the "
                         "uncapped 4,000/495 runs")
    ap.add_argument("--out-dir", default=CHECKPOINT_DIR,
                    help="one directory per depth; two depths sharing one would overwrite "
                         "each other's best.pt")
    ap.add_argument("--lora", action="store_true",
                    help="also train LoRA adapters on the LLM's q/v projections (lora.py). "
                         "Without it the LLM is frozen, as in every run up to now")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-run", default=None, help="run name; W&B generates one if unset")
    args = ap.parse_args()

    # authenticate before anything expensive — a missing key should not surface after the
    # 13 GB checkpoint has been loaded
    if args.wandb:
        import wandb
        load_dotenv(ENV_FILE)
        wandb.login(key=os.environ["WANDB_KEY"])

    # Imported here, like wandb above, so a run without --lora does not depend on peft.
    if args.lora:
        from lora import add_lora, load_lora_state, lora_state_dict

    torch.manual_seed(SEED)
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
    train_loader = make_loader(data, sets["train"], BATCH_SIZE, shuffle=True,
                               gnn_layers=args.gnn_layers)
    valid_loader = make_loader(data, sets["valid"], BATCH_SIZE, shuffle=False,
                               gnn_layers=args.gnn_layers)

    corpus = Corpus()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
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
        example = lambda node_id: corpus.example(tokenizer, node_id)  # noqa: E731

    # --- models -----------------------------------------------------------
    print(f"Loading {BASE_MODEL} in {dtype} …")
    llm = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    llm.config.use_cache = False
    for p in llm.parameters():
        p.requires_grad_(False)

    # Before checkpointing is enabled, so the wrappers are placed around the final module
    # tree rather than around one the injection then edits underneath them.
    lora_params = []
    if args.lora:
        lora_params = add_lora(llm)
        print(f"LoRA r={LORA_R} on {LORA_TARGETS}: {sum(p.numel() for p in lora_params):,} "
              f"parameters across {len(lora_params) // 2} modules")

    # reentrant checkpointing needs an input that requires grad; with a frozen LLM there is none
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()

    encoder = GraphSAGEEncoder(num_layers=args.gnn_layers).to(device)
    adapter = PrefixAdapter().to(device)
    trainable = list(encoder.parameters()) + list(adapter.parameters()) + lora_params
    n_trainable = sum(p.numel() for p in trainable)
    print(f"{args.gnn_layers}-layer encoder over {args.gnn_layers} sampled hops; "
          f"trainable parameters: {n_trainable:,}")

    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / WARMUP_STEPS)
    )

    run = None
    if args.wandb:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=args.wandb_run,
            # `vars(args)` no longer carries the hyperparameters — they are config.py
            # constants now — so they are named here. W&B is where the runs are compared,
            # and a run whose config omits its own batch size is not comparable to anything.
            config={**vars(args), "trainable_parameters": n_trainable,
                    "model": BASE_MODEL, "seed": SEED,
                    "epochs": EPOCHS, "patience": PATIENCE,
                    "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
                    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
                    "eval_every": EVAL_EVERY, "log_every": LOG_EVERY,
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
        # A depth mismatch would already fail on the conv keys below, but the message would
        # name tensors rather than the mistake — and with several depths sharing one
        # checkpoint tree, picking up the wrong run is the likely way to get here.
        was_layers = ckpt.get("gnn_layers", 2)   # checkpoints predating --gnn-layers are 2-layer
        assert was_layers == args.gnn_layers, (
            f"{args.resume} was trained with {was_layers} GNN layers but --gnn-layers is "
            f"{args.gnn_layers}; use a separate --out-dir per depth")
        # Resuming a LoRA run without --lora would continue a prefix against a different set
        # of LLM weights than it was trained against — no error, just a wrong continuation.
        was_lora = ckpt.get("lora") is not None
        assert was_lora == args.lora, (
            f"{args.resume} was trained {'with' if was_lora else 'without'} LoRA but --lora "
            f"is {'set' if args.lora else 'not set'}; the LLM's weights differ between the "
            f"two, so the prefix would continue against a model it never saw")
        encoder.load_state_dict(ckpt["encoder"])
        adapter.load_state_dict(ckpt["adapter"])
        if args.lora:
            load_lora_state(llm, ckpt["lora"])
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
    if start_epoch >= EPOCHS or (PATIENCE and bad_epochs >= PATIENCE):
        why = ("early stopping already fired" if PATIENCE and bad_epochs >= PATIENCE
               else f"--epochs is {EPOCHS}")
        print(f"nothing to do: checkpoint is at epoch {start_epoch} and {why}")
        if run is not None:
            run.finish()
        return

    # --- train ------------------------------------------------------------
    encoder.train()
    adapter.train()
    running, running_n, tokens, clock = 0.0, 0, 0, time.time()

    for epoch in range(start_epoch, EPOCHS):
        for i, batch in enumerate(train_loader):
            node_ids = batch.n_id[:batch.batch_size].tolist()
            input_ids, labels, mask = collate(tokenizer, example, node_ids, device)

            loss, prefix = step_loss(llm, encoder, adapter, batch.to(device),
                                     input_ids, labels, mask, dtype)
            (loss / GRAD_ACCUM).backward()

            running += loss.item()
            running_n += 1
            tokens += int(mask.sum())

            if (i + 1) % GRAD_ACCUM:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % LOG_EVERY == 0:
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

            if opt_step % EVAL_EVERY == 0:
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
              f"  {'best' if improved else f'no improvement {bad_epochs}/{PATIENCE}'}",
              flush=True)
        if run is not None:
            run.log({"valid/graph": losses["graph"], "valid/gate0": losses["gate=0"],
                     "valid/delta": losses["gate=0"] - losses["graph"],
                     "valid/best": best_loss, "valid/bad_epochs": bad_epochs}, step=opt_step)

        path = os.path.join(args.out_dir, f"epoch{epoch + 1}.pt")
        torch.save({"encoder": encoder.state_dict(), "adapter": adapter.state_dict(),
                    "lora": lora_state_dict(llm) if args.lora else None,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "step": opt_step, "sets": args.sets,
                    "gnn_layers": args.gnn_layers,
                    "valid_loss": losses["graph"], "best_loss": best_loss,
                    "bad_epochs": bad_epochs}, path)
        print(f"saved {path}")

        # The last epoch is by definition not the best one when early stopping fires, and
        # the eval job picks a checkpoint by name — so the best is copied out under a fixed
        # name rather than left to be identified from the W&B curve months later.
        if improved:
            shutil.copyfile(path, best_path)
            print(f"saved {best_path} (epoch {epoch + 1}, valid {best_loss:.4f})")

        if PATIENCE and bad_epochs >= PATIENCE:
            print(f"\nearly stop: {bad_epochs} epochs without improvement on the validation "
                  f"loss (best {best_loss:.4f}); {EPOCHS - epoch - 1} epoch(s) skipped")
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
