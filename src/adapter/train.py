"""

Usage:
    python adapter/train.py
    python adapter/train.py --wandb                             # track metrics online
    python adapter/train.py --sets legacy                       # the old 4,000/495 run
    python adapter/train.py --gnn-layers 3 --out-dir checkpoints/gnn3
    python adapter/train.py --oracle --epochs 2                 # ceiling run, see --oracle
    python adapter/train.py --gnn-conv sage --neighbour-tokens 0  # the step-5 architecture

    # stage 10 — a pretrained encoder, the ranking loss and the gates opened:
    python adapter/train.py --lora --neighbour-tokens 0 \
        --pretrained-encoder ../../checkpoints/encoder-gatv2.pt \
        --gate-init 0.2 --lambda-rank 1.0
    python adapter/train.py --lambda-rank 0 --gate-init 0   # the step-6 objective, for ablation

    # run 9 — the embedding -> abstract decoder (docs/run9_embedding_decoder.md):
    python adapter/train.py --oracle --lora --sets run9 --oracle-noise 0.3 --epoch-size 15000 \
        --neighbour-tokens 0 --lambda-rank 0 --epochs 6

    # run 10a — step 4's recipe (raw features, gate 0) on 90k papers, half fed the reference mean:
    python adapter/train.py --oracle --lora --sets run9 --features raw --refs-mix 0.5 \
        --gate-init 0 --epoch-size 15000 --neighbour-tokens 0 --lambda-rank 0 --epochs 6

    # run 11, phase 3 — joint fine-tune of a phase-1 reader and a phase-2 GNN (--reader):
    python adapter/train.py --lora --sets run9 --features raw --reader <reader best.pt> \
        --pretrained-encoder <gnn.pt> --gnn-conv <its conv> --gnn-layers <its depth> --gate-init 0 \
        --epoch-size 15000 --neighbour-tokens 0 --lambda-rank 0 --epochs 3
"""

import argparse
import json
import os
import shutil
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

from contextlib import contextmanager

from config import (BASE_MODEL, BATCH_SIZE, CANARY_MIN_DISTINCT, CANARY_NODES, CANARY_TOKENS,
                    CHECKPOINT_DIR, CLEAN_TARGETS, DEFAULT_SETS, ENV_FILE, EPOCH_SIZE, EPOCHS, EVAL_EVERY,
                    FEATURES, FROZEN_DIR, ORACLE_MIN_COS, REFS_MIX,
                    GATE_INIT_FRACTION, GNN_CONV, GNN_LAYERS, GRAD_ACCUM, GRAD_CLIP,
                    HIDDEN_SIZE, JOINT_GNN_LR, JOINT_LAMBDA_COS, JOINT_MAX_SUBGRAPH_EDGES,
                    LAMBDA_RANK, LEARNING_RATE,
                    LOG_EVERY, LORA_R, LORA_TARGETS,
                    NUM_NEIGHBOUR_TOKENS, NUM_PREFIX_TOKENS, PATIENCE,
                    RANK_HARD_FRACTION, RANK_NEGATIVES, RANK_TOKENS, RANK_WARMUP_STEPS, SEED,
                    VAL_ID_GATE_EPOCH, VAL_ID_MIN_LIFT, VAL_ID_NODES, VAL_ID_TOKENS,
                    WANDB_PROJECT, WARMUP_STEPS, WEIGHT_DECAY, check_sets)
from graph import (load_graph, make_loader, select_neighbour_rows, with_references,
                   within_subgraph)
from identify import ranks, summary
from model import (EmbeddingEstimator, GraphEncoder, NeighbourTokenAdapter, PrefixAdapter,
                   corrupt_to_cosine, mix_in_refs, oracle_vector, refs_mean_vector)
from prefix_generation import generate_with_prefix, patch_for_prefix
from rank import NegativePool, rank_loss
from text import PROMPT, Corpus, encode, graph_only_example

# Which frozen node sets to train on. Since the soft-prompt-only change the prompt no longer
# depends on the choice — no references are shown — so this selects nodes only. It is still
# recorded in the checkpoint and generate.py refuses a mismatch, because the two sets cover
# different nodes and different out-degree ranges.
SETS = {
    "capped": "training_sets_capped.json",
    "legacy": "training_sets.json",
    # Run 9: every train-split core paper (prepare_run9_sets.py), same 806 validation nodes.
    "run9": "training_sets_run9.json",
    # DYTAG_DATASET=children: src/data/children_books/prepare_sets.py.
    "children": "training_sets.json",
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

    `example` is the (prompt, target) builder — text.graph_only_example, bound to a tokenizer.
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


def sliced_loss(llm, hidden, labels):
    """Cross-entropy over the supervised positions only. -> a scalar, mean over target tokens.

    `llm(..., labels=...)` runs `lm_head` over every position. With Qwen3's 151,936-token
    vocabulary that is a ~1 GB bf16 logits tensor plus its fp32 copy for the loss, on a ~3,400
    token sequence — and roughly 93% of it is thrown away, because labels are -100 across the
    whole prompt. Selecting the hidden states first computes the identical number from ~250
    rows instead of ~3,400. On a 29.6 GB model on a 40 GB card that is the difference between
    fitting and not.

    Identical, not approximately so: this is the same shift, the same fp32 upcast and the same
    mean over non-ignored tokens that `ForCausalLMLoss` performs internally. verify_sliced_loss.py
    is what holds that claim to 1e-4 against the `labels=` path.
    """
    # position i predicts token i+1 — the shift `labels=` does internally
    keep = labels[:, 1:] != -100
    logits = llm.lm_head(hidden[:, :-1][keep]).float()
    return F.cross_entropy(logits, labels[:, 1:][keep])


def step_loss(llm, encoder, adapter, ntoken, batch, input_ids, labels, mask, dtype,
              oracle=False, noise=None, refs_mix=None):
    """One forward pass. -> (mean loss over this batch's target tokens, the prefix, the vector
    the adapter was fed — run 11's joint fine-tune scores that vector against the paper's own
    embedding).

    Two graph channels reach the LLM here, and they are deliberately different kinds of thing:

      * the **pooled** seed embedding becomes a 32-token KV prefix injected at all 40 layers
        (PrefixAdapter) — unchanged since step 2, so every earlier run stays comparable;
      * each **1-hop neighbour** becomes one ordinary input embedding prepended to the prompt
        (NeighbourTokenAdapter), so the 40 layers compute on it rather than only attend to it.

    Sequence layout, and the one cache_position convention generation must match:

        positions 0 .. P-1          KV prefix (in the cache, never in inputs_embeds)
        positions P .. P+k-1        neighbour tokens
        positions P+k .. P+k+L-1    the prompt and the target abstract

    `ntoken` is None in oracle mode: that arm bypasses the encoder entirely, so there are no
    neighbour embeddings to project, and keeping k=0 leaves the ceiling run byte-identical to
    the one already measured.

    `noise` (run 9) is the lowest cosine the oracle vector is rotated to before the adapter
    sees it — model.corrupt_to_cosine. Training passes it; validation does not, so the
    validation loss stays the clean-oracle number.

    `refs_mix` (run 10a) is the probability that a training paper is fed its reference mean
    instead of its own embedding (model.mix_in_refs). Training only, like `noise`.
    """
    p, (b, length) = NUM_PREFIX_TOKENS, input_ids.shape
    device = input_ids.device

    neighbour_tokens, k = None, 0
    if oracle:
        graph_vector = oracle_vector(batch)
        if noise is not None:
            graph_vector = corrupt_to_cosine(graph_vector, noise)
        if refs_mix:
            graph_vector = mix_in_refs(batch, graph_vector, refs_mix)
    else:
        h = encoder(batch.x, batch.edge_index, batch.batch_size)
        graph_vector = h[:batch.batch_size]
        if ntoken is not None:
            rows = select_neighbour_rows(batch, ntoken.max_tokens)
            neighbour_tokens = ntoken(h[rows]).unsqueeze(0).to(dtype)   # (1, k, hidden)
            k = neighbour_tokens.size(1)

    prefix = adapter(graph_vector).to(dtype)

    # Embedding lookup by hand, because the neighbour tokens have no token ids to look up:
    # the model is called with inputs_embeds, so the prompt has to be embedded here too.
    embeds = llm.model.embed_tokens(input_ids)
    if k:
        embeds = torch.cat([neighbour_tokens, embeds], dim=1)
        # -100 over the neighbour positions: they are conditioning, never predicted. The
        # shift in sliced_loss is computed over this padded tensor, so the k rows must be
        # present in it rather than trimmed off the hidden states afterwards.
        labels = torch.cat([torch.full((b, k), -100, dtype=labels.dtype, device=device),
                            labels], dim=1)

    attention = torch.cat([torch.ones(b, p + k, dtype=mask.dtype, device=device), mask], dim=1)

    # `llm.model`, not `llm`: the head is applied in sliced_loss, on the supervised rows only.
    # LoRA is injected in place by peft, so the injected modules are still inside `llm.model`.
    hidden = llm.model(
        inputs_embeds=embeds,
        past_key_values=PrefixCache(prefix),
        attention_mask=attention,
        cache_position=torch.arange(p, p + k + length, device=device),
        use_cache=False,
    ).last_hidden_state
    return sliced_loss(llm, hidden, labels), prefix, graph_vector


def ranked(llm, prefix, node_id, pool, negatives, tokens, dtype, pad_id, prompt_ids=None,
           random_offset=True):
    """The contrastive term for one node. -> (loss, whether the gold won)

    Thin wrapper over rank.rank_loss that owns the sampling, so the training loop and
    validate() draw negatives the same way and neither has to know how.

    `random_offset` slides the scored window along the abstract instead of always taking its
    opening. Training passes True — the opening states the problem, the middle states the
    method, and methodology is what the run is trying to recover. Validation passes False, so
    `rank_acc` measures one fixed thing across epochs and stays comparable with step 7.
    """
    gold = pool.window(pool.index[node_id], tokens, random_offset).to(prefix.device)
    negs = pool.sample(node_id, negatives, tokens, random_offset).to(prefix.device)
    return rank_loss(llm, prefix, gold, negs, PrefixCache, dtype, pad_id, prompt_ids)


def graph_fraction(adapter, prefix):
    """Mean ‖gate·delta‖ / ‖base‖ over the batch — how much of the prefix the graph supplies.

    `base` is the node-independent prefix, so `prefix - base` is exactly the graph's
    contribution. Zero at initialisation, by construction.
    """
    with torch.no_grad():
        delta = (prefix.float() - adapter.base).flatten(1).norm(dim=1)
        return (delta / adapter.base.norm()).mean().item()


def mean_cosine(v):
    """Mean pairwise cosine between the rows of `v`, excluding the diagonal."""
    u = F.normalize(v, dim=1)
    s = u @ u.T
    n = u.size(0)
    return float((s.sum() - s.diagonal().sum()) / (n * (n - 1)))


@torch.no_grad()
def contrast(encoder, adapter, probe, oracle=False):
    """-> (pairwise cosine between the graph deltas, pairwise cosine of the encoder output)

    `graph_fraction` measures the graph channel's AMPLITUDE. It read 1.8 rising to 7.0 through
    the whole of step 7 while every node received the same prefix, because amplitude is not
    contrast and no logged number was measuring contrast. This is that number: 1.0 means every
    node gets an identical delta, which is exactly the failure step 7 shipped.

    Cheap enough to log at LOG_EVERY — the probe batch is sampled once and kept, so this is a
    64-node GNN forward with the 14B model untouched.

    For scale: the oracle, which generates 944 distinct abstracts out of 945, is fed vectors at
    pairwise cosine 0.378; step 7's graph arm was fed 0.660.
    """
    encoder.eval()
    adapter.eval()
    h = (oracle_vector(probe) if oracle
         else encoder(probe.x, probe.edge_index, probe.batch_size))
    g = h[:probe.batch_size].float()
    delta = (adapter(g).float() - adapter.base).flatten(1)
    encoder.train()
    adapter.train()
    return mean_cosine(delta), mean_cosine(g)


@contextmanager
def generation_mode(llm, *modules):
    """Switch everything to eval with the KV cache on, and back to training afterwards.

    Generation needs the KV cache and gradient checkpointing forbids it, so both are toggled
    around the call and restored. Shared by the canary and run 9's identifiability pass.
    `modules` may contain None (an absent neighbour-token adapter).
    """
    modules = [m for m in modules if m is not None]
    llm.gradient_checkpointing_disable()
    llm.config.use_cache = True
    llm.eval()
    for m in modules:
        m.eval()
    try:
        yield
    finally:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        llm.config.use_cache = False
        llm.train()
        for m in modules:
            m.train()


@torch.no_grad()
def canary(llm, encoder, adapter, ntoken, data, node_ids, tokenizer, example, device, dtype,
           gnn_layers, oracle=False):
    """Generate greedily for a handful of validation nodes. -> (distinct strings, first one)

    Not a proxy for collapse — collapse itself, measured the way the evaluation measures it.
    Steps 5, 6 and 7 each shipped a model that wrote one abstract for every paper, and each
    time the training curves looked reasonable: step 7 posted a falling ranking loss and a
    rising rank_hit while emitting 2 distinct openings across 945 nodes. This is one integer,
    logged every EVAL_EVERY steps, that would have said so at epoch 1.

    ~18 s per call for 16 nodes x 48 tokens.
    """
    with generation_mode(llm, encoder, adapter, ntoken):
        outputs = _canary_outputs(llm, encoder, adapter, ntoken, data, node_ids, tokenizer,
                                  example, device, dtype, gnn_layers, oracle)
    return len(set(outputs)), outputs[0]


def _canary_outputs(llm, encoder, adapter, ntoken, data, node_ids, tokenizer, example, device,
                    dtype, gnn_layers, oracle):
    outputs = []
    for batch in make_loader(data, node_ids, 1, shuffle=False, gnn_layers=gnn_layers):
        batch = batch.to(device)
        node_id = int(batch.n_id[0])
        h = encoder(batch.x, batch.edge_index, batch.batch_size)
        g = oracle_vector(batch) if oracle else h[:batch.batch_size]
        tokens = None
        if ntoken is not None:
            tokens = ntoken(h[select_neighbour_rows(batch, ntoken.max_tokens)]).unsqueeze(0)
        prompt, _ = example(node_id)
        input_ids = torch.tensor([tokenizer(prompt).input_ids], device=device)
        new = generate_with_prefix(
            llm, input_ids, adapter(g).to(dtype),
            neighbour_tokens=None if tokens is None else tokens.to(dtype),
            max_new_tokens=CANARY_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        outputs.append(tokenizer.decode(new, skip_special_tokens=True).strip())
    return outputs


@torch.no_grad()
def identifiability(llm, adapter, data, node_ids, tokenizer, example, golds, device, dtype,
                    arms, hops, encoder=None):
    """Run 9's selection metric. -> {arm: {top1, mrr}} for each of `arms`

    Writes VAL_ID_TOKENS tokens for each of `node_ids`, greedily, once per arm — a function of
    the batch returning the vector the adapter is fed. Run 9/10a score the paper's own
    embedding (the clean oracle — can the decoder read a vector at all?) and the plain mean of
    its references' embeddings (the graph arm the thesis reports); run 11's joint fine-tune
    scores its GNN estimate beside that mean. Each text is ranked against the golds of the same
    nodes (identify.py); chance top-1 is 1 / len(node_ids).

    `hops` is what the deepest arm reads: 1 for the reference mean, the encoder depth for a
    GNN. `encoder` is switched to eval mode with the rest when an arm runs it.

    Generated text rather than LM loss, because LM loss is the number that has misled this
    project: step 6 posted the lowest loss on record with the worst conditioning on record.
    """
    texts = {arm: [] for arm in arms}
    with generation_mode(llm, adapter, encoder):
        for batch in make_loader(data, node_ids, 1, shuffle=False, hops=hops):
            batch = batch.to(device)
            prompt, _ = example(int(batch.n_id[0]))
            input_ids = torch.tensor([tokenizer(prompt).input_ids], device=device)
            for arm, vector in arms.items():
                new = generate_with_prefix(llm, input_ids, adapter(vector(batch)).to(dtype),
                                           max_new_tokens=VAL_ID_TOKENS, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
                texts[arm].append(tokenizer.decode(new, skip_special_tokens=True).strip())
    gold = [str(golds[n]) for n in node_ids]
    return {arm: summary(ranks(t, gold)) for arm, t in texts.items()}


STOP_EXIT_CODE = 3


def stop(message):
    """Run 9's stop rules end the process with exit code 3, and only they do.

    A crash exits 1 and a SLURM timeout kills the job, and both of those should simply be
    resubmitted. A stop rule should not: the resume would continue the run that just failed.
    The distinct code is how run_decoder_all.sh tells the two apart and refuses a resubmit.
    """
    print(message, flush=True)
    raise SystemExit(STOP_EXIT_CODE)


def warn(message, out_dir):
    """`--stop-rules warn`: the rule is reported and appended to WARNINGS.txt, training goes on.
    The books' run uses it: their identification check sits below the papers' 10x threshold
    while the vector still lowers validation loss (job 14626133)."""
    print(f"WARNING, stop rule not enforced: {message}", flush=True)
    with open(os.path.join(out_dir, "WARNINGS.txt"), "a") as f:
        f.write(message + "\n")


def epoch_nodes(nodes, epoch, size, seed=SEED):
    """-> the training nodes of `epoch`: a fixed-size slice of ONE seeded permutation.

    Run 9 trains on ~90k papers; a full pass is ~5,600 optimizer steps, too long between
    checkpoints for a 48 h walltime. Slicing one permutation (rather than drawing a fresh
    sample per epoch) means consecutive epochs walk through every node before any repeats,
    and a resumed run gets exactly the slice it would have had. `size` 0 or None is the whole
    set, which is every run before run 9.
    """
    if not size:
        return nodes
    order = torch.randperm(len(nodes), generator=torch.Generator().manual_seed(seed)).tolist()
    start = epoch * size
    return [nodes[order[(start + i) % len(nodes)]] for i in range(size)]


@torch.no_grad()
def validate(llm, encoder, adapter, ntoken, loader, tokenizer, example, device, dtype,
             oracle=False, pool=None, negatives=RANK_NEGATIVES, tokens=RANK_TOKENS,
             pad_id=None, prompt_ids=None):
    """Token-weighted mean loss with the graph on, and with each gate forced to zero.

    Three numbers rather than two since stage 9, because there are now two channels and
    closing them together would not say which one carried anything:

        graph    both channels open — the run's own loss
        gate=0   the KV prefix disconnected; the neighbour tokens still carry the graph
        ngate=0  the k neighbour slots kept but emptied; only the pooled prefix carries it

    These are the ablations the design turns on, so they are measured every epoch rather
    than left to a separate run — step 5 is what happens when the control is missing.

    Since stage 10 it also returns `rank_acc`: how often the node's own abstract outscores M
    wrong ones under its own prefix. **That is what model selection reads, not the loss.**
    Selecting on LM validation loss is what step 6 did, and step 6 had the lowest LM loss ever
    recorded in this project together with its worst conditioning — so that criterion actively
    prefers the generic-template failure. It is computed for the `graph` arm only; under the
    ablations the prefix is node-independent by construction and the answer is chance.
    """
    encoder.eval()
    adapter.eval()
    if ntoken is not None:
        ntoken.eval()
    totals = {}

    gates = [("graph", []), ("gate=0", [adapter.gate])]
    if ntoken is not None:
        gates.append(("ngate=0", [ntoken.gate]))

    for name, zeroed in gates:
        saved = [g.detach().clone() for g in zeroed]
        for g in zeroed:
            g.zero_()

        loss_sum = token_sum = 0
        won = seen = 0
        for batch in loader:
            batch = batch.to(device)
            node_ids = batch.n_id[:batch.batch_size].tolist()
            input_ids, labels, mask = collate(tokenizer, example, node_ids, device)
            n_tokens = int((labels != -100).sum())
            loss, prefix, _ = step_loss(llm, encoder, adapter, ntoken, batch,
                                        input_ids, labels, mask, dtype, oracle)
            loss_sum += loss.item() * n_tokens
            token_sum += n_tokens

            if name == "graph" and pool is not None:
                for row, node_id in enumerate(node_ids):
                    if node_id not in pool.index:
                        continue
                    _, correct = ranked(llm, prefix[row:row + 1], node_id, pool, negatives,
                                        tokens, dtype, pad_id, prompt_ids,
                                        random_offset=False)
                    won += correct
                    seen += 1

        for g, was in zip(zeroed, saved):
            g.copy_(was)
        totals[name] = loss_sum / token_sum
        if name == "graph" and seen:
            totals["rank_acc"] = won / seen

    encoder.train()
    adapter.train()
    if ntoken is not None:
        ntoken.train()
    return totals


def report(losses):
    """The validation line: each channel's loss and what closing it costs."""
    parts = [f"graph {losses['graph']:.4f}"]
    for name in ("gate=0", "ngate=0"):
        if name in losses:
            parts.append(f"{name} {losses[name]:.4f} ({losses[name] - losses['graph']:+.4f})")
    if "rank_acc" in losses:
        parts.append(f"rank {losses['rank_acc']:.3f}")
    return "  ".join(parts)


def logged(losses):
    """The same numbers, keyed for W&B. 'delta' keeps its old meaning — the prefix ablation."""
    out = {"valid/graph": losses["graph"], "valid/gate0": losses["gate=0"],
           "valid/delta": losses["gate=0"] - losses["graph"]}
    if "ngate=0" in losses:
        out["valid/ngate0"] = losses["ngate=0"]
        out["valid/ndelta"] = losses["ngate=0"] - losses["graph"]
    if "rank_acc" in losses:
        out["valid/rank_acc"] = losses["rank_acc"]
    return out


def selection_score(losses):
    """What early stopping and best.pt read. Higher is better.

    Ranking accuracy when it is available, negated LM loss otherwise. The fallback exists only
    for --lambda-rank 0 runs that reproduce the old objective; it is the criterion step 6 used,
    and step 6 posted the project's lowest LM validation loss while its graph channel was worth
    about 0.001 nats. Optimising a metric that is anti-correlated with the goal is how the
    wrong epoch gets shipped.

    Run 9 adds a third, preferred above both: the MRR of text generated from the graph arm the
    thesis reports (`identifiability`) — the plain reference mean for a reader, the GNN
    estimate in run 11's joint fine-tune — read off generated text, on validation nodes only.
    """
    if "val_id_mrr" in losses:
        return losses["val_id_mrr"]
    if "rank_acc" in losses:
        return losses["rank_acc"]
    return -losses["graph"]


def main():
    # Only what varies between two submissions of the same experiment. Everything else —
    # epochs, patience, batch size, gradient accumulation, logging cadence, seed, model,
    # W&B project — is a constant in config.py, so the job script does not restate it.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnn-layers", type=int, default=GNN_LAYERS,
                    help="encoder depth; also sets how many hops the loader "
                         "samples, so the two can never disagree. A flag rather than a "
                         "config edit because two depths get submitted minutes apart")
    ap.add_argument("--gnn-conv", choices=["sage", "gatv2"], default=GNN_CONV,
                    help="message-passing layer: 'gatv2' weights each reference by learned "
                         "attention, 'sage' takes the unweighted mean and reproduces every "
                         "run up to step 5. Recorded in the checkpoint; a resume across a "
                         "change of it is refused, since the parameter names differ")
    ap.add_argument("--neighbour-tokens", type=int, default=NUM_NEIGHBOUR_TOKENS,
                    help="ceiling on 1-hop neighbours given their own input token, chosen "
                         "by centroid distance. 0 disables the channel and reproduces the "
                         "pooled-only runs. Ignored under --oracle, which has no encoder")
    ap.add_argument("--sets", choices=sorted(SETS), default=DEFAULT_SETS,
                    help="which frozen node sets to train on; 'legacy' reproduces the "
                         "uncapped 4,000/495 runs")
    ap.add_argument("--out-dir", default=CHECKPOINT_DIR,
                    help="one directory per depth; two depths sharing one would overwrite "
                         "each other's best.pt")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="upper bound on epochs; a flag rather than a config edit for the "
                         "same reason --gnn-layers is one — it differs between submissions "
                         "that may be queued at the same time. Recorded in the checkpoint "
                         "only through 'epoch', so lowering it on a resume just stops sooner")
    ap.add_argument("--oracle", action="store_true",
                    help="CEILING EXPERIMENT, not a result: bypass the encoder and feed the "
                         "adapter the target abstract's own embedding (model.oracle_vector). "
                         "Measures whether a KV prefix from one dense vector can steer the "
                         "frozen LLM at all. The arm sees the answer, so its score is not "
                         "comparable to any arm that does not")
    ap.add_argument("--oracle-noise", type=float, default=None, metavar="MIN_COS",
                    help="run 9: rotate each training oracle vector to a cosine drawn uniformly "
                         "from [MIN_COS, 1] with the truth (model.corrupt_to_cosine), so the "
                         "decoder learns to write from an estimate of the embedding, which is "
                         "what the graph arm will feed it. --oracle only; validation is clean. "
                         f"config.ORACLE_MIN_COS = {ORACLE_MIN_COS} is the measured value")
    ap.add_argument("--refs-mix", type=float, default=None, metavar="P",
                    help="run 10a: feed each training paper its reference mean instead of its "
                         "own embedding with probability P (model.mix_in_refs). --oracle only; "
                         "validation stays clean. Loads each paper's references, one hop. "
                         f"config.REFS_MIX = {REFS_MIX}")
    ap.add_argument("--features", choices=sorted(FEATURES), default="whitened",
                    help="node features: 'whitened' is run 8 and run 9, 'raw' is every run "
                         "up to step 7 and run 10a. Recorded in the checkpoint")
    ap.add_argument("--epoch-size", type=int, default=None,
                    help="run 9: nodes per epoch, sliced from one seeded permutation of the "
                         "training set (epoch_nodes). Unset is the whole set per epoch, as in "
                         f"every run before run 9. config.EPOCH_SIZE = {EPOCH_SIZE}")
    ap.add_argument("--lora", action="store_true",
                    help="also train LoRA adapters on the LLM's attention projections "
                         "(config.LORA_TARGETS, lora.py). Without it the LLM is frozen. "
                         "Not optional alongside --neighbour-tokens: those are input "
                         "embeddings, and k_proj is what makes them attendable")
    ap.add_argument("--pretrained-encoder", default=None,
                    help="encoder weights from pretrain_encoder.py (stage 10a). The encoder "
                         "arrives already able to identify the target from its neighbourhood "
                         "instead of having to discover that through forty layers and a token "
                         "loss that never asked for it. The head is not loaded: it existed to "
                         "put the anchor in voyage space, and nothing downstream is in it")
    ap.add_argument("--lambda-rank", type=float, default=LAMBDA_RANK,
                    help="weight on the contrastive ranking loss (rank.py). 0 disables it and "
                         "reproduces the step-6 objective. L_LM stays primary and unweighted")
    ap.add_argument("--rank-negatives", type=int, default=RANK_NEGATIVES,
                    help="M wrong abstracts ranked against the gold under each node's prefix")
    ap.add_argument("--rank-tokens", type=int, default=RANK_TOKENS,
                    help="T tokens of each candidate that are scored")
    ap.add_argument("--rank-warmup", type=int, default=RANK_WARMUP_STEPS,
                    help="optimizer steps over which lambda ramps linearly from 0 to its "
                         "full value. L_LM and L_rank genuinely conflict — one rewards likely "
                         "text, the other discriminative text — and step 7 ran at 1.0 from "
                         "step 0 and took a -0.39 coherence regression at exactly the point "
                         "the rank term started winning. 0 restores step 7's behaviour")
    ap.add_argument("--rank-offset", choices=["random", "opening"], default="random",
                    help="where the scored window is cut from. 'opening' reproduces step 7. "
                         "'random' slides it along the abstract, so the objective asks about "
                         "the method and not only the problem statement. Validation always "
                         "uses the opening, whichever this is, so rank_acc stays comparable")
    ap.add_argument("--rank-context", choices=["prefix", "prompt"], default="prefix",
                    help="where candidates are scored. 'prefix' puts them directly behind the "
                         "KV prefix: ~+60%% on top of the LM forward. 'prompt' reproduces the "
                         "training context exactly by replaying the 241-token prompt for every "
                         "candidate, which is ~+460%% — it exists to check that the cheap "
                         "variant ranks the same way, not to train with. See rank.py")
    ap.add_argument("--gate-init", type=float, default=GATE_INIT_FRACTION,
                    help="share of the prefix norm the graph supplies at step 0, calibrated "
                         "on a real batch. 0 reproduces the step-6 initialisation, where "
                         "ignoring the graph was both the default and the starting point")
    ap.add_argument("--reader", default=None,
                    help="run 11, phase 3: start the adapter and LoRA from this --oracle reader "
                         "checkpoint and train them jointly with the GNN + head of "
                         "--pretrained-encoder (model.EmbeddingEstimator), which now feeds the "
                         "adapter. Fresh optimizer; --resume continues such a run")
    ap.add_argument("--lambda-cos", type=float, default=JOINT_LAMBDA_COS,
                    help="--reader only: weight on 1 - cosine(GNN estimate, own embedding), "
                         "kept from phase 2 so the LM loss cannot drag the estimate away from "
                         "the space the reader was trained on")
    ap.add_argument("--gnn-lr", type=float, default=JOINT_GNN_LR,
                    help="--reader only: learning rate of the GNN + head; the adapter and LoRA "
                         f"keep config.LEARNING_RATE = {LEARNING_RATE}")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--stop-rules", choices=["stop", "warn"], default="stop",
                    help="'stop' exits with code 3 when a reader stop rule fires; 'warn' logs it "
                         "to <out-dir>/WARNINGS.txt and keeps training")
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-run", default=None, help="run name; W&B generates one if unset")
    args = ap.parse_args()
    assert args.oracle_noise is None or args.oracle, (
        "--oracle-noise corrupts the oracle vector and means nothing without --oracle")
    assert args.refs_mix is None or args.oracle, (
        "--refs-mix swaps the oracle vector for the reference mean and needs --oracle")
    embeddings = FEATURES[args.features]
    joint = args.reader is not None
    assert not joint or (args.pretrained_encoder and not args.oracle and args.lora), (
        "--reader fine-tunes a LoRA reader together with a pretrained GNN: it needs --lora and "
        "--pretrained-encoder, and cannot be combined with --oracle")

    # authenticate before anything expensive — a missing key should not surface after the
    # 13 GB checkpoint has been loaded
    if args.wandb:
        import wandb
        load_dotenv(ENV_FILE)
        wandb.login(key=os.environ["WANDB_KEY"])

    # Imported here, like wandb above, so a run without --lora does not depend on peft.
    if args.lora:
        from lora import add_lora, load_lora_state, lora_state_dict

    if args.oracle:
        print("=" * 78, "\nORACLE CEILING RUN — the adapter is fed the target abstract's own "
              "embedding.\nThe encoder is bypassed and the seed mask does not apply: this arm "
              "sees the\nanswer. Its score is a ceiling on the prefix channel, never a result.\n"
              + "=" * 78, flush=True)

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)

    # --- data -------------------------------------------------------------
    check_sets(args.sets)
    sets_file = SETS[args.sets]
    with open(os.path.join(FROZEN_DIR, sets_file)) as f:
        sets = json.load(f)
    print(f"{args.sets} sets ({sets_file}): "
          f"{len(sets['train']):,} train nodes, {len(sets['valid']):,} valid nodes")

    data = load_graph(embeddings)                         # stays on CPU; sampling runs there
    print(f"node features: {args.features} ({os.path.basename(embeddings)})")
    # The training loader is built per epoch (epoch_nodes), below. Under --oracle the encoder
    # is bypassed, so both loaders return the seeds alone: run 9 trains on ~90k papers, some
    # with 2-hop neighbourhoods in the tens of thousands, and not one of those rows is read.
    # --refs-mix (run 10a) reads the references, so its training loader takes one hop.
    hops = 0 if args.oracle else None
    train_hops = 1 if args.refs_mix else hops
    valid_loader = make_loader(data, sets["valid"], BATCH_SIZE, shuffle=False,
                               gnn_layers=args.gnn_layers, hops=hops)

    corpus = Corpus(clean=CLEAN_TARGETS)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:                        # Llama ships without one
        tokenizer.pad_token = tokenizer.eos_token

    # No references in the prompt: the graph prefix is the only channel to the target paper.
    example = lambda node_id: graph_only_example(tokenizer, corpus, node_id)  # noqa: E731

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

    # Run 11's joint fine-tune feeds the adapter the head output, so the head comes too.
    encoder = (EmbeddingEstimator if joint else GraphEncoder)(
        num_layers=args.gnn_layers, conv=args.gnn_conv).to(device)
    adapter = PrefixAdapter().to(device)

    if args.pretrained_encoder:
        pre = torch.load(args.pretrained_encoder, map_location=device)
        # The same two mismatches --resume guards, for the same reason: the parameter names
        # differ between convs and between depths, so a wrong file half-loads rather than
        # failing, and the run would train a partly random encoder as if it were pretrained.
        assert pre.get("gnn_conv", "sage") == args.gnn_conv, (
            f"{args.pretrained_encoder} pretrained a '{pre.get('gnn_conv')}' encoder but "
            f"--gnn-conv is '{args.gnn_conv}'")
        assert pre.get("gnn_layers", 2) == args.gnn_layers, (
            f"{args.pretrained_encoder} pretrained {pre.get('gnn_layers')} layers but "
            f"--gnn-layers is {args.gnn_layers}")
        if joint:
            was_features = pre.get("node_embeddings", "node_embeddings_whitened.npy")
            assert was_features == os.path.basename(embeddings), (
                f"{args.pretrained_encoder} was trained on {was_features} but --features is "
                f"{args.features}")
            encoder.encoder.load_state_dict(pre["encoder"])
            encoder.head.load_state_dict(pre["head"])
            print(f"GNN + head initialised from {args.pretrained_encoder} (phase 2, epoch "
                  f"{pre['epoch']}, validation cosine {pre['valid']['mean_cosine']:.4f})")
        else:
            encoder.load_state_dict(pre["encoder"])
            print(f"encoder initialised from {args.pretrained_encoder} "
                  f"(stage 10a, eval top-1 {pre.get('eval_top1', float('nan')):.2%})")

    if joint and not args.resume:
        reader = torch.load(args.reader, map_location=device)
        # The reader must have been trained to decode embedding-space vectors on the same
        # features and nodes; anything else loads cleanly and decodes a space it never saw.
        assert reader.get("oracle") and reader.get("sets") == args.sets, (
            f"{args.reader} is not an --oracle reader on the '{args.sets}' sets")
        assert reader.get("node_embeddings") == os.path.basename(embeddings), (
            f"{args.reader} was trained on {reader.get('node_embeddings')}, not {args.features}")
        adapter.load_state_dict(reader["adapter"])
        load_lora_state(llm, reader["lora"])
        print(f"adapter and LoRA initialised from {args.reader} (phase 1, epoch "
              f"{reader['epoch']})")

    # None disables the channel entirely rather than running it with k=0: the module would
    # otherwise sit in the optimizer collecting no gradient, and its (untrained) weights
    # would ride into the checkpoint as if the arm had used them. Under --oracle there is
    # no encoder output to project, so the ceiling run keeps the layout it was measured on.
    ntoken = None
    if args.neighbour_tokens and not args.oracle:
        ntoken = NeighbourTokenAdapter(max_tokens=args.neighbour_tokens).to(device)
        # Scale the tokens to the norm of a real Qwen3 embedding before any training —
        # an out-of-distribution input embedding degrades quietly, it does not raise.
        ntoken.match_embeddings(llm.model.embed_tokens.weight)
        print(f"neighbour tokens: up to {args.neighbour_tokens} per node, "
              f"{sum(p.numel() for p in ntoken.parameters()):,} parameters, "
              f"scaled to embedding norm {float(ntoken.scale) * HIDDEN_SIZE ** 0.5:.3f}")

    # --- open the gates -----------------------------------------------------
    # Calibrated on a real batch rather than set to a literal: `delta`'s scale depends on the
    # encoder's output distribution, and a pretrained encoder's differs from a random one's.
    # Step 6 reached graph_fraction 0.70 at a gate of 0.0076, so a hard-coded value would mean
    # something different for every run.
    # Sampled once and kept: `contrast` runs on it at every logging step, and a NeighborLoader
    # call per log would cost more than the GNN forward it feeds.
    probe = next(iter(make_loader(data, sets["train"][:64], 64, shuffle=False,
                                  gnn_layers=args.gnn_layers, hops=hops))).to(device)
    # 16 validation nodes, fixed for the run, so the canary's distinct count means the same
    # thing at every step. Validation nodes rather than training ones: a collapse the model has
    # memorised its way out of on the training set is still a collapse.
    canary_nodes = sets["valid"][:CANARY_NODES]
    # Run 9's selection set: fixed validation nodes, so the score means the same thing at
    # every epoch. Only under --oracle, where the decoder reads embedding-space vectors and
    # the reference mean is a meaningful input; every other run keeps its old selection.
    val_id_nodes = sets["valid"][:VAL_ID_NODES] if args.oracle or joint else []
    # The arms it writes from, and the one best.pt is chosen by: the reference mean for a
    # reader, the GNN estimate for the joint fine-tune.
    if joint:
        val_id_arms = {"gnn": lambda b: encoder(b.x, b.edge_index, b.batch_size)[:b.batch_size],
                       "refs": refs_mean_vector}
        val_id_select, val_id_hops = "gnn", args.gnn_layers
    else:
        val_id_arms = {"oracle": oracle_vector, "refs": refs_mean_vector}
        val_id_select, val_id_hops = "refs", 1
    # generate() has to be told the cache starts non-empty. Installed once here; it only
    # affects llm.generate, never the forward passes training runs.
    patch_for_prefix(llm, NUM_PREFIX_TOKENS,
                     max_extra=0 if ntoken is None else ntoken.max_tokens)

    if args.gate_init > 0 and not args.resume and not joint:
        with torch.no_grad():
            h_probe = (oracle_vector(probe) if args.oracle
                       else encoder(probe.x, probe.edge_index, probe.batch_size))
            g = adapter.calibrate_gate(h_probe[:probe.batch_size], args.gate_init)
            print(f"prefix gate opened to {g:.5f} "
                  f"(graph supplies ~{args.gate_init:.0%} of the prefix norm at step 0)")
            if ntoken is not None:
                # rows batch_size.. are the sampled neighbours of this same batch, which is
                # exactly the distribution the token adapter will be fed.
                gn = ntoken.calibrate_gate(h_probe[probe.batch_size:], args.gate_init)
                print(f"neighbour gate opened to {gn:.5f}")

    # In oracle mode the encoder is bypassed, so its parameters would sit in the optimizer
    # collecting no gradient. Excluding them keeps the reported count honest.
    encoder_params = [] if args.oracle else list(encoder.parameters())
    ntoken_params = list(ntoken.parameters()) if ntoken is not None else []
    trainable = encoder_params + list(adapter.parameters()) + ntoken_params + lora_params
    n_trainable = sum(p.numel() for p in trainable)
    which = ("encoder bypassed (oracle)" if args.oracle else
             f"{args.gnn_layers}-layer {args.gnn_conv} encoder over "
             f"{args.gnn_layers} sampled hops")
    print(f"{which}; trainable parameters: {n_trainable:,}")

    # --- the ranking loss's candidates --------------------------------------
    # Over the training AND validation nodes. It was training-only through step 7, and that was
    # a bug with a silent consequence: validate() skips any node not in `pool.index`, so no
    # validation node was ever scored, `rank_acc` was never computed, and `selection_score`
    # fell back to LM validation loss — the criterion step 6 proved is anti-correlated with
    # conditioning. best.pt was chosen by exactly the wrong number. A validation node can now
    # be drawn as some training node's random negative, which is harmless: it is a wrong
    # abstract like any other, and `sample` still cannot draw a node against itself.
    pool = prompt_ids = None
    if args.lambda_rank > 0 and not args.oracle:
        pool = NegativePool(tokenizer, corpus, sets["train"] + sets["valid"], seed=SEED)
        if args.rank_context == "prompt":
            prompt_ids = torch.tensor(tokenizer(example(sets["train"][0])[0]).input_ids,
                                      device=device)
        print(f"ranking loss: lambda {args.lambda_rank} over {args.rank_warmup} warmup steps, "
              f"M={args.rank_negatives} negatives "
              f"(~{RANK_HARD_FRACTION:.0%} from each node's own references), "
              f"T={args.rank_tokens} tokens from a {args.rank_offset} window, scored behind "
              f"the {args.rank_context}")
        print(f"negative pool: {len(pool.node_ids):,} candidates "
              f"({pool.dropped} dropped as under 30 words)")
    elif args.oracle:
        # The oracle is fed the target's own embedding, so ranking its own abstract first is
        # not a measurement of anything. Silently training on it would make the ceiling look
        # like a method.
        print("ranking loss disabled under --oracle (the arm already sees the answer)")

    # The joint fine-tune moves the pretrained GNN more slowly than the reader around it.
    groups = ([{"params": encoder_params, "lr": args.gnn_lr},
               {"params": [p for p in trainable if all(p is not q for q in encoder_params)]}]
              if joint else trainable)
    optimizer = AdamW(groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
                    "epochs": args.epochs, "patience": PATIENCE,
                    "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
                    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
                    "eval_every": EVAL_EVERY, "log_every": LOG_EVERY,
                    "train_nodes": len(sets["train"]), "valid_nodes": len(sets["valid"]),
                    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                    "warmup_steps": WARMUP_STEPS, "grad_clip": GRAD_CLIP,
                    "num_prefix_tokens": NUM_PREFIX_TOKENS,
                    "gnn_conv": args.gnn_conv,
                    "neighbour_tokens": ntoken.max_tokens if ntoken is not None else 0,
                    "lora_r": LORA_R, "lora_targets": list(LORA_TARGETS),
                    "oracle_noise": args.oracle_noise, "epoch_size": args.epoch_size,
                    "refs_mix": args.refs_mix, "joint": joint,
                    "lambda_cos": args.lambda_cos if joint else None,
                    "gnn_lr": args.gnn_lr if joint else None},
        )

    # Early-stopping state lives in the checkpoint: a 10-epoch run does not fit one job, so
    # it has to survive the resubmit that the walltime forces. Without this the patience
    # counter would reset on every restart and the run could never stop early.
    start_epoch, opt_step = 0, 0
    # Higher is better since stage 10: this is ranking accuracy, not a loss. Initialised to
    # -inf so the first epoch always improves on it, whichever criterion is in use.
    best_score, bad_epochs = float("-inf"), 0
    best_path = os.path.join(args.out_dir, "best.pt")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # The HPC job script resumes from whatever checkpoint is newest, so a stale run on
        # the other node sets is a live hazard rather than a hypothetical one.
        was = ckpt.get("sets", "legacy")     # checkpoints predating --sets are all legacy
        assert was == args.sets, (
            f"{args.resume} was trained on the '{was}' node sets but --sets is "
            f"'{args.sets}'; the nodes differ, so this would not be a continuation")
        # The one mismatch nothing else catches: a Qwen text-path checkpoint has exactly the
        # shapes of a graph-only one, so it would load cleanly and continue a prefix trained to
        # sit beside ~13 abstracts as if it had to carry the paper alone.
        was_prompt = ckpt.get("prompt", "text")  # checkpoints predating PROMPT read abstracts
        assert was_prompt == PROMPT, (
            f"{args.resume} was trained with the '{was_prompt}' prompt but this run uses "
            f"'{PROMPT}'; use a separate --out-dir")
        # A depth mismatch would already fail on the conv keys below, but the message would
        # name tensors rather than the mistake — and with several depths sharing one
        # checkpoint tree, picking up the wrong run is the likely way to get here.
        was_layers = ckpt.get("gnn_layers", 2)   # checkpoints predating --gnn-layers are 2-layer
        assert was_layers == args.gnn_layers, (
            f"{args.resume} was trained with {was_layers} GNN layers but --gnn-layers is "
            f"{args.gnn_layers}; use a separate --out-dir per depth")
        # A conv mismatch fails on the parameter names below, but only after the 29.6 GB
        # model is on the card, and the message would name tensors rather than the mistake.
        was_conv = ckpt.get("gnn_conv", "sage")  # everything before stage 9 was GraphSAGE
        assert was_conv == args.gnn_conv, (
            f"{args.resume} was trained with the '{was_conv}' encoder but --gnn-conv is "
            f"'{args.gnn_conv}'; the layers differ, so use a separate --out-dir")
        # Neighbour tokens change the sequence the prefix sits beside, so a prefix trained
        # with them would continue against a shorter context it never saw — fluent, no error.
        was_ntokens = ckpt.get("neighbour_tokens", 0)   # nothing before stage 9 had them
        assert was_ntokens == (args.neighbour_tokens if not args.oracle else 0), (
            f"{args.resume} was trained with {was_ntokens} neighbour tokens but this run "
            f"uses {args.neighbour_tokens}; use a separate --out-dir")
        # Resuming a LoRA run without --lora would continue a prefix against a different set
        # of LLM weights than it was trained against — no error, just a wrong continuation.
        was_lora = ckpt.get("lora") is not None
        assert was_lora == args.lora, (
            f"{args.resume} was trained {'with' if was_lora else 'without'} LoRA but --lora "
            f"is {'set' if args.lora else 'not set'}; the LLM's weights differ between the "
            f"two, so the prefix would continue against a model it never saw")
        # An oracle checkpoint and a graph one have identical shapes, so this is the only
        # thing standing between a ceiling run and a graph run continuing each other.
        was_oracle = ckpt.get("oracle", False)
        assert was_oracle == args.oracle, (
            f"{args.resume} was trained {'with' if was_oracle else 'without'} --oracle but "
            f"--oracle is {'set' if args.oracle else 'not set'}; one arm sees the target "
            f"abstract and the other does not, so this would not be a continuation")
        # Whitened and raw features have identical shapes, so this loads cleanly and
        # continues a trained encoder on a distribution it has never seen.
        was_features = ckpt.get("node_embeddings", "node_embeddings.npy")
        assert was_features == os.path.basename(embeddings), (
            f"{args.resume} was trained on {was_features} but --features is "
            f"{args.features} ({os.path.basename(embeddings)}); the feature distributions "
            f"differ, so this would not be a continuation")
        # Run 9. The noise defines the training distribution of the decoder's input, and the
        # epoch size defines which nodes each epoch slices; resuming across a change of either
        # would continue a different run under the same directory.
        # Run 11: a joint checkpoint holds GNN + head under "encoder", a reader's a bare GNN.
        assert ckpt.get("joint", False) == joint, (
            f"{args.resume} was {'' if ckpt.get('joint') else 'not '}a --reader joint run; "
            f"use a separate --out-dir")
        for key, now in (("oracle_noise", args.oracle_noise), ("epoch_size", args.epoch_size),
                         ("refs_mix", args.refs_mix)):
            assert ckpt.get(key) == now, (
                f"{args.resume} was trained with {key}={ckpt.get(key)} but this run uses "
                f"{now}; use a separate --out-dir")
        encoder.load_state_dict(ckpt["encoder"])
        adapter.load_state_dict(ckpt["adapter"])
        if ntoken is not None:
            ntoken.load_state_dict(ckpt["ntoken"])
        if args.lora:
            load_lora_state(llm, ckpt["lora"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, opt_step = ckpt["epoch"], ckpt["step"]
        # Checkpoints predating stage 10 stored best_loss (lower better); their value is not
        # comparable to a ranking accuracy, so a resume across the change restarts selection
        # rather than inheriting a number that means something else.
        best_score = ckpt.get("best_score", float("-inf"))
        bad_epochs = ckpt.get("bad_epochs", 0)
        print(f"resumed from {args.resume} at epoch {start_epoch}, step {opt_step} "
              f"(best selection score {best_score:.4f}, {bad_epochs} epoch(s) without "
              f"improvement)")

    # The job script resubmits blindly after a timeout, so it will eventually resubmit a run
    # that has nothing left to do. Say so and exit, rather than entering an empty epoch loop
    # and failing on the summary that follows it.
    if start_epoch >= args.epochs or (PATIENCE and bad_epochs >= PATIENCE):
        why = ("early stopping already fired" if PATIENCE and bad_epochs >= PATIENCE
               else f"--epochs is {args.epochs}")
        print(f"nothing to do: checkpoint is at epoch {start_epoch} and {why}")
        if run is not None:
            run.finish()
        return

    # --- train ------------------------------------------------------------
    encoder.train()
    adapter.train()
    if ntoken is not None:
        ntoken.train()
    running, running_n, tokens, clock = 0.0, 0, 0, time.time()
    rank_running, rank_hits, rank_n = 0.0, 0, 0

    # A GNN estimate for a paper with no references is the [MASK] row alone (graph.py).
    train_nodes = with_references(data, sets["train"]) if joint else sets["train"]
    if joint and JOINT_MAX_SUBGRAPH_EDGES is not None:
        kept = within_subgraph(data, train_nodes, JOINT_MAX_SUBGRAPH_EDGES, args.gnn_layers)
        print(f"{len(train_nodes) - len(kept):,} training nodes over {JOINT_MAX_SUBGRAPH_EDGES:,} "
              f"{args.gnn_layers}-hop subgraph edges left out")
        train_nodes = kept
    if joint:
        print(f"joint fine-tune on {len(train_nodes):,} training papers with references; "
              f"loss = LM + {args.lambda_cos} x (1 - cosine to own embedding), GNN lr "
              f"{args.gnn_lr:.0e}")
    cos_running = 0.0

    for epoch in range(start_epoch, args.epochs):
        train_loader = make_loader(data, epoch_nodes(train_nodes, epoch, args.epoch_size),
                                   BATCH_SIZE, shuffle=True, gnn_layers=args.gnn_layers,
                                   hops=train_hops)
        for i, batch in enumerate(train_loader):
            node_ids = batch.n_id[:batch.batch_size].tolist()
            input_ids, labels, mask = collate(tokenizer, example, node_ids, device)

            batch = batch.to(device)
            lm_loss, prefix, graph_vector = step_loss(
                llm, encoder, adapter, ntoken, batch, input_ids, labels, mask, dtype,
                args.oracle, noise=args.oracle_noise, refs_mix=args.refs_mix)

            # L_LM stays primary and unweighted; the ranking term is what makes the objective
            # ask about *this* paper at all. They genuinely conflict — one rewards likely text,
            # the other discriminative text — so both are logged separately rather than only
            # their sum, and lambda is doing real work.
            rank_term, rank_hit = None, None
            if pool is not None and node_ids[0] in pool.index:
                rank_term, rank_hit = ranked(llm, prefix[:1], node_ids[0], pool,
                                             args.rank_negatives, args.rank_tokens, dtype,
                                             tokenizer.pad_token_id, prompt_ids,
                                             random_offset=args.rank_offset == "random")
            # Ramped rather than constant: the two terms conflict, and giving L_LM the first
            # epoch alone lets the register settle before the ranking term starts pulling
            # against it. It reaches full weight at the end of epoch 1 by construction.
            lam = args.lambda_rank * (min(1.0, opt_step / args.rank_warmup)
                                      if args.rank_warmup else 1.0)
            loss = lm_loss if rank_term is None else lm_loss + lam * rank_term
            if joint:
                cos_term = 1 - F.cosine_similarity(
                    graph_vector.float(), oracle_vector(batch).float(), dim=-1).mean()
                loss = loss + args.lambda_cos * cos_term
                cos_running += cos_term.item()
            (loss / GRAD_ACCUM).backward()

            running += lm_loss.item()
            running_n += 1
            tokens += int(mask.sum())
            if rank_term is not None:
                rank_running += rank_term.item()
                rank_hits += rank_hit
                rank_n += 1

            if (i + 1) % GRAD_ACCUM:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % LOG_EVERY == 0:
                elapsed = time.time() - clock
                delta_cos, encoder_cos = contrast(encoder, adapter, probe, args.oracle)
                extra = (f"  rank {rank_running / rank_n:.4f} (hit {rank_hits / rank_n:.2f})"
                         if rank_n else "")
                print(f"epoch {epoch} step {opt_step:>5}  loss {running / running_n:.4f}{extra}"
                      f"  contrast {delta_cos:.3f}"
                      f"  lr {scheduler.get_last_lr()[-1]:.2e}  {tokens / elapsed:,.0f} tok/s",
                      flush=True)
                if run is not None:
                    gate = adapter.gate.detach().float().cpu()
                    run.log({
                        "train/loss": running / running_n,
                        "train/lr": scheduler.get_last_lr()[-1],
                        "train/tokens_per_sec": tokens / elapsed,
                        "train/grad_norm": grad_norm.item(),   # pre-clip
                        "train/epoch": epoch,
                        # the gate is what turns the graph on; these say when and where
                        "gate/mean_abs": gate.abs().mean().item(),
                        "gate/max_abs": gate.abs().max().item(),
                        "gate/per_layer": wandb.Histogram(gate.numpy()),
                        **({"gate/neighbour": float(ntoken.gate)}
                           if ntoken is not None else {}),
                        # snapshot from the last micro-batch, not an average over the window
                        "prefix/graph_fraction": graph_fraction(adapter, prefix),
                        # and the number graph_fraction is NOT: amplitude said 7.0 through the
                        # whole of step 7 while every node got the same prefix. 1.0 here means
                        # the channel is constant, whatever its loudness.
                        "prefix/graph_contrast": delta_cos,
                        "encoder/output_cosine": encoder_cos,
                        **({"train/rank_loss": rank_running / rank_n,
                            "train/rank_hit": rank_hits / rank_n,
                            "train/lambda_rank": lam} if rank_n else {}),
                        # the GNN estimate's distance from the paper's own embedding
                        **({"train/cos_loss": cos_running / running_n} if joint else {}),
                    }, step=opt_step)
                running, running_n, tokens, clock = 0.0, 0, 0, time.time()
                cos_running = 0.0
                rank_running, rank_hits, rank_n = 0.0, 0, 0

            if opt_step % EVAL_EVERY == 0:
                losses = validate(llm, encoder, adapter, ntoken, valid_loader, tokenizer,
                                  example, device, dtype, args.oracle, pool,
                                  args.rank_negatives, args.rank_tokens,
                                  tokenizer.pad_token_id, prompt_ids)
                distinct, first = canary(llm, encoder, adapter, ntoken, data, canary_nodes,
                                         tokenizer, example, device, dtype, args.gnn_layers,
                                         args.oracle)
                print(f"  valid @ step {opt_step}: {report(losses)}  "
                      f"canary {distinct}/{len(canary_nodes)} distinct", flush=True)
                print(f"    >> {first[:160]}", flush=True)
                if run is not None:
                    run.log({**logged(losses), "canary/distinct": distinct}, step=opt_step)
                clock = time.time()

        # End-of-epoch validation is what early stopping reads. The periodic --eval-every
        # checks above exist for the curve; this one decides whether the run continues.
        losses = validate(llm, encoder, adapter, ntoken, valid_loader, tokenizer, example,
                          device, dtype, args.oracle, pool, args.rank_negatives,
                          args.rank_tokens, tokenizer.pad_token_id, prompt_ids)
        distinct, _ = canary(llm, encoder, adapter, ntoken, data, canary_nodes, tokenizer,
                             example, device, dtype, args.gnn_layers, args.oracle)
        val_id = None
        if val_id_nodes:
            val_id = identifiability(llm, adapter, data, val_id_nodes, tokenizer, example,
                                     corpus.abstract, device, dtype, val_id_arms, val_id_hops,
                                     encoder if joint else None)
            losses["val_id_mrr"] = val_id[val_id_select]["mrr"]
            print(f"  identifiability over {len(val_id_nodes)} validation nodes "
                  f"(chance top-1 {1 / len(val_id_nodes):.3f}): "
                  + "  ".join(f"{arm} top1 {v['top1']:.3f} mrr {v['mrr']:.3f}"
                              for arm, v in val_id.items()), flush=True)
            if run is not None:
                run.log({f"val_id/{arm}_{k}": x for arm, v in val_id.items()
                         for k, x in v.items()}, step=opt_step)
        score = selection_score(losses)
        improved = score > best_score
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1

        print(f"epoch {epoch + 1} valid: {report(losses)}"
              f"  canary {distinct}/{len(canary_nodes)} distinct"
              f"  {'best' if improved else f'no improvement {bad_epochs}/{PATIENCE}'}",
              flush=True)
        if run is not None:
            run.log({**logged(losses), "valid/best": best_score,
                     "canary/distinct": distinct,
                     "valid/bad_epochs": bad_epochs}, step=opt_step)

        path = os.path.join(args.out_dir, f"epoch{epoch + 1}.pt")
        torch.save({"encoder": encoder.state_dict(), "adapter": adapter.state_dict(),
                    "ntoken": ntoken.state_dict() if ntoken is not None else None,
                    "lora": lora_state_dict(llm) if args.lora else None,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "step": opt_step, "sets": args.sets, "prompt": PROMPT,
                    "gnn_layers": args.gnn_layers, "gnn_conv": args.gnn_conv,
                    "neighbour_tokens": ntoken.max_tokens if ntoken is not None else 0,
                    "oracle": args.oracle,
                    "oracle_noise": args.oracle_noise, "epoch_size": args.epoch_size,
                    "refs_mix": args.refs_mix,
                    "joint": joint, "reader": args.reader,
                    "pretrained_encoder": args.pretrained_encoder,
                    "lora_targets": list(LORA_TARGETS) if args.lora else None,
                    "val_id": val_id, "canary_distinct": distinct,
                    # Which node features the encoder was trained on. Whitened and raw features
                    # have identical shapes, so a checkpoint loaded against the wrong ones
                    # would run a trained encoder on a distribution it has never seen and
                    # produce fluent text rather than an error. generate.py refuses a mismatch.
                    "node_embeddings": os.path.basename(embeddings),
                    "valid_loss": losses["graph"], "best_score": best_score,
                    "rank_acc": losses.get("rank_acc"),
                    "bad_epochs": bad_epochs}, path)
        print(f"saved {path}")

        # The last epoch is by definition not the best one when early stopping fires, and
        # the eval job picks a checkpoint by name — so the best is copied out under a fixed
        # name rather than left to be identified from the W&B curve months later.
        if improved:
            shutil.copyfile(path, best_path)
            print(f"saved {best_path} (epoch {epoch + 1}, selection score {best_score:.4f})")

        # Run 9's stop rules — code, not a sentence in a plan. Run 8's plan said "flat at
        # chance by step 800 means stop"; it was flat by step 200 and ran to the end. The epoch
        # is already saved, so the failure can be inspected; the job then exits non-zero.
        if args.oracle and epoch + 1 >= VAL_ID_GATE_EPOCH:
            fail = stop if args.stop_rules == "stop" else lambda m: warn(m, args.out_dir)
            if distinct < CANARY_MIN_DISTINCT:
                fail(
                    f"STOP: canary wrote {distinct} distinct texts for {len(canary_nodes)} "
                    f"papers at epoch {epoch + 1} (minimum {CANARY_MIN_DISTINCT}). The decoder "
                    f"is collapsing to a template; resubmitting would only continue it.")
            floor = VAL_ID_MIN_LIFT / len(val_id_nodes)
            if val_id["oracle"]["top1"] < floor:
                fail(
                    f"STOP: from its own embedding the decoder identifies "
                    f"{val_id['oracle']['top1']:.3f} of validation papers at epoch {epoch + 1}, "
                    f"below {VAL_ID_MIN_LIFT}x chance ({floor:.3f}). It cannot read the "
                    f"vector, so no graph estimate of that vector can help.")

        if PATIENCE and bad_epochs >= PATIENCE:
            print(f"\nearly stop: {bad_epochs} epochs without improvement on the validation "
                  f"selection score (best {best_score:.4f}); "
                  f"{args.epochs - epoch - 1} epoch(s) skipped")
            break

        clock = time.time()

    print(f"\nbest selection score: {best_score:.4f}  ->  {best_path}")

    if run is not None:
        run.summary.update({**{k.replace("valid/", "final/"): v
                               for k, v in logged(losses).items()},
                            "best/score": best_score,
                            "best/epoch": epoch + 1 - bad_epochs})
        run.finish()


if __name__ == "__main__":
    main()
