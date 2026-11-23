"""Train the reader (prefix adapter + LoRA), alone or jointly with the GNN estimator.

The reader is trained with `--oracle`, on the paper's own embedding or its reference mean; joint
fine-tuning adds `--reader <checkpoint> --pretrained-encoder <gnn>`. The exact commands of the
final run are in hpc/komondor/jobs/run11_all.sh.
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

SETS = {
    "capped": "training_sets_capped.json",
    "legacy": "training_sets.json",
    # Every train-split paper (prepare_run9_sets.py), the 806 validation nodes.
    "run9": "training_sets_run9.json",
    # DYTAG_DATASET=children.
    "children": "training_sets.json",
}


def load_dotenv(path):
    """Same helper as archive/graphsage_classification/train_graphsage_full.py — WANDB_KEY lives in a
    gitignored .env rather than on the command line or in the sbatch script.
    """
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
    """A read-only cache: concatenates a fixed prefix and stores nothing."""

    def __init__(self, prefix):
        self.prefix = prefix                      # (batch, layers, 2, heads, tokens, head_dim)
        self.length = prefix.size(4)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        return (torch.cat([self.prefix[:, layer_idx, 0], key], dim=-2),
                torch.cat([self.prefix[:, layer_idx, 1], value], dim=-2))

    def get_seq_length(self, layer_idx=0):
        return self.length

    def get_max_cache_shape(self):
        # `get_max_length` was renamed in newer transformers; None means unbounded.
        return None

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.length


def collate(tokenizer, example, node_ids, device):
    """Tokenise one batch of nodes, right-padded. Labels are -100 on prompt and padding."""
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
    """Cross-entropy over the supervised positions only. -> a scalar, mean over target tokens."""
    # position i predicts token i+1 — the shift `labels=` does internally
    keep = labels[:, 1:] != -100
    logits = llm.lm_head(hidden[:, :-1][keep]).float()
    return F.cross_entropy(logits, labels[:, 1:][keep])


def step_loss(llm, encoder, adapter, ntoken, batch, input_ids, labels, mask, dtype,
              oracle=False, noise=None, refs_mix=None):
    """One forward pass. -> (mean loss over this batch's target tokens, the prefix, the vector the
    adapter was fed; joint fine-tuning scores that vector against the paper's own embedding).
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

    embeds = llm.model.embed_tokens(input_ids)
    if k:
        embeds = torch.cat([neighbour_tokens, embeds], dim=1)
        labels = torch.cat([torch.full((b, k), -100, dtype=labels.dtype, device=device),
                            labels], dim=1)

    attention = torch.cat([torch.ones(b, p + k, dtype=mask.dtype, device=device), mask], dim=1)

    # `llm.model`, not `llm`: the LM head is applied in sliced_loss, on the supervised rows only.
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
    """The contrastive term for one node. -> (loss, whether the gold won)"""
    gold = pool.window(pool.index[node_id], tokens, random_offset).to(prefix.device)
    negs = pool.sample(node_id, negatives, tokens, random_offset).to(prefix.device)
    return rank_loss(llm, prefix, gold, negs, PrefixCache, dtype, pad_id, prompt_ids)


def graph_fraction(adapter, prefix):
    """Mean ‖gate·delta‖ / ‖base‖ over the batch — how much of the prefix the graph supplies."""
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
    """-> (pairwise cosine between the graph deltas, pairwise cosine of the encoder output)"""
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
    """Switch everything to eval with the KV cache on, and back to training afterwards."""
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
    """Generate greedily for a handful of validation nodes. -> (distinct strings, first one)"""
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
    """Identification on validation, used to select best.pt. -> {arm: {top1, mrr}} for each of `arms`"""
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
    """The stop rules end the process with exit code 3, and only they do."""
    print(message, flush=True)
    raise SystemExit(STOP_EXIT_CODE)


def warn(message, out_dir):
    """`--stop-rules warn`: the rule is reported and appended to WARNINGS.txt, training goes on. The
    books' run uses it: their identification check sits below the papers' 10x threshold while the
    vector still lowers validation loss (job 14626133).
    """
    print(f"WARNING, stop rule not enforced: {message}", flush=True)
    with open(os.path.join(out_dir, "WARNINGS.txt"), "a") as f:
        f.write(message + "\n")


def epoch_nodes(nodes, epoch, size, seed=SEED):
    """-> the training nodes of `epoch`: a fixed-size slice of ONE seeded permutation."""
    if not size:
        return nodes
    order = torch.randperm(len(nodes), generator=torch.Generator().manual_seed(seed)).tolist()
    start = epoch * size
    return [nodes[order[(start + i) % len(nodes)]] for i in range(size)]


@torch.no_grad()
def validate(llm, encoder, adapter, ntoken, loader, tokenizer, example, device, dtype,
             oracle=False, pool=None, negatives=RANK_NEGATIVES, tokens=RANK_TOKENS,
             pad_id=None, prompt_ids=None):
    """Token-weighted mean loss with the graph on, and with each gate forced to zero."""
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
    """What early stopping and best.pt read. Higher is better."""
    if "val_id_mrr" in losses:
        return losses["val_id_mrr"]
    if "rank_acc" in losses:
        return losses["rank_acc"]
    return -losses["graph"]


def main():
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

    # LoRA is injected before gradient checkpointing wraps the modules.
    lora_params = []
    if args.lora:
        lora_params = add_lora(llm)
        print(f"LoRA r={LORA_R} on {LORA_TARGETS}: {sum(p.numel() for p in lora_params):,} "
              f"parameters across {len(lora_params) // 2} modules")

    # reentrant checkpointing needs an input that requires grad; with a frozen LLM there is none
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()

    # Joint fine-tuning feeds the adapter the head output, so the head is loaded too.
    encoder = (EmbeddingEstimator if joint else GraphEncoder)(
        num_layers=args.gnn_layers, conv=args.gnn_conv).to(device)
    adapter = PrefixAdapter().to(device)

    if args.pretrained_encoder:
        pre = torch.load(args.pretrained_encoder, map_location=device)
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
        assert reader.get("oracle") and reader.get("sets") == args.sets, (
            f"{args.reader} is not an --oracle reader on the '{args.sets}' sets")
        assert reader.get("node_embeddings") == os.path.basename(embeddings), (
            f"{args.reader} was trained on {reader.get('node_embeddings')}, not {args.features}")
        adapter.load_state_dict(reader["adapter"])
        load_lora_state(llm, reader["lora"])
        print(f"adapter and LoRA initialised from {args.reader} (phase 1, epoch "
              f"{reader['epoch']})")

    ntoken = None
    if args.neighbour_tokens and not args.oracle:
        ntoken = NeighbourTokenAdapter(max_tokens=args.neighbour_tokens).to(device)
        ntoken.match_embeddings(llm.model.embed_tokens.weight)
        print(f"neighbour tokens: up to {args.neighbour_tokens} per node, "
              f"{sum(p.numel() for p in ntoken.parameters()):,} parameters, "
              f"scaled to embedding norm {float(ntoken.scale) * HIDDEN_SIZE ** 0.5:.3f}")

    # --- open the gates -----------------------------------------------------
    probe = next(iter(make_loader(data, sets["train"][:64], 64, shuffle=False,
                                  gnn_layers=args.gnn_layers, hops=hops))).to(device)
    canary_nodes = sets["valid"][:CANARY_NODES]
    val_id_nodes = sets["valid"][:VAL_ID_NODES] if args.oracle or joint else []
    # best.pt is chosen on the reference mean for a reader, on the GNN estimate for joint training.
    if joint:
        val_id_arms = {"gnn": lambda b: encoder(b.x, b.edge_index, b.batch_size)[:b.batch_size],
                       "refs": refs_mean_vector}
        val_id_select, val_id_hops = "gnn", args.gnn_layers
    else:
        val_id_arms = {"oracle": oracle_vector, "refs": refs_mean_vector}
        val_id_select, val_id_hops = "refs", 1
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
                gn = ntoken.calibrate_gate(h_probe[probe.batch_size:], args.gate_init)
                print(f"neighbour gate opened to {gn:.5f}")

    encoder_params = [] if args.oracle else list(encoder.parameters())
    ntoken_params = list(ntoken.parameters()) if ntoken is not None else []
    trainable = encoder_params + list(adapter.parameters()) + ntoken_params + lora_params
    n_trainable = sum(p.numel() for p in trainable)
    which = ("encoder bypassed (oracle)" if args.oracle else
             f"{args.gnn_layers}-layer {args.gnn_conv} encoder over "
             f"{args.gnn_layers} sampled hops")
    print(f"{which}; trainable parameters: {n_trainable:,}")

    # --- the ranking loss's candidates --------------------------------------
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

    start_epoch, opt_step = 0, 0
    best_score, bad_epochs = float("-inf"), 0
    best_path = os.path.join(args.out_dir, "best.pt")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        was = ckpt.get("sets", "legacy")
        assert was == args.sets, (
            f"{args.resume} was trained on the '{was}' node sets but --sets is "
            f"'{args.sets}'; the nodes differ, so this would not be a continuation")
        was_prompt = ckpt.get("prompt", "text")
        assert was_prompt == PROMPT, (
            f"{args.resume} was trained with the '{was_prompt}' prompt but this run uses "
            f"'{PROMPT}'; use a separate --out-dir")
        was_layers = ckpt.get("gnn_layers", 2) 
        assert was_layers == args.gnn_layers, (
            f"{args.resume} was trained with {was_layers} GNN layers but --gnn-layers is "
            f"{args.gnn_layers}; use a separate --out-dir per depth")
        was_conv = ckpt.get("gnn_conv", "sage")
        assert was_conv == args.gnn_conv, (
            f"{args.resume} was trained with the '{was_conv}' encoder but --gnn-conv is "
            f"'{args.gnn_conv}'; the layers differ, so use a separate --out-dir")
        was_ntokens = ckpt.get("neighbour_tokens", 0)
        assert was_ntokens == (args.neighbour_tokens if not args.oracle else 0), (
            f"{args.resume} was trained with {was_ntokens} neighbour tokens but this run "
            f"uses {args.neighbour_tokens}; use a separate --out-dir")
        was_lora = ckpt.get("lora") is not None
        assert was_lora == args.lora, (
            f"{args.resume} was trained {'with' if was_lora else 'without'} LoRA but --lora "
            f"is {'set' if args.lora else 'not set'}; the LLM's weights differ between the "
            f"two, so the prefix would continue against a model it never saw")
        was_oracle = ckpt.get("oracle", False)
        assert was_oracle == args.oracle, (
            f"{args.resume} was trained {'with' if was_oracle else 'without'} --oracle but "
            f"--oracle is {'set' if args.oracle else 'not set'}; one arm sees the target "
            f"abstract and the other does not, so this would not be a continuation")
        was_features = ckpt.get("node_embeddings", "node_embeddings.npy")
        assert was_features == os.path.basename(embeddings), (
            f"{args.resume} was trained on {was_features} but --features is "
            f"{args.features} ({os.path.basename(embeddings)}); the feature distributions "
            f"differ, so this would not be a continuation")
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
        best_score = ckpt.get("best_score", float("-inf"))
        bad_epochs = ckpt.get("bad_epochs", 0)
        print(f"resumed from {args.resume} at epoch {start_epoch}, step {opt_step} "
              f"(best selection score {best_score:.4f}, {bad_epochs} epoch(s) without "
              f"improvement)")

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

            rank_term, rank_hit = None, None
            if pool is not None and node_ids[0] in pool.index:
                rank_term, rank_hit = ranked(llm, prefix[:1], node_ids[0], pool,
                                             args.rank_negatives, args.rank_tokens, dtype,
                                             tokenizer.pad_token_id, prompt_ids,
                                             random_offset=args.rank_offset == "random")
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

        # End-of-epoch validation decides early stopping.
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
                    "node_embeddings": os.path.basename(embeddings),
                    "valid_loss": losses["graph"], "best_score": best_score,
                    "rank_acc": losses.get("rank_acc"),
                    "bad_epochs": bad_epochs}, path)
        print(f"saved {path}")

        if improved:
            shutil.copyfile(path, best_path)
            print(f"saved {best_path} (epoch {epoch + 1}, selection score {best_score:.4f})")

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
