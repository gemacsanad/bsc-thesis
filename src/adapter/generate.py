"""
Stage 5 — generate abstracts for a frozen evaluation set.

Produces one CSV per condition, over the same nodes, with the same prompts, in the same order:

    --mode baseline   frozen Qwen3-14B, nothing else                 -> generations.csv
    --mode adapter    GraphSAGE + prefix adapter + frozen Qwen3-14B  -> generations.csv

Each mode writes into its own directory under experiments/; pass --output to name the run.

Since the soft-prompt-only change (2026-09-16) the prompt carries no references, so
`--mode baseline` receives no information about the paper at all: under greedy decoding it
writes the same abstract for every node. It is a floor, not a control. The comparison arm for
the graph-only adapter is the 2026-09-10 text-path Qwen baseline — see docs/soft_prompt_only.md.

The two runs differ in exactly one thing — whether a graph-conditioned KV prefix is present.
Everything else is held fixed on purpose:

  * **Prompts** come from `text.graph_only_example`, the same builder training used: one
    fixed system message and user turn, identical for every node. Neither the target paper's
    title nor any reference text is in it; the prefix is the only channel to the paper.
  * **Greedy decoding** (`do_sample=False`) with `REPETITION_PENALTY`, from one `decoding`
    dict shared by both branches so they cannot drift apart. The penalty is deterministic, so
    reruns still reproduce; it is applied to both conditions, so the repair cannot be mistaken
    for an adapter effect — but the degeneration it repairs is the adapter's (84 of its 945
    outputs repeat a whole sentence, against 0 for the frozen LLM). Sampling noise would sit
    on top of the effect being measured. The older `experiments/_archive/vicuna-7b-8bit`
    baseline sampled (temperature 0.35); it is therefore not a drop-in comparison for these
    two, which is why the frozen-LLM condition is regenerated here rather than reused.
  * **bfloat16, not 8-bit.** `src/generation/generate_vicuna.py` loaded the baseline in 8-bit, but
    the adapter was trained against a bf16 model and its prefix is calibrated to those
    numerics. Both conditions therefore run bf16, so quantisation error cannot be mistaken
    for an adapter effect.
  * **Batch size 1.** Padded batches combined with a pre-filled prefix cache were never
    verified. One node at a time removes the
    question entirely.

Node sets
---------
`--sets capped` (default) generates for the 945 nodes of `eval_nodes_capped.json`;
`--sets legacy` for the older 495 nodes of `eval_nodes.json`. With no references in the prompt
the choice selects nodes only. Only 2 nodes overlap, and a checkpoint records which sets it was
trained on, so a mismatched pairing is refused rather than silently measured.

Context overflow
----------------
The graph-only prompt is a few hundred tokens against a 4,096 context, so it cannot overflow.
The budget check and its ERROR row are kept for that reason alone: if one ever fires, the
prompt path has changed.

Usage:
    cd src/adapter
    python generate.py --mode baseline --output ../../experiments/<run>/generations.csv
    python generate.py --mode adapter  --output ../../experiments/<run>/generations.csv \
                       --checkpoint ../../checkpoints/epoch2.pt

Both are resumable: rerunning appends only the node_ids missing from the output CSV.
"""

import argparse
import csv
import json
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (BASE_MODEL, CHECKPOINT_DIR, DEFAULT_SETS, EXPERIMENTS_DIR,
                    FROZEN_DIR, GNN_LAYERS, JOINT_DIR,
                    MAX_LENGTH, MAX_NEW_TOKENS, NUM_PREFIX_TOKENS, REPETITION_PENALTY)
from graph import load_graph, make_loader
from model import GraphSAGEEncoder, PrefixAdapter
from prefix_generation import generate_with_prefix, patch_for_prefix
from text import PROMPT, Corpus, graph_only_example

# Mirrors train.py's SETS: which frozen eval nodes.
SETS = {
    "capped": "eval_nodes_capped.json",
    "legacy": "eval_nodes.json",
}


def remap_encoder(state):
    """`conv1`/`conv2` -> `convs.0`/`convs.1` for checkpoints written before the encoder
    became a ModuleList (4105279, 2026-08-19, the commit that added --gnn-layers).

    Only the attribute name moved: at two layers both definitions build the same pair of
    1024-wide `SAGEConv(aggr="mean")` layers, with relu and dropout between them and no
    activation on the output. `state` is returned unchanged unless an old-style key is
    actually present, so checkpoints from every run since 08-19 do not go through this.
    """
    if not any(re.match(r"conv\d+\.", key) for key in state):
        return state
    print("checkpoint predates the ModuleList encoder — remapping convN. -> convs.N-1.")
    return {re.sub(r"^conv(\d+)\.", lambda m: f"convs.{int(m.group(1)) - 1}.", key): value
            for key, value in state.items()}


def load_adapter(path, device, sets, llm):
    """Restore the trained encoder and adapter from a `train.py` checkpoint.

    Also returns the encoder depth, which is read off the checkpoint rather than passed in:
    the same number has to build the encoder and set the loader's hop count, and there is no
    reason to let the command line disagree with the weights being loaded.

    If the checkpoint carries LoRA, it is injected into `llm` here. That is not optional and
    there is no flag for it: the prefix was trained against those LLM weights, so generating
    without them measures a prefix on a model it never saw — fluent text, no error.
    """
    checkpoint = torch.load(path, map_location=device)
    # A capped-trained prefix evaluated on legacy prompts (or the reverse) produces fluent
    # text from a neighbourhood the adapter was never calibrated on — silent, not an error.
    was = checkpoint.get("sets", "legacy")   # checkpoints predating --sets are all legacy
    assert was == sets, (
        f"{path} was trained on the '{was}' node sets but --sets is '{sets}'; "
        f"the nodes differ, so this pairing would not measure the trained adapter")
    # A text-path Qwen checkpoint has the same shapes and would load cleanly, then be scored
    # without the abstracts its prefix was trained to sit beside.
    was_prompt = checkpoint.get("prompt", "text")   # predating PROMPT means abstracts were shown
    assert was_prompt == PROMPT, (
        f"{path} was trained with the '{was_prompt}' prompt but generation uses '{PROMPT}'")
    gnn_layers = checkpoint.get("gnn_layers", 2)   # predating --gnn-layers means 2-layer
    encoder = GraphSAGEEncoder(num_layers=gnn_layers).to(device)
    adapter = PrefixAdapter().to(device)
    encoder.load_state_dict(remap_encoder(checkpoint["encoder"]))
    adapter.load_state_dict(checkpoint["adapter"])
    encoder.eval()
    adapter.eval()

    lora = checkpoint.get("lora")     # None for every checkpoint trained without --lora
    if lora is not None:
        from lora import add_lora, load_lora_state
        add_lora(llm)
        load_lora_state(llm, lora)

    print(f"loaded {path} (epoch {checkpoint['epoch']}, step {checkpoint['step']}, "
          f"{gnn_layers}-layer encoder, "
          f"{'LoRA restored' if lora is not None else 'frozen LLM'})")
    return encoder, adapter, gnn_layers


@torch.no_grad()
def iter_prefixes(node_ids, encoder, adapter, data, device, gnn_layers):
    """Yield `(node_id, prefix)` for each node, in the order given. `prefix` is None in
    baseline mode.

    In adapter mode the order comes from `NeighborLoader(shuffle=False)`, and the node_id is
    read back off the batch rather than assumed, so a row can never be paired with another
    node's neighbourhood.
    """
    if encoder is None:
        for node_id in node_ids:
            yield node_id, None
        return

    for batch in make_loader(data, node_ids, batch_size=1, shuffle=False,
                             gnn_layers=gnn_layers):
        batch = batch.to(device)
        graph_vector = encoder(batch.x, batch.edge_index, batch.batch_size)
        yield int(batch.n_id[0]), adapter(graph_vector)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["baseline", "adapter", "gate0"], required=True)
    ap.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "epoch2.pt"),
                    help="train.py checkpoint; --mode adapter only")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--output", default=None,
                    help="default: experiments/<mode>/generations.csv; pass a run directory "
                         "explicitly so each run keeps its own folder")
    ap.add_argument("--sets", choices=sorted(SETS), default=DEFAULT_SETS,
                    help="which frozen eval nodes to generate for; 'legacy' reproduces the "
                         "uncapped 495-node runs")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    eval_file = SETS[args.sets]

    output = args.output or os.path.join(EXPERIMENTS_DIR, args.mode, "generations.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # --- data -------------------------------------------------------------
    with open(os.path.join(FROZEN_DIR, eval_file)) as f:
        eval_nodes = json.load(f)
    corpus = Corpus()
    mag_id = pd.read_csv(os.path.join(JOINT_DIR, "nodes.csv"),
                         usecols=["node_id", "mag_id"]).set_index("node_id")["mag_id"]
    print(f"{len(eval_nodes)} eval nodes, mode={args.mode}, sets={args.sets} ({eval_file})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:                        # Llama ships without one
        tokenizer.pad_token = tokenizer.eos_token

    # The same builder train.py uses, so both conditions get the byte-identical prompt.
    example = lambda node_id: graph_only_example(tokenizer, corpus, node_id)  # noqa: E731

    # --- models -----------------------------------------------------------
    print(f"Loading {args.model} in {dtype} …")
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    llm.config.use_cache = True            # train.py disables it; decoding needs it back
    llm.eval()

    encoder = adapter = data = None
    gnn_layers = GNN_LAYERS
    if args.mode != "baseline":
        encoder, adapter, gnn_layers = load_adapter(args.checkpoint, device, args.sets, llm)
        # `gate0` is the ablation validate() runs every epoch, carried through to generation:
        # the same weights, the same LoRA, with the graph disconnected from the prefix. Under
        # LoRA it is the honest control, since the frozen-Vicuna arm differs by two things.
        if args.mode == "gate0":
            with torch.no_grad():
                adapter.gate.zero_()
            print("gate forced to zero — the prefix is node-independent from here on")
        data = load_graph()                                # stays on CPU; sampling runs there
        patch_for_prefix(llm, NUM_PREFIX_TOKENS)

    # After any LoRA injection: peft builds its modules in train mode, and lora_dropout would
    # otherwise be live at generation time, making greedy decoding non-deterministic.
    llm.eval()

    # --- resume -----------------------------------------------------------
    done = set()
    write_header = True
    if os.path.exists(output):
        done = set(pd.read_csv(output)["node_id"].tolist())
        write_header = False
        print(f"resuming — {len(done)} rows already written")

    out_file = open(output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=["node_id", "mag_id", "original_abstract",
                                                  "generated_abstract"])
    if write_header:
        writer.writeheader()

    # --- generate ---------------------------------------------------------
    # The prefix occupies real attention positions, so it is charged against the context in
    # both modes — the two CSVs must overflow on the same nodes to stay comparable.
    budget = MAX_LENGTH - NUM_PREFIX_TOKENS - args.max_new_tokens
    overflowed = 0

    # One dict for both branches: the conditions must not drift apart in how they decode, and
    # `generate_with_prefix` forwards **kwargs straight to `generate()`, so they cannot.
    decoding = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                    repetition_penalty=REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id)

    for i, (node_id, prefix) in enumerate(iter_prefixes(eval_nodes, encoder, adapter,
                                                        data, device, gnn_layers), start=1):
        if node_id in done:
            continue

        prompt, original = example(node_id)
        input_ids = torch.tensor([tokenizer(prompt).input_ids], device=device)
        n_tokens = input_ids.size(1)

        print(f"[{i}/{len(eval_nodes)}] node_id={node_id}  refs={corpus.out_degree(node_id)}  "
              f"prompt_tokens={n_tokens}", flush=True)

        if n_tokens > budget:
            overflowed += 1
            generated = f"ERROR: prompt too long ({n_tokens} tokens > {budget} limit)"
        else:
            try:
                if prefix is None:
                    out = llm.generate(input_ids, **decoding)
                    new_tokens = out[0, n_tokens:]
                else:
                    new_tokens = generate_with_prefix(
                        llm, input_ids, prefix.to(dtype), **decoding)
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError as e:
                generated = f"ERROR: CUDA out of memory — {e}"
            except Exception as e:
                generated = f"ERROR: {type(e).__name__}: {e}"
            print(f"  >> {generated[:200]}", flush=True)

        writer.writerow({"node_id": node_id, "mag_id": mag_id.get(node_id, ""),
                         "original_abstract": original, "generated_abstract": generated})
        out_file.flush()

    out_file.close()
    print(f"\nDone. {overflowed} prompts overflowed the {budget}-token budget.")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
