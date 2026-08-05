"""
Stage 5 — generate abstracts for the frozen 495-node evaluation set.

Produces one CSV per condition, over the same nodes, with the same prompts, in the same order:

    --mode baseline   frozen Vicuna-7B, nothing else                 -> generations.csv
    --mode adapter    GraphSAGE + prefix adapter + frozen Vicuna-7B  -> generations.csv

Each mode writes into its own directory under experiments/; pass --output to name the run.

The two runs differ in exactly one thing — whether a graph-conditioned KV prefix is present.
Everything else is held fixed on purpose:

  * **Prompts** come from `text.build_prompt` via `text.Corpus`, the same code path training
    used. The target paper's own title is not in the prompt; the model sees only the titles
    and abstracts of the papers it cites, and infers the topic from the neighbourhood.
  * **Greedy decoding** (`do_sample=False`). Sampling noise would sit on top of the effect
    being measured, and reruns would not reproduce. The older
    `experiments/_archive/vicuna-7b-8bit` baseline
    sampled (temperature 0.35); it is therefore not a drop-in comparison for these two, which
    is why the frozen-LLM condition is regenerated here rather than reused.
  * **bfloat16, not 8-bit.** `src/generation/generate_vicuna.py` loaded the baseline in 8-bit, but
    the adapter was trained against a bf16 model and its prefix is calibrated to those
    numerics. Both conditions therefore run bf16, so quantisation error cannot be mistaken
    for an adapter effect.
  * **Batch size 1.** Prompts range from ~500 to ~5,600 tokens, and padded batches combined
    with a pre-filled prefix cache were never verified. One node at a time removes the
    question entirely.

Context overflow
----------------
63 of the 495 eval nodes (12.7%) have prompts too long to leave room for 256 new tokens inside
Vicuna's 4,096-token context. Following `src/generation/generate_vicuna.py`, those rows are still
written, with `generated_abstract = "ERROR: prompt too long (...)"`; `src/evaluation/evaluate_results.py`
already skips them. The 10 prefix positions are charged against the budget in **both** modes,
even though the baseline has no prefix, so the two CSVs carry identical ERROR rows and the
comparison runs on the same 432 nodes.

Those 63 are not a random sample — they concentrate at high out-degree, precisely where the
graph has the most to contribute. See `docs/adapter.md`, Stage 4c.

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

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (CHECKPOINT_DIR, EXPERIMENTS_DIR, FROZEN_DIR, JOINT_DIR,
                    MAX_LENGTH, NUM_PREFIX_TOKENS)
from graph import load_graph, make_loader
from model import GraphSAGEEncoder, PrefixAdapter
from prefix_generation import generate_with_prefix, patch_for_prefix
from text import Corpus


def load_adapter(path, device):
    """Restore the trained encoder and adapter from a `train.py` checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    encoder = GraphSAGEEncoder().to(device)
    adapter = PrefixAdapter().to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    adapter.load_state_dict(checkpoint["adapter"])
    encoder.eval()
    adapter.eval()
    print(f"loaded {path} (epoch {checkpoint['epoch']}, step {checkpoint['step']})")
    return encoder, adapter


@torch.no_grad()
def iter_prefixes(node_ids, encoder, adapter, data, device):
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

    for batch in make_loader(data, node_ids, batch_size=1, shuffle=False):
        batch = batch.to(device)
        graph_vector = encoder(batch.x, batch.edge_index, batch.batch_size)
        yield int(batch.n_id[0]), adapter(graph_vector)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["baseline", "adapter"], required=True)
    ap.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "epoch2.pt"),
                    help="train.py checkpoint; --mode adapter only")
    ap.add_argument("--model", default="lmsys/vicuna-7b-v1.5")
    ap.add_argument("--output", default=None,
                    help="default: experiments/<mode>/generations.csv; pass a run directory "
                         "explicitly so each run keeps its own folder")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    output = args.output or os.path.join(EXPERIMENTS_DIR, args.mode, "generations.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # --- data -------------------------------------------------------------
    with open(os.path.join(FROZEN_DIR, "eval_nodes.json")) as f:
        eval_nodes = json.load(f)
    corpus = Corpus()
    mag_id = pd.read_csv(os.path.join(JOINT_DIR, "nodes.csv"),
                         usecols=["node_id", "mag_id"]).set_index("node_id")["mag_id"]
    print(f"{len(eval_nodes)} eval nodes, mode={args.mode}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:                        # Llama ships without one
        tokenizer.pad_token = tokenizer.eos_token

    # --- models -----------------------------------------------------------
    print(f"Loading {args.model} in {dtype} …")
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    llm.config.use_cache = True            # train.py disables it; decoding needs it back
    llm.eval()

    encoder = adapter = data = None
    if args.mode == "adapter":
        encoder, adapter = load_adapter(args.checkpoint, device)
        data = load_graph()                                # stays on CPU; sampling runs there
        patch_for_prefix(llm, NUM_PREFIX_TOKENS)

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

    for i, (node_id, prefix) in enumerate(iter_prefixes(eval_nodes, encoder, adapter,
                                                        data, device), start=1):
        if node_id in done:
            continue

        prompt, original = corpus.example(node_id)
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
                    out = llm.generate(input_ids, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, pad_token_id=tokenizer.pad_token_id)
                    new_tokens = out[0, n_tokens:]
                else:
                    new_tokens = generate_with_prefix(
                        llm, input_ids, prefix.to(dtype),
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id)
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
