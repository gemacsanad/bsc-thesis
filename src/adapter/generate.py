"""
Stage 5 — generate abstracts for a frozen evaluation set.

Produces one CSV per condition, over the same nodes, with the same prompts, in the same order:

    --mode baseline   frozen Vicuna-7B, nothing else                 -> generations.csv
    --mode adapter    GraphSAGE + prefix adapter + frozen Vicuna-7B  -> generations.csv

Each mode writes into its own directory under experiments/; pass --output to name the run.

The two runs differ in exactly one thing — whether a graph-conditioned KV prefix is present.
Everything else is held fixed on purpose:

  * **Prompts** come from `text.build_prompt` via `text.Corpus`, the same code path training
    used. The target paper's own title is not in the prompt; the model sees only the titles
    and abstracts of the papers it cites, and infers the topic from the neighbourhood.
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
  * **Batch size 1.** Prompts range from ~500 to ~5,600 tokens, and padded batches combined
    with a pre-filled prefix cache were never verified. One node at a time removes the
    question entirely.

Node sets
---------
`--sets capped` (default) generates for the 945 nodes of `eval_nodes_capped.json`, with
prompts trimmed to the centroid-closest references by `text.capped_example` — the same
builder training used, so the adapter meets the prompt shape it was trained on.

`--sets legacy` reproduces the older 495-node runs from `eval_nodes.json` with untrimmed
prompts. The two are not comparable: only 2 nodes overlap, and a checkpoint records which
sets it was trained on so a mismatched pairing is refused rather than silently measured.

Context overflow
----------------
Under `--sets capped` overflow cannot happen: the freeze selected each node with its prompt
already trimmed to fit, so a prompt over budget means the prompt path has drifted and the run
aborts. `--max-new-tokens` above the reserve the sets were frozen under is refused for the
same reason.

Under `--sets legacy` the old behaviour stands: 63 of the 495 nodes (12.7%) overflow, are
written as `generated_abstract = "ERROR: prompt too long (...)"`, and are skipped by
`src/evaluation/evaluate_results.py`. The 10 prefix positions are charged against the budget in
**both** modes, even though the baseline has no prefix, so the two CSVs carry identical ERROR
rows and the comparison runs on the same 432 nodes. Those 63 are not a random sample — they
concentrate at high out-degree, precisely where the graph has the most to contribute. See
`docs/adapter.md`, Stage 4c.

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

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (BASE_MODEL, CHECKPOINT_DIR, DEFAULT_SETS, EXPERIMENTS_DIR,
                    FROZEN_DIR, GENERATION_RESERVE, GNN_LAYERS, JOINT_DIR,
                    MAX_LENGTH, MAX_NEW_TOKENS, NUM_PREFIX_TOKENS, REPETITION_PENALTY)
from graph import load_graph, make_loader
from model import GraphSAGEEncoder, PrefixAdapter
from prefix_generation import generate_with_prefix, patch_for_prefix
from text import Corpus, capped_example

# Mirrors train.py's SETS: which frozen eval nodes, and whether their prompts are capped.
SETS = {
    "capped": ("eval_nodes_capped.json", True),
    "legacy": ("eval_nodes.json", False),
}


def load_adapter(path, device, sets):
    """Restore the trained encoder and adapter from a `train.py` checkpoint.

    Also returns the encoder depth, which is read off the checkpoint rather than passed in:
    the same number has to build the encoder and set the loader's hop count, and there is no
    reason to let the command line disagree with the weights being loaded.
    """
    checkpoint = torch.load(path, map_location=device)
    # A capped-trained prefix evaluated on legacy prompts (or the reverse) produces fluent
    # text from a neighbourhood the adapter was never calibrated on — silent, not an error.
    was = checkpoint.get("sets", "legacy")   # checkpoints predating --sets are all legacy
    assert was == sets, (
        f"{path} was trained on the '{was}' node sets but --sets is '{sets}'; "
        f"the prompts differ, so this pairing would not measure the trained adapter")
    gnn_layers = checkpoint.get("gnn_layers", 2)   # predating --gnn-layers means 2-layer
    encoder = GraphSAGEEncoder(num_layers=gnn_layers).to(device)
    adapter = PrefixAdapter().to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    adapter.load_state_dict(checkpoint["adapter"])
    encoder.eval()
    adapter.eval()
    print(f"loaded {path} (epoch {checkpoint['epoch']}, step {checkpoint['step']}, "
          f"{gnn_layers}-layer encoder)")
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
    ap.add_argument("--mode", choices=["baseline", "adapter"], required=True)
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

    eval_file, capped = SETS[args.sets]
    # The capped sets were frozen holding GENERATION_RESERVE tokens back for the abstract.
    # Asking for more at generation time would need references the freeze already dropped,
    # so the guarantee that nothing overflows would quietly stop holding.
    if capped and args.max_new_tokens > GENERATION_RESERVE:
        raise SystemExit(
            f"--max-new-tokens {args.max_new_tokens} exceeds the {GENERATION_RESERVE}-token "
            f"reserve the capped sets were frozen under; use --sets legacy or lower it")

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

    # Both conditions must build byte-identical prompts — that is the whole comparison — so
    # the embeddings are loaded in baseline mode too, where no graph is otherwise needed.
    # Memory-mapped: only the sampled rows are ever touched.
    if capped:
        embeddings = np.load(os.path.join(JOINT_DIR, "node_embeddings.npy"), mmap_mode="r")
        def example(node_id):
            prompt, target, _ = capped_example(tokenizer, corpus, embeddings, node_id)
            return prompt, target
    else:
        example = corpus.example

    # --- models -----------------------------------------------------------
    print(f"Loading {args.model} in {dtype} …")
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    llm.config.use_cache = True            # train.py disables it; decoding needs it back
    llm.eval()

    encoder = adapter = data = None
    gnn_layers = GNN_LAYERS
    if args.mode == "adapter":
        encoder, adapter, gnn_layers = load_adapter(args.checkpoint, device, args.sets)
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

        # Capping makes overflow structurally impossible. If it happens anyway the prompt
        # path has drifted from the freeze, and every row already written is suspect — so
        # this stops the run rather than writing another plausible-looking ERROR row.
        assert not (capped and n_tokens > budget), (
            f"node {node_id}: capped prompt is {n_tokens} tokens, over the {budget} budget; "
            f"the prompt path no longer matches the frozen sets")

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
