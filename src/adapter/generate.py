"""Generate abstracts for the evaluation papers with a trained checkpoint.

`--input` selects the vector the reader receives: `stage-a` (the GNN estimate, from
`--stage-a-encoder`), `refs-mean` (the mean of the reference embeddings), `oracle` (the paper's
own embedding) or `zero`. Without `--input`, a joint checkpoint uses its own GNN. Reruns are
resumable: only node_ids missing from the output CSV are generated.

The commands used for the thesis are in hpc/komondor/jobs/run11_all.sh.
"""

import argparse
import csv
import json
import os
import re

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (BASE_MODEL, CHECKPOINT_DIR, DATASET, DEFAULT_SETS, EXPERIMENTS_DIR,
                    FEATURES, FROZEN_DIR, GNN_CONV, GNN_LAYERS, JOINT_DIR, MAX_LENGTH, MAX_NEW_TOKENS,
                    NO_REPEAT_NGRAM, NUM_PREFIX_TOKENS, OUT_CHANNELS, REPETITION_PENALTY, SEED,
                    SOURCE_ID, check_sets)
from graph import load_graph, make_loader, select_neighbour_rows
from model import (EmbeddingEstimator, GraphEncoder, NeighbourTokenAdapter, PrefixAdapter,
                   oracle_vector, refs_mean_vector)
from prefix_generation import generate_with_prefix, patch_for_prefix
from text import PROMPT, Corpus, book_text_example, graph_only_example

# Mirrors train.py's SETS: which frozen eval nodes.
SETS = {
    "capped": "eval_nodes_capped.json",
    "legacy": "eval_nodes.json",
    # run9 trains on its own node set and is evaluated on the same 945 papers.
    "run9": "eval_nodes_capped.json",
    "children": "eval_nodes.json",   # DYTAG_DATASET=children
}

# Inputs in the embedding space an --oracle checkpoint is trained on.
EMBEDDING_INPUTS = ("oracle", "refs-mean", "stage-a", "zero")


def remap_encoder(state):
    """Rename `convN.` keys of older checkpoints to `convs.N-1.`."""
    if not any(re.match(r"conv\d+\.", key) for key in state):
        return state
    print("checkpoint predates the ModuleList encoder — remapping convN. -> convs.N-1.")
    return {re.sub(r"^conv(\d+)\.", lambda m: f"convs.{int(m.group(1)) - 1}.", key): value
            for key, value in state.items()}


def load_adapter(path, device, sets, llm, oracle, embeddings):
    """Restore the trained encoder and adapter from a `train.py` checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    was = checkpoint.get("sets", "legacy")
    assert was == sets, (
        f"{path} was trained on the '{was}' node sets but --sets is '{sets}'; "
        f"the nodes differ, so this pairing would not measure the trained adapter")
    was_prompt = checkpoint.get("prompt", "text")
    assert was_prompt == PROMPT, (
        f"{path} was trained with the '{was_prompt}' prompt but generation uses '{PROMPT}'")
    was_oracle = checkpoint.get("oracle", False)
    assert was_oracle == oracle, (
        f"{path} was trained {'with' if was_oracle else 'without'} --oracle but generation "
        f"has --oracle {'set' if oracle else 'not set'}; the adapter would be fed a kind of "
        f"vector it was never trained on")
    was_features = checkpoint.get("node_embeddings", "node_embeddings.npy")
    assert was_features == os.path.basename(embeddings), (
        f"{path} was trained on {was_features} but --features loads "
        f"{os.path.basename(embeddings)}; the adapter would be fed a feature "
        f"distribution it was never trained on")
    gnn_layers = checkpoint.get("gnn_layers", 2)
    gnn_conv = checkpoint.get("gnn_conv", "sage")
    # A joint checkpoint stores GNN + head (model.EmbeddingEstimator) as its encoder.
    if checkpoint.get("joint", False):
        encoder = EmbeddingEstimator(num_layers=gnn_layers, conv=gnn_conv).to(device)
        encoder.load_state_dict(checkpoint["encoder"])
    else:
        encoder = GraphEncoder(num_layers=gnn_layers, conv=gnn_conv).to(device)
        encoder.load_state_dict(remap_encoder(checkpoint["encoder"]))
    adapter = PrefixAdapter().to(device)
    adapter.load_state_dict(checkpoint["adapter"])
    encoder.eval()
    adapter.eval()

    ntoken = None
    n_tokens = checkpoint.get("neighbour_tokens", 0)
    if n_tokens:
        ntoken = NeighbourTokenAdapter(max_tokens=n_tokens).to(device)
        ntoken.load_state_dict(checkpoint["ntoken"])
        ntoken.eval()

    lora = checkpoint.get("lora")     # None for every checkpoint trained without --lora
    if lora is not None:
        from lora import add_lora, load_lora_state, targets_of
        # The LoRA targets are read from the checkpoint, not from config.py.
        add_lora(llm, targets_of(lora))
        load_lora_state(llm, lora)

    print(f"loaded {path} (epoch {checkpoint['epoch']}, step {checkpoint['step']}, "
          f"{gnn_layers}-layer {gnn_conv} encoder, "
          f"{n_tokens or 'no'} neighbour tokens, "
          f"{'LoRA restored' if lora is not None else 'frozen LLM'})")
    return encoder, adapter, ntoken, gnn_layers


def load_stage_a(path, device, embeddings):
    """-> (vector function, encoder depth) for the `--input stage-a` arm (the GNN estimate)."""
    pre = torch.load(path, map_location=device)
    was = pre.get("node_embeddings", "node_embeddings_whitened.npy")
    assert was == os.path.basename(embeddings), (
        f"{path} was trained on {was} but --features loads {os.path.basename(embeddings)}")
    encoder = GraphEncoder(num_layers=pre["gnn_layers"], conv=pre["gnn_conv"]).to(device)
    encoder.load_state_dict(pre["encoder"])
    head = torch.nn.Linear(OUT_CHANNELS, OUT_CHANNELS).to(device)
    head.load_state_dict(pre["head"])
    encoder.eval()
    head.eval()
    print(f"stage-a: {pre['gnn_layers']}-layer {pre['gnn_conv']} encoder + head from {path} "
          f"(stage A epoch {pre.get('epoch')}, output cosine {pre.get('output_cosine')})")

    def vector(batch):
        h = encoder(batch.x, batch.edge_index, batch.batch_size)[:batch.batch_size]
        return F.normalize(head(h), dim=-1)
    return vector, pre["gnn_layers"]


@torch.no_grad()
def iter_prefixes(node_ids, encoder, adapter, ntoken, data, device, gnn_layers, oracle=False,
                  vector=None):
    """Yield `(node_id, prefix, neighbour_tokens)` for each node, in the order given."""
    if encoder is None:
        for node_id in node_ids:
            yield node_id, None, None
        return

    for batch in make_loader(data, node_ids, batch_size=1, shuffle=False,
                             gnn_layers=gnn_layers):
        batch = batch.to(device)
        tokens = None
        if vector is not None:
            graph_vector = vector(batch)
        elif oracle:
            graph_vector = oracle_vector(batch)
        else:
            h = encoder(batch.x, batch.edge_index, batch.batch_size)
            graph_vector = h[:batch.batch_size]
            if ntoken is not None:
                rows = select_neighbour_rows(batch, ntoken.max_tokens)
                tokens = ntoken(h[rows]).unsqueeze(0)
        yield int(batch.n_id[0]), adapter(graph_vector), tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["baseline", "adapter", "gate0", "ngate0", "text"],
                    required=True,
                    help="gate0 disconnects the graph from BOTH channels; ngate0 keeps the "
                         "k neighbour slots and empties them, which separates 'k tokens "
                         "help' from 'their content helps'")
    ap.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "epoch2.pt"),
                    help="train.py checkpoint; --mode adapter only")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--output", default=None,
                    help="default: experiments/<mode>/generations.csv; pass a run directory "
                         "explicitly so each run keeps its own folder")
    ap.add_argument("--sets", choices=sorted(SETS), default=DEFAULT_SETS,
                    help="which frozen eval nodes to generate for; 'legacy' reproduces the "
                         "uncapped 495-node runs")
    ap.add_argument("--oracle", action="store_true",
                    help="CEILING EXPERIMENT, not a result: feed the adapter the target "
                         "abstract's own embedding instead of the encoder output "
                         "(model.oracle_vector). Must match how the checkpoint was trained, "
                         "which is asserted. The arm sees the answer it is asked to write, "
                         "so its score belongs beside no other arm without that label")
    ap.add_argument("--input", choices=("encoder",) + EMBEDDING_INPUTS, default=None,
                    help="run 9: which vector the adapter is fed (see the module docstring). "
                         "Defaults to 'oracle' with --oracle and 'encoder' otherwise, which "
                         "is every run before run 9")
    ap.add_argument("--stage-a-encoder", default=None,
                    help="--input stage-a: a pretrain_encoder.py checkpoint. Its head IS "
                         "loaded — the head output is the vector Stage A trained into "
                         "embedding space, and run 8 failed by discarding it")
    ap.add_argument("--features", choices=sorted(FEATURES), default="whitened",
                    help="node features; must match the checkpoint, which is asserted. "
                         "'raw' for run 10a and every run up to step 7")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--no-repeat-ngram", type=int, default=NO_REPEAT_NGRAM,
                    help="ban any repeated n-gram of this length while decoding; 0 is off, as "
                         "in every run before run 11 (config.NO_REPEAT_NGRAM)")
    ap.add_argument("--subset", type=int, default=None,
                    help="generate for a fixed random N of the eval papers (seeded), not all")
    args = ap.parse_args()
    args.input = args.input or ("oracle" if args.oracle else "encoder")
    embedding_input = args.input in EMBEDDING_INPUTS
    assert args.input != "stage-a" or args.stage_a_encoder, "--input stage-a needs --stage-a-encoder"
    assert args.mode == "adapter" or args.input in ("encoder", "oracle"), (
        "--input selects the vector for --mode adapter; the gate ablations keep their own")

    check_sets(args.sets)
    eval_file = SETS[args.sets]

    output = args.output or os.path.join(EXPERIMENTS_DIR, args.mode, "generations.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # --- data -------------------------------------------------------------
    with open(os.path.join(FROZEN_DIR, eval_file)) as f:
        eval_nodes = json.load(f)
    if args.subset:
        order = torch.randperm(len(eval_nodes), generator=torch.Generator().manual_seed(SEED))
        eval_nodes = [eval_nodes[i] for i in sorted(order[:args.subset].tolist())]
    corpus = Corpus(clean=False)
    source_id = pd.read_csv(os.path.join(JOINT_DIR, "nodes.csv"),
                            usecols=["node_id", SOURCE_ID]).set_index("node_id")[SOURCE_ID]
    print(f"{len(eval_nodes)} eval nodes, mode={args.mode}, sets={args.sets} ({eval_file})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:                        # Llama ships without one
        tokenizer.pad_token = tokenizer.eos_token

    # The same builder train.py uses, so both conditions get the byte-identical prompt.
    example = lambda node_id: graph_only_example(tokenizer, corpus, node_id)  # noqa: E731
    if args.mode == "text":
        assert DATASET == "children", "--mode text is the books' text arm: DYTAG_DATASET=children"
        example = lambda node_id: book_text_example(tokenizer, corpus, node_id)  # noqa: E731

    # --- models -----------------------------------------------------------
    print(f"Loading {args.model} in {dtype} …")
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    llm.config.use_cache = True            # train.py disables it; decoding needs it back
    llm.eval()

    encoder = adapter = ntoken = data = vector = None
    gnn_layers = GNN_LAYERS
    if args.mode not in ("baseline", "text"):
        encoder, adapter, ntoken, gnn_layers = load_adapter(args.checkpoint, device,
                                                            args.sets, llm, embedding_input,
                                                            FEATURES[args.features])
        if args.input == "oracle":
            print("=" * 78, "\nORACLE CEILING RUN — the adapter is fed the target abstract's "
                  "own embedding.\nThis arm sees the answer. Its score is a ceiling on the "
                  "prefix channel,\nnever a result, and is not comparable to any arm that "
                  "does not see the target.\n" + "=" * 78, flush=True)
        if args.mode == "gate0":
            with torch.no_grad():
                adapter.gate.zero_()
                if ntoken is not None:
                    ntoken.gate.zero_()
            print("both gates forced to zero — prefix and neighbour tokens are "
                  "node-independent from here on")
        if args.mode == "ngate0":
            assert ntoken is not None, (
                f"--mode ngate0 needs a checkpoint trained with neighbour tokens; "
                f"{args.checkpoint} has none")
            with torch.no_grad():
                ntoken.gate.zero_()
            print("neighbour gate forced to zero — the k slots stay, their content goes")
        data = load_graph(FEATURES[args.features])         # stays on CPU; sampling runs there
        if args.input == "refs-mean":
            vector = refs_mean_vector
        elif args.input == "zero":
            vector = lambda batch: torch.zeros(batch.batch_size, batch.x.size(1),  # noqa: E731
                                               device=batch.x.device)
        elif args.input == "stage-a":
            vector, gnn_layers = load_stage_a(args.stage_a_encoder, device,
                                              FEATURES[args.features])
        patch_for_prefix(llm, NUM_PREFIX_TOKENS,
                         max_extra=ntoken.max_tokens if ntoken is not None else 0)

    # peft builds its modules in train mode; eval() turns LoRA dropout off for generation.
    llm.eval()

    # --- resume -----------------------------------------------------------
    done = set()
    write_header = True
    if os.path.exists(output):
        done = set(pd.read_csv(output)["node_id"].tolist())
        write_header = False
        print(f"resuming — {len(done)} rows already written")

    out_file = open(output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=["node_id", SOURCE_ID, "original_abstract",
                                                  "generated_abstract"])
    if write_header:
        writer.writeheader()

    # --- generate ---------------------------------------------------------
    budget = (MAX_LENGTH - NUM_PREFIX_TOKENS - args.max_new_tokens
              - (ntoken.max_tokens if ntoken is not None else 0))
    overflowed = 0
    constant, floor_texts = None, []

    decoding = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                    repetition_penalty=REPETITION_PENALTY,
                    no_repeat_ngram_size=args.no_repeat_ngram,
                    pad_token_id=tokenizer.pad_token_id)

    for i, (node_id, prefix, tokens) in enumerate(
            iter_prefixes(eval_nodes, encoder, adapter, ntoken, data, device, gnn_layers,
                          args.input == "oracle", vector),
            start=1):
        if node_id in done:
            continue

        prompt, original = example(node_id)
        input_ids = torch.tensor([tokenizer(prompt).input_ids], device=device)
        n_tokens = input_ids.size(1)

        print(f"[{i}/{len(eval_nodes)}] node_id={node_id}  refs={corpus.out_degree(node_id)}  "
              f"prompt_tokens={n_tokens}  "
              f"neighbour_tokens={0 if tokens is None else tokens.size(1)}", flush=True)

        if constant is not None:
            # --input zero: prompt and prefix are both node-independent, so greedy decoding
            generated = constant
        elif n_tokens > budget:
            overflowed += 1
            generated = f"ERROR: prompt too long ({n_tokens} tokens > {budget} limit)"
        else:
            try:
                if prefix is None:
                    out = llm.generate(input_ids, **decoding)
                    new_tokens = out[0, n_tokens:]
                else:
                    new_tokens = generate_with_prefix(
                        llm, input_ids, prefix.to(dtype),
                        neighbour_tokens=None if tokens is None else tokens.to(dtype),
                        **decoding)
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError as e:
                generated = f"ERROR: CUDA out of memory — {e}"
            except Exception as e:
                generated = f"ERROR: {type(e).__name__}: {e}"
            print(f"  >> {generated[:200]}", flush=True)
            if args.input == "zero":
                floor_texts.append(generated)
                if len(floor_texts) == 2:
                    assert floor_texts[0] == floor_texts[1], (
                        "--input zero wrote two different texts for two nodes: something "
                        "node-dependent reaches the decoder, so this is not the floor")
                    constant = generated
                    print("floor is constant — copying it to the remaining rows", flush=True)

        writer.writerow({"node_id": node_id, SOURCE_ID: source_id.get(node_id, ""),
                         "original_abstract": original, "generated_abstract": generated})
        out_file.flush()

    out_file.close()
    print(f"\nDone. {overflowed} prompts overflowed the {budget}-token budget.")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
