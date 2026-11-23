"""
Stage 5 — generate abstracts for a frozen evaluation set.

Produces one CSV per condition, over the same nodes, with the same prompts, in the same order:

    --mode baseline   frozen Qwen3-14B, nothing else                 -> generations.csv
    --mode adapter    GraphSAGE + prefix adapter + frozen Qwen3-14B  -> generations.csv
    --mode text       books only: frozen Qwen3-14B reading the co-purchased books as text,
                      no prefix (text.book_text_example)

Each mode writes into its own directory under experiments/; pass --output to name the run.

Since the soft-prompt-only change (2026-09-16) the prompt carries no references, so
`--mode baseline` receives no information about the paper at all: under greedy decoding it
writes the same abstract for every node. It is a floor, not a control. The comparison arm for
the graph-only adapter is the 2026-09-10 text-path Qwen baseline.

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
  * **bfloat16, not 8-bit.** The earlier Vicuna baseline was loaded in 8-bit, but
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

Run 9 — what the adapter is fed
-------------------------------
A run-9 checkpoint is a decoder trained on embedding-space vectors (train.py --oracle), so one
checkpoint is evaluated several ways by changing only the vector, with `--input`:

    oracle     the paper's own embedding — a ceiling, it sees the answer
    refs-mean  the plain mean of its references' embeddings — the graph arm, no graph training
    stage-a    head(GNN(neighbourhood)) from pretrain_encoder.py — does a GNN beat averaging?
               (run 11's phase-2 GNN, fed to its phase-1 reader, is this arm)
    zero       the zero vector — the floor; generated once and copied to every row

    python generate.py --mode adapter --sets run9 --input refs-mean \
        --checkpoint ../../checkpoints/qwen14b-run9-decoder/best.pt --output ...

Run 11
------
A joint fine-tune checkpoint (train.py --reader) carries its GNN + head and loads with the
default `--input encoder`. `--no-repeat-ngram 6` stops sentence loops at decoding time, and
`--subset N` generates for a fixed random N of the eval papers — the loop-fix test.
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
    # Run 9 trains on its own node set but is scored on the same 945 papers as every run.
    "run9": "eval_nodes_capped.json",
    "children": "eval_nodes.json",   # DYTAG_DATASET=children
}

# Inputs that live in the embedding space an --oracle checkpoint was trained on. Any of them can
# be fed to such a checkpoint; none of them to a checkpoint trained behind the graph encoder.
EMBEDDING_INPUTS = ("oracle", "refs-mean", "stage-a", "zero")


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


def load_adapter(path, device, sets, llm, oracle, embeddings):
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
    # An oracle checkpoint has exactly the shapes of a graph one. Generating an oracle-trained
    # prefix from graph vectors (or the reverse) is fluent and wrong, so it is refused here.
    was_oracle = checkpoint.get("oracle", False)
    assert was_oracle == oracle, (
        f"{path} was trained {'with' if was_oracle else 'without'} --oracle but generation "
        f"has --oracle {'set' if oracle else 'not set'}; the adapter would be fed a kind of "
        f"vector it was never trained on")
    # Whitened and raw node features have identical shapes (run 8), so a
    # mismatch loads cleanly and runs a trained encoder on a distribution it has never seen.
    was_features = checkpoint.get("node_embeddings", "node_embeddings.npy")
    assert was_features == os.path.basename(embeddings), (
        f"{path} was trained on {was_features} but --features loads "
        f"{os.path.basename(embeddings)}; the adapter would be fed a feature "
        f"distribution it was never trained on")
    gnn_layers = checkpoint.get("gnn_layers", 2)   # predating --gnn-layers means 2-layer
    gnn_conv = checkpoint.get("gnn_conv", "sage")  # everything before stage 9 was GraphSAGE
    # Run 11's joint checkpoint stores GNN + head (model.EmbeddingEstimator) as its encoder.
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

    # Read off the checkpoint, never off config.py: the neighbour-token width is part of the
    # sequence the prefix was trained beside, so a run generated at a different k would be
    # fluent and wrong. `strict` loading also catches a checkpoint whose ntoken is absent.
    ntoken = None
    n_tokens = checkpoint.get("neighbour_tokens", 0)
    if n_tokens:
        ntoken = NeighbourTokenAdapter(max_tokens=n_tokens).to(device)
        ntoken.load_state_dict(checkpoint["ntoken"])
        ntoken.eval()

    lora = checkpoint.get("lora")     # None for every checkpoint trained without --lora
    if lora is not None:
        from lora import add_lora, load_lora_state, targets_of
        # Read off the checkpoint: run 9 moved config.LORA_TARGETS to q/v, and steps 6-8 used
        # q/k/v. Taking them from config.py would refuse those checkpoints.
        add_lora(llm, targets_of(lora))
        load_lora_state(llm, lora)

    print(f"loaded {path} (epoch {checkpoint['epoch']}, step {checkpoint['step']}, "
          f"{gnn_layers}-layer {gnn_conv} encoder, "
          f"{n_tokens or 'no'} neighbour tokens, "
          f"{'LoRA restored' if lora is not None else 'frozen LLM'})")
    return encoder, adapter, ntoken, gnn_layers


def load_stage_a(path, device, embeddings):
    """-> (vector function, encoder depth) for run 9's `--input stage-a` arm.

    A pretrain_encoder.py checkpoint: the GNN reads the masked neighbourhood and a linear head
    maps its seed output into the (whitened) embedding space, trained by InfoNCE + VICReg
    against the seed's own embedding. The HEAD IS LOADED. Run 8 discarded it and fed the
    adapter the raw GNN output, which was 0.85 collinear across papers; the head output is the
    tensor Stage A actually decorrelated, and the one that lives in the space the run-9
    decoder was trained on. Normalised, because every other vector that decoder sees is.

    Stage A trained on whatever config.NODE_EMBEDDINGS was at the time; run 8's encoders
    (encoder-*-vicreg.pt) were trained on the whitened features, as run 9 is, and record
    nothing. Run 11's record which, and a mismatch with `embeddings` is refused: the head would
    map a feature distribution it never saw into the decoder's space.
    """
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
    """Yield `(node_id, prefix, neighbour_tokens)` for each node, in the order given.

    Both tensors are None in baseline mode; `neighbour_tokens` is also None for a checkpoint
    trained without them. The two come out of ONE encoder pass, the same way training builds
    them, so the pooled prefix and the neighbour tokens can never describe different
    neighbourhoods.

    In adapter mode the order comes from `NeighborLoader(shuffle=False)`, and the node_id is
    read back off the batch rather than assumed, so a row can never be paired with another
    node's neighbourhood.

    `vector` (run 9) replaces the encoder: a function of the batch returning the
    embedding-space vector to feed the adapter (refs-mean, stage-a, zero).
    """
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
    # clean=False: the evaluation golds stay byte-identical to all eight earlier runs, so the
    # scores remain comparable. Training cleans its targets (text.clean_abstract) because the
    # prompt instructs "no markup, mathematical notation or headings" while 13.5% of the
    # targets carried exactly that — but cleaning what the generations are SCORED against
    # would silently move every number in experiments/.
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
        # `gate0` is the ablation validate() runs every epoch, carried through to generation:
        # the same weights, the same LoRA, with the graph disconnected from the prefix. Under
        # LoRA it is the honest control, since the frozen-Vicuna arm differs by two things.
        if args.mode == "gate0":
            with torch.no_grad():
                adapter.gate.zero_()
                # Both channels, or this stops being the "no graph" arm: with the neighbour
                # tokens still open, zeroing only the prefix would leave the graph's larger
                # channel running and the control would measure nothing it claims to.
                if ntoken is not None:
                    ntoken.gate.zero_()
            print("both gates forced to zero — prefix and neighbour tokens are "
                  "node-independent from here on")
        if args.mode == "ngate0":
            # Not a no-op on a pooled-only checkpoint — it would be the *adapter* arm under
            # another name, and land in a directory labelled as an ablation.
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
        # The neighbour tokens are prefilled into the cache before generate() sees it, so the
        # cache runs P..P+k long and k varies with the node's out-degree.
        patch_for_prefix(llm, NUM_PREFIX_TOKENS,
                         max_extra=ntoken.max_tokens if ntoken is not None else 0)

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
    writer = csv.DictWriter(out_file, fieldnames=["node_id", SOURCE_ID, "original_abstract",
                                                  "generated_abstract"])
    if write_header:
        writer.writeheader()

    # --- generate ---------------------------------------------------------
    # The prefix occupies real attention positions, so it is charged against the context in
    # both modes — the two CSVs must overflow on the same nodes to stay comparable.
    # The neighbour tokens occupy real attention positions too, so the whole graph side is
    # charged against the context: P + k + prompt + generation must fit MAX_LENGTH.
    budget = (MAX_LENGTH - NUM_PREFIX_TOKENS - args.max_new_tokens
              - (ntoken.max_tokens if ntoken is not None else 0))
    overflowed = 0
    constant, floor_texts = None, []

    # One dict for both branches: the conditions must not drift apart in how they decode, and
    # `generate_with_prefix` forwards **kwargs straight to `generate()`, so they cannot.
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
            # writes one string for every node. It was generated twice and checked equal
            # (below); the remaining ~7 GPU-hours would only reproduce it.
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
