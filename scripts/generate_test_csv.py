"""
Generate test_results.csv for LLM evaluation.

For each is_test=True node in nodes_subset.csv, the script:
  1. Looks up its neighbouring publications via edges_subset.csv
  2. Builds a prompt listing those neighbours' titles + abstracts
  3. Asks the LLM to generate a new abstract inspired by those sources
  4. Writes node_id, mag_id, original_abstract, generated_abstract to CSV

If the prompt exceeds the model's context window the row is still written
with generated_abstract = "ERROR: <reason>".

Usage:
    python generate_test_csv.py \
        --model_path  /path/to/base/model \
    --adapter_path /path/to/lora/adapter \
        --nodes  ogbn_arxiv_dataset/nodes_subset.csv \
        --edges  ogbn_arxiv_dataset/edges_subset.csv \
        --output ogbn_arxiv_dataset/test_results.csv
"""

import argparse
import csv
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",   required=True,  help="Path to the base model")
parser.add_argument("--adapter_path", default=None,   help="Path to LoRA adapter (optional)")
parser.add_argument("--nodes",  default="ogbn_arxiv_dataset/nodes_subset.csv")
parser.add_argument("--edges",  default="ogbn_arxiv_dataset/edges_subset.csv")
parser.add_argument("--output", default="ogbn_arxiv_dataset/test_results.csv")
parser.add_argument("--max_length", type=int, default=4096,
                    help="Model context window size (prompt + generation must fit within this)")
parser.add_argument("--max_new_tokens", type=int, default=256,
                    help="Maximum new tokens to generate per abstract (decoupled from context window)")
parser.add_argument("--temperature", type=float, default=0.35,
                    help="Sampling temperature; lower values give more focused scientific text")
parser.add_argument("--top_p", type=float, default=0.90,
                    help="Nucleus sampling threshold")
parser.add_argument("--top_k", type=int, default=40,
                    help="Top-k sampling cutoff")
parser.add_argument("--repetition_penalty", type=float, default=1.08,
                    help="Penalty to reduce repetitive phrasing")
parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                    help="Disallow repeating n-grams of this size (0=disabled; avoid >0 with long prompts "
                         "because the n-gram check covers the full input, not just generated tokens)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
# Llama models have no pad token by default — required for batch safety
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model in 8-bit (bfloat16 compute) …")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if args.adapter_path:
    print(f"Applying LoRA adapter from {args.adapter_path} …")
    base_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    base_model = base_model.merge_and_unload()
else:
    print("Running baseline model — no LoRA adapter.")

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    repetition_penalty=args.repetition_penalty,
    no_repeat_ngram_size=args.no_repeat_ngram_size,
    do_sample=True,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
nodes_df = pd.read_csv(args.nodes)
edges_df = pd.read_csv(args.edges)

# Only test nodes are evaluated; all nodes (incl. neighbours) provide text
test_nodes = nodes_df[nodes_df["is_test"] == 1].reset_index(drop=True)
node_lookup = nodes_df.set_index("node_id")  # fast neighbour text lookup

# Pre-build neighbour list per src node
neighbours_of = edges_df.groupby("src_node_id")["dst_node_id"].apply(list).to_dict()

# ---------------------------------------------------------------------------
# Helper: build prompt
# ---------------------------------------------------------------------------
_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

_USER_INSTRUCTION = (
    "You are a scientific writing assistant. "
    "Based on the source publications listed below, write a concise abstract "
    "for a new application paper that builds on and combines these works. "
    "Output ONLY the abstract text, without any heading or explanation."
)


def build_prompt(neighbour_ids: list[int]) -> str:
    sources = []
    for nid in neighbour_ids:
        if nid not in node_lookup.index:
            continue
        row = node_lookup.loc[nid]
        title    = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        sources.append(f"- Title: {title}\n  Abstract: {abstract}")

    sources_text = "\n\n".join(sources) if sources else "(no neighbouring sources available)"
    user_message = (
        f"{_USER_INSTRUCTION}\n\n"
        f"Source publications:\n{sources_text}\n\n"
        f"Generated abstract:"
    )
    # Vicuna-7b-v1.5 conversation format — prevents early EOS from the chat model
    return f"{_VICUNA_SYSTEM} USER: {user_message} ASSISTANT:"


def fits_in_context(prompt: str) -> tuple[bool, int]:
    """Return (fits, token_count). True when prompt + max_new_tokens fits in the context window."""
    token_count = len(tokenizer.encode(prompt))
    fits = token_count + args.max_new_tokens <= args.max_length
    return fits, token_count


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
# Resume from existing output (skip already-processed node_ids)
processed_ids: set[int] = set()
write_header = True
if os.path.exists(args.output):
    existing = pd.read_csv(args.output)
    processed_ids = set(existing["node_id"].tolist())
    write_header = False
    print(f"Resuming — {len(processed_ids)} rows already written.")

out_file = open(args.output, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(
    out_file,
    fieldnames=["node_id", "mag_id", "original_abstract", "generated_abstract"],
)
if write_header:
    writer.writeheader()

total = len(test_nodes)
for idx, row in test_nodes.iterrows():
    node_id = int(row["node_id"])
    if node_id in processed_ids:
        continue

    mag_id            = row["mag_id"]
    original_abstract = str(row.get("abstract", "")).strip()
    neighbour_ids     = neighbours_of.get(node_id, [])

    prompt = build_prompt(neighbour_ids)
    fits, n_tokens = fits_in_context(prompt)

    print(f"[{idx+1}/{total}] node_id={node_id}  neighbours={len(neighbour_ids)}  "
          f"prompt_tokens={n_tokens}  fits={fits}", flush=True)

    if not fits:
        generated = f"ERROR: prompt too long ({n_tokens} tokens > {args.max_length - args.max_new_tokens} limit)"
    else:
        try:
            output = pipe(
                prompt,
                max_new_tokens=args.max_new_tokens,
                return_full_text=False,
            )[0]["generated_text"]
            generated = output.strip()
            print(f"  >> {generated}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            generated = f"ERROR: CUDA out of memory — {e}"
        except Exception as e:
            generated = f"ERROR: {type(e).__name__}: {e}"

    writer.writerow({
        "node_id":           node_id,
        "mag_id":            mag_id,
        "original_abstract": original_abstract,
        "generated_abstract": generated,
    })
    out_file.flush()

out_file.close()
print(f"\nDone. Results written to {args.output}")
