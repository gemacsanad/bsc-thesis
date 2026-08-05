"""
Generate test_results.csv for LLM evaluation using OpenAI API.

For each is_test=True node in nodes_subset.csv, the script:
  1. Looks up its neighbouring publications via edges_subset.csv
  2. Builds a prompt listing those neighbours' titles + abstracts
  3. Asks gpt-5.4-mini to generate a new abstract inspired by those sources
  4. Writes node_id, mag_id, original_abstract, generated_abstract to CSV

If the API call fails the row is still written
with generated_abstract = "ERROR: <reason>".

Usage:
    python generate_test_csv_gpt.py \
        --nodes  data/processed/ogbn_subset/nodes_subset.csv \
        --edges  data/processed/ogbn_subset/edges_subset.csv \
        --output experiments/<run>/generations.csv
"""

import argparse
import csv
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--nodes",  default="data/processed/ogbn_subset/nodes_subset.csv")
parser.add_argument("--edges",  default="data/processed/ogbn_subset/edges_subset.csv")
parser.add_argument("--output", default="experiments/gpt/generations.csv")
parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model to use")
parser.add_argument("--max_new_tokens", type=int, default=256,
                    help="Maximum new tokens to generate per abstract")
parser.add_argument("--temperature", type=float, default=0.35,
                    help="Sampling temperature; lower values give more focused scientific text")
parser.add_argument("--top_p", type=float, default=0.90,
                    help="Nucleus sampling threshold")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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


def build_prompt(neighbour_ids: list[int]) -> tuple[str, str]:
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
    return _VICUNA_SYSTEM, user_message


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

    system_msg, user_msg = build_prompt(neighbour_ids)

    print(f"[{idx+1}/{total}] node_id={node_id}  neighbours={len(neighbour_ids)}", flush=True)

    generated = None
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                max_completion_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            generated = response.choices[0].message.content.strip()
            print(f"  >> {generated}", flush=True)
            break
        except RateLimitError:
            wait = 2 ** attempt
            print(f"  Rate limit — retrying in {wait}s …", flush=True)
            time.sleep(wait)
        except Exception as e:
            generated = f"ERROR: {type(e).__name__}: {e}"
            break
    if generated is None:
        generated = "ERROR: max retries exceeded"

    writer.writerow({
        "node_id":           node_id,
        "mag_id":            mag_id,
        "original_abstract": original_abstract,
        "generated_abstract": generated,
    })
    out_file.flush()

out_file.close()
print(f"\nDone. Results written to {args.output}")
