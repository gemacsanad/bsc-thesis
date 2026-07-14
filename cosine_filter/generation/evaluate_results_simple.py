"""
evaluate_results_simple.py

Simplified sibling of scripts/evaluate_results.py: evaluates generated abstracts
against originals but only computes the raw scores — no LLM reasoning text and
no reasoning categorization. Runs over all 6 test_results.csv files found in
this folder's subdirectories and writes each evaluation next to its input.

Metrics added per row:
  rouge1_f, rouge2_f, rougeL_f            — ROUGE F1 scores
  bertscore_p, bertscore_r, bertscore_f1  — BERTScore
  llm_accuracy, llm_coherence, llm_language (1-10) — LLM-as-a-judge scores

Requires OPENAI_API_KEY in .env (or environment).

Usage:
    python evaluate_results_simple.py
    python evaluate_results_simple.py --input_name test_results.csv --output_name evaluated_results.csv

Install dependencies:
    pip install rouge-score bert-score openai python-dotenv pandas torch
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rouge_score import rouge_scorer as rs_lib
from bert_score import score as bs_fn
from openai import OpenAI, RateLimitError

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_name",  default="test_results.csv",
                    help="Filename to look for inside each subfolder")
parser.add_argument("--output_name", default="evaluated_results.csv",
                    help="Filename to write inside each subfolder")
parser.add_argument("--nodes",  default=str(SCRIPT_DIR.parents[1] / "ogbn_arxiv_dataset" / "nodes_subset.csv"),
                    help="Path to nodes_subset.csv for num_of_neighbors join")
parser.add_argument("--model",  default="gpt-5.4-mini", help="OpenAI model for LLM judge")
parser.add_argument("--bertscore_batch_size", type=int, default=32,
                    help="Batch size for BERTScore; lower if you hit memory issues")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Discover the 6 subfolder result files
# ---------------------------------------------------------------------------
input_paths = sorted(SCRIPT_DIR.glob(f"*/{args.input_name}"))
print(f"Found {len(input_paths)} '{args.input_name}' file(s) under {SCRIPT_DIR}:")
for p in input_paths:
    print(f"  - {p.relative_to(SCRIPT_DIR)}")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_JUDGE_SYSTEM = """\
You are a rigorous scientific peer reviewer. Given an original paper abstract and a generated \
abstract for the same paper, evaluate the generated abstract on three metrics. \
Return valid JSON with exactly this structure:
{
  "accuracy":  <int 1-10>,
  "coherence": <int 1-10>,
  "language":  <int 1-10>
}

Rubrics:
- accuracy  (1-10): How closely does the generated abstract align with the core scientific \
objective, problem domain, and baseline methodologies of the original? \
Penalise explicit contradictions; do not penalise creative but plausible predictions.
- coherence (1-10): Is the argument logically coherent? Does it progress smoothly from a \
background problem to a proposed solution, or does it feel like a disjointed list of ideas?
- language  (1-10): Does the text adhere to scientific publishing norms? Score on precise \
domain terminology, appropriate passive/active voice, and absence of generic conversational fluff.\
"""


def call_judge(original: str, generated: str, retries: int = 6) -> dict:
    user_msg = f"ORIGINAL ABSTRACT:\n{original}\n\nGENERATED ABSTRACT:\n{generated}"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content)
        except RateLimitError:
            wait = 2 ** attempt
            print(f"    Rate limit — retrying in {wait}s …", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"    LLM error: {e}", flush=True)
            return {"accuracy": f"ERROR: {e}", "coherence": f"ERROR: {e}", "language": f"ERROR: {e}"}
    return {"accuracy": "ERROR: max retries exceeded", "coherence": "ERROR: max retries exceeded",
            "language": "ERROR: max retries exceeded"}


LLM_COLS = ["llm_accuracy", "llm_coherence", "llm_language"]
nodes = pd.read_csv(args.nodes, usecols=["node_id", "num_of_neighbors"])


def evaluate_file(input_path: Path, output_path: Path) -> None:
    print(f"\n{'=' * 80}\nProcessing {input_path.relative_to(SCRIPT_DIR)}\n{'=' * 80}")

    df = pd.read_csv(input_path)
    is_valid = (
        df["generated_abstract"].notna()
        & ~df["generated_abstract"].astype(str).str.startswith("ERROR:")
    )
    print(f"Rows: {len(df)} total | {is_valid.sum()} valid | {(~is_valid).sum()} skipped (generation errors)")

    # -----------------------------------------------------------------
    # ROUGE
    # -----------------------------------------------------------------
    print("\nComputing ROUGE …")
    rouge = rs_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_data: dict[int, dict] = {}
    for _, row in df[is_valid].iterrows():
        s = rouge.score(str(row["original_abstract"]), str(row["generated_abstract"]))
        rouge_data[int(row["node_id"])] = {
            "rouge1_f": round(s["rouge1"].fmeasure, 4),
            "rouge2_f": round(s["rouge2"].fmeasure, 4),
            "rougeL_f": round(s["rougeL"].fmeasure, 4),
        }
    print(f"  Done ({len(rouge_data)} rows).")

    # -----------------------------------------------------------------
    # BERTScore
    # -----------------------------------------------------------------
    print("\nComputing BERTScore (first run downloads ~1 GB model) …")
    valid_df   = df[is_valid].copy()
    references = valid_df["original_abstract"].tolist()
    candidates = valid_df["generated_abstract"].tolist()
    node_ids   = valid_df["node_id"].tolist()

    P, R, F1 = bs_fn(
        candidates, references,
        lang="en",
        batch_size=args.bertscore_batch_size,
        verbose=True,
    )
    bertscore_data: dict[int, dict] = {
        int(nid): {
            "bertscore_p":  round(float(p), 4),
            "bertscore_r":  round(float(r), 4),
            "bertscore_f1": round(float(f), 4),
        }
        for nid, p, r, f in zip(node_ids, P, R, F1)
    }
    print(f"  Done ({len(bertscore_data)} rows).")

    # -----------------------------------------------------------------
    # LLM judge — load existing scores for resume support
    # -----------------------------------------------------------------
    llm_cache: dict[int, dict] = {}
    if output_path.exists():
        prev = pd.read_csv(output_path)
        for _, r in prev.iterrows():
            nid = int(r["node_id"])
            if pd.notna(r.get("llm_accuracy")):
                llm_cache[nid] = {c: r.get(c) for c in LLM_COLS}
        print(f"\nLoaded {len(llm_cache)} cached LLM scores — will skip those rows.")

    print("\nRunning LLM judge …")
    rows = []
    to_judge = is_valid.sum() - len(llm_cache)
    judged = 0

    for idx, row in df.iterrows():
        nid = int(row["node_id"])
        out = row.to_dict()

        if not is_valid[idx]:
            out.update({c: None for c in ["rouge1_f", "rouge2_f", "rougeL_f",
                                           "bertscore_p", "bertscore_r", "bertscore_f1"]})
            out.update({c: "skipped (generation error)" for c in LLM_COLS})
            rows.append(out)
            continue

        out.update(rouge_data[nid])
        out.update(bertscore_data[nid])

        if nid in llm_cache:
            out.update(llm_cache[nid])
        else:
            judged += 1
            print(f"  [{judged}/{to_judge}] node_id={nid} …", end=" ", flush=True)
            result = call_judge(str(row["original_abstract"]), str(row["generated_abstract"]))
            scores = {
                "llm_accuracy":  result.get("accuracy"),
                "llm_coherence": result.get("coherence"),
                "llm_language":  result.get("language"),
            }
            print(
                f"acc={scores['llm_accuracy']}  "
                f"coh={scores['llm_coherence']}  "
                f"lang={scores['llm_language']}",
                flush=True,
            )
            out.update(scores)

        rows.append(out)

    # -----------------------------------------------------------------
    # Add neighbor count and save
    # -----------------------------------------------------------------
    out_df = pd.DataFrame(rows).merge(nodes, on="node_id", how="left")
    print(f"num_of_neighbors added for {out_df['num_of_neighbors'].notna().sum()}/{len(out_df)} rows.")

    out_df.to_csv(output_path, index=False)
    print(f"\nDone. Results written to {output_path.relative_to(SCRIPT_DIR)}")


for input_path in input_paths:
    output_path = input_path.parent / args.output_name
    evaluate_file(input_path, output_path)

print(f"\nAll {len(input_paths)} file(s) processed.")
