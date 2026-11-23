"""
evaluate_results_scores_only.py

Scores generated abstracts with ROUGE, BERTScore and an LLM judge that returns only numeric
scores, with no free-text reasoning.

Metrics added per row:
  rouge1_f, rouge2_f, rougeL_f          — ROUGE F1 scores
  bertscore_p, bertscore_r, bertscore_f1 — BERTScore
  llm_accuracy  (1-10)                   — alignment with original objective/domain/methods
  llm_coherence (1-10)                   — logical flow from problem to solution
  llm_language  (1-10)                   — scientific register, terminology, no fluff

--dataset children scores children's book descriptions in the same three columns.

Requires OPENAI_API_KEY in .env (or environment).

Usage:
    python evaluate_results_scores_only.py [--input test_results.csv] [--output evaluated_results_scores_only.csv]

Install dependencies:
    pip install rouge-score bert-score openai python-dotenv pandas torch
"""

import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from rouge_score import rouge_scorer as rs_lib
from bert_score import score as bs_fn
from openai import (APIConnectionError, APITimeoutError, InternalServerError, OpenAI,
                    RateLimitError)

load_dotenv(".env_local")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="generations.csv")
parser.add_argument("--output", default="scored_scores_only.csv")
parser.add_argument("--dataset", choices=["papers", "children"], default="papers",
                    help="which judge prompt to use: paper abstracts or children's book descriptions")
parser.add_argument("--edges",  default="data/processed/joint/edges.csv",
                    help="Path to joint edges.csv used to compute num_of_neighbors (out-degree)")
parser.add_argument("--model",  default="gpt-5.4-mini", help="OpenAI model for LLM judge")
parser.add_argument("--no_llm", action="store_true",
                    help="ROUGE and BERTScore only; leave the old 1-10 judge columns empty "
                         "(thesis Ch. 8 uses judge_scores.py instead)")
parser.add_argument("--bertscore_batch_size", type=int, default=32,
                    help="Batch size for BERTScore; lower if you hit memory issues")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(args.input)
is_valid = (
    df["generated_abstract"].notna()
    & ~df["generated_abstract"].astype(str).str.startswith("ERROR:")
)
print(f"Rows: {len(df)} total | {is_valid.sum()} valid | {(~is_valid).sum()} skipped (generation errors)")

# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------
print("\nComputing ROUGE …")
rouge = rs_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge_data: dict[int, dict] = {}
for idx, row in df[is_valid].iterrows():
    s = rouge.score(str(row["original_abstract"]), str(row["generated_abstract"]))
    rouge_data[int(row["node_id"])] = {
        "rouge1_f": round(s["rouge1"].fmeasure, 4),
        "rouge2_f": round(s["rouge2"].fmeasure, 4),
        "rougeL_f": round(s["rougeL"].fmeasure, 4),
    }
print(f"  Done ({len(rouge_data)} rows).")

# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# LLM judge — load existing scores for resume support
# ---------------------------------------------------------------------------
LLM_COLS = ["llm_accuracy", "llm_coherence", "llm_language"]

llm_cache: dict[int, dict] = {}
if os.path.exists(args.output):
    prev = pd.read_csv(args.output)
    for _, r in prev.iterrows():
        nid = int(r["node_id"])
        # Only real scores are cached: an "ERROR: ..." row is judged again on the next run.
        if pd.notna(pd.to_numeric(r.get("llm_accuracy"), errors="coerce")):
            llm_cache[nid] = {c: r.get(c) for c in LLM_COLS}
    print(f"\nLoaded {len(llm_cache)} cached LLM scores — will skip those rows.")

# A stalled request otherwise waits the SDK default of 600 s, three times, and the run looks
# frozen. Retries are handled in call_judge instead.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60, max_retries=0)

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

# Children's books, in the same shape as the paper prompt. Accuracy names the aspects where the
# GNN+LLM arms differ from the baselines: the specific
# subject, the names, and the kind of book; review boilerplate is excluded because it gave the
# GNN arms a false lead on names.
_BOOK_JUDGE_SYSTEM = """\
You are an experienced children's book editor. Given an original book description and a generated \
description for the same book, evaluate the generated description on three metrics. \
Return valid JSON with exactly this structure:
{
  "accuracy":  <int 1-10>,
  "coherence": <int 1-10>,
  "language":  <int 1-10>
}

Rubrics:
- accuracy  (1-10): How recognisably does the generated description describe the same book? \
Reward matching specifics: the subject and premise, the names of characters, author, illustrator \
or series, and the kind of book (board book, picture book, chapter book, novel, nonfiction). \
Generic text that could fit any children's book scores low; ignore review boilerplate such as \
journal names and copyright lines. Penalise explicit contradictions; do not penalise creative but \
plausible details.
- coherence (1-10): Is it one consistent description of one book, or does it repeat itself, \
loop, break off, or drift between different stories?
- language  (1-10): Does it read like real published copy for a children's book, such as a \
publisher's blurb or a library review? Score on concrete, natural wording and absence of filler \
that says nothing about the book.\
"""
JUDGE_SYSTEM = _BOOK_JUDGE_SYSTEM if args.dataset == "children" else _JUDGE_SYSTEM
TEXT_LABEL = "DESCRIPTION" if args.dataset == "children" else "ABSTRACT"


def call_judge(original: str, generated: str, retries: int = 6) -> dict:
    user_msg = f"ORIGINAL {TEXT_LABEL}:\n{original}\n\nGENERATED {TEXT_LABEL}:\n{generated}"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            wait = 2 ** attempt
            print(f"    {type(e).__name__} — retrying in {wait}s …", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"    LLM error: {e}", flush=True)
            return {"accuracy": f"ERROR: {e}", "coherence": f"ERROR: {e}",
                    "language": f"ERROR: {e}"}
    return {"accuracy": "ERROR: max retries exceeded", "coherence": "ERROR: max retries exceeded",
            "language": "ERROR: max retries exceeded"}


# ---------------------------------------------------------------------------
# Build output rows
# ---------------------------------------------------------------------------
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

    if args.no_llm:
        out.update({c: None for c in LLM_COLS})
    elif nid in llm_cache:
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
        # Save as it goes, so a stopped run resumes from here rather than from row 0.
        if judged % 10 == 0:
            pd.DataFrame(rows + [out]).to_csv(args.output, index=False)

    rows.append(out)

# ---------------------------------------------------------------------------
# Add neighbor count — out-degree computed from the joint edges.csv (src cites
# dst, so out-edges are the paper's own references). nodes_subset.csv only
# covers the old 200-node test set, so joining against it leaves most rows
# empty for the current eval sets.
# ---------------------------------------------------------------------------
edges = pd.read_csv(args.edges, usecols=["src_node_id"])
out_degree = edges.groupby("src_node_id").size()
df = pd.DataFrame(rows)
df["num_of_neighbors"] = df["node_id"].map(out_degree).fillna(0).astype(int)
print(f"num_of_neighbors added for {df['num_of_neighbors'].notna().sum()}/{len(df)} rows.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df.to_csv(args.output, index=False)
print(f"\nDone. Results written to {args.output}")
