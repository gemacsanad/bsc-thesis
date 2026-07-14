"""
evaluate_results.py

Evaluates generated abstracts against originals and writes an enriched CSV.

Metrics added per row:
  rouge1_f, rouge2_f, rougeL_f          — ROUGE F1 scores
  bertscore_p, bertscore_r, bertscore_f1 — BERTScore
  llm_accuracy  (1-10) + reason          — alignment with original objective/domain/methods
  llm_coherence (1-10) + reason          — logical flow from problem to solution
  llm_language  (1-10) + reason          — scientific register, terminology, no fluff

Requires OPENAI_API_KEY in .env (or environment).

Usage:
    python evaluate_results.py [--input test_results.csv] [--output evaluated_results.csv]

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
from openai import OpenAI, RateLimitError

load_dotenv()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="test_results.csv")
parser.add_argument("--output", default="evaluated_results.csv")
parser.add_argument("--nodes",  default="ogbn_arxiv_dataset/nodes_subset.csv",
                    help="Path to nodes_subset.csv for num_of_neighbors join")
parser.add_argument("--model",  default="gpt-5.4-mini", help="OpenAI model for LLM judge")
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
LLM_COLS = ["llm_accuracy", "llm_coherence", "llm_language", "llm_reason"]

_CATEGORY_NAMES = [
    "wrong_domain", "multi_paper_contamination", "wrong_methodology",
    "wrong_task", "too_generic", "truncated_incomplete",
    "correct_domain", "correct_task", "methodology_partially_correct",
    "scientific_language", "coherent_structure",
]
CATEGORY_COLS = [f"cat_{n}" for n in _CATEGORY_NAMES]

llm_cache: dict[int, dict] = {}
cat_cache: dict[int, dict] = {}
if os.path.exists(args.output):
    prev = pd.read_csv(args.output)
    for _, r in prev.iterrows():
        nid = int(r["node_id"])
        if pd.notna(r.get("llm_accuracy")):
            llm_cache[nid] = {c: r.get(c) for c in LLM_COLS}
        if pd.notna(r.get("cat_wrong_domain")):
            cat_cache[nid] = {c: r.get(c) for c in CATEGORY_COLS}
    print(f"\nLoaded {len(llm_cache)} cached LLM scores and {len(cat_cache)} cached categories — will skip those rows.")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_JUDGE_SYSTEM = """\
You are a rigorous scientific peer reviewer. Given an original paper abstract and a generated \
abstract for the same paper, evaluate the generated abstract on three metrics. \
Return valid JSON with exactly this structure:
{
  "accuracy":  <int 1-10>,
  "coherence": <int 1-10>,
  "language":  <int 1-10>,
  "reason":    <string, 3-4 sentences explaining the scores overall>
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


_CATEGORIZER_SYSTEM = """\
You are a classifier for scientific abstract evaluation reasoning. Given a judge's brief \
reasoning about a generated abstract, classify it into the categories below. \
Return valid JSON with exactly these keys — each a boolean:
{
  "wrong_domain":                <bool>,
  "multi_paper_contamination":   <bool>,
  "wrong_methodology":           <bool>,
  "wrong_task":                  <bool>,
  "too_generic":                 <bool>,
  "truncated_incomplete":        <bool>,
  "correct_domain":              <bool>,
  "correct_task":                <bool>,
  "methodology_partially_correct": <bool>,
  "scientific_language":         <bool>,
  "coherent_structure":          <bool>
}

Category definitions:
- wrong_domain: The generated abstract is about a completely different research field or \
application domain from the original.
- multi_paper_contamination: The text reads like a stitched-together mixture of multiple \
unrelated papers rather than one coherent abstract.
- wrong_methodology: The domain is broadly correct but describes different techniques or \
algorithms than the original.
- wrong_task: Same broad field but the specific problem being solved is different (e.g., \
classification instead of regression, MT instead of summarisation).
- too_generic: Correct domain but described at such a high level of generality that it \
lacks the specific terminology and framing of the original.
- truncated_incomplete: The abstract ends abruptly mid-sentence or is clearly unfinished.
- correct_domain: The research field is correctly identified.
- correct_task: The specific problem or task is correctly described.
- methodology_partially_correct: At least some of the methods or techniques are correctly \
named and used in the right context.
- scientific_language: The text uses appropriate academic register, domain terminology, \
and publication-like phrasing.
- coherent_structure: The abstract has a clear logical progression from background to \
problem to method to results.\
"""


def call_categorizer(reason: str, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": _CATEGORIZER_SYSTEM},
                    {"role": "user",   "content": reason},
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
            print(f"    Categorizer error: {e}", flush=True)
            return {}
    return {}


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
            return {"accuracy": f"ERROR: {e}", "coherence": f"ERROR: {e}",
                    "language": f"ERROR: {e}", "reason": f"ERROR: {e}"}
    return {"accuracy": "ERROR: max retries exceeded", "coherence": "ERROR: max retries exceeded",
            "language": "ERROR: max retries exceeded", "reason": "ERROR: max retries exceeded"}


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
        out.update({c: "skipped (generation error)" for c in ["llm_accuracy", "llm_coherence", "llm_language"]})
        out["llm_reason"] = None
        out.update({c: None for c in CATEGORY_COLS})
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
            "llm_reason":    result.get("reason"),
        }
        print(
            f"acc={scores['llm_accuracy']}  "
            f"coh={scores['llm_coherence']}  "
            f"lang={scores['llm_language']}",
            flush=True,
        )
        out.update(scores)

    if nid in cat_cache:
        out.update(cat_cache[nid])
    else:
        reason_text = out.get("llm_reason")
        if reason_text and not str(reason_text).startswith("ERROR:"):
            cat_result = call_categorizer(str(reason_text))
            cat_scores = {f"cat_{n}": int(bool(cat_result.get(n, False))) for n in _CATEGORY_NAMES}
        else:
            cat_scores = {c: None for c in CATEGORY_COLS}
        out.update(cat_scores)

    rows.append(out)

# ---------------------------------------------------------------------------
# Add neighbor count
# ---------------------------------------------------------------------------
nodes = pd.read_csv(args.nodes, usecols=["node_id", "num_of_neighbors"])
df = pd.DataFrame(rows).merge(nodes, on="node_id", how="left")
print(f"num_of_neighbors added for {df['num_of_neighbors'].notna().sum()}/{len(df)} rows.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df.to_csv(args.output, index=False)
print(f"\nDone. Results written to {args.output}")
