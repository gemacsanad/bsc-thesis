"""
evaluate_detailed.py

evaluate_results_scores_only.py with its single llm_accuracy score split into separate metrics,
so that what a model gets right can be read dimension by dimension. Everything else it reports
is kept unchanged: ROUGE, BERTScore, llm_coherence, llm_language (same rubric text, now asked in
the group-B call) and num_of_neighbors.

Four groups, from four kinds of judge call (one call per group per paper; the groups differ in
what the judge is allowed to see, which is why they are separate):

  A. Blind facets — the judge labels ONE abstract from fixed lists, without seeing the other.
     Run on the gold once (cached in --gold-facets, shared by every arm) and on each generation.
       field_match, contribution_match, method_family_match  (0/1)
       modality_jaccard                                       (0-1)
  B. Comparison — the judge sees the gold and the generation.
       subfield, problem, task                                (0-3, anchored)
       claims_total, claims_supported, claims_contradicted, claim_precision
       llm_coherence, llm_language                            (1-10, unchanged rubric)
  C. Identification — the judge sees the generation and five real abstracts (the gold and its
     four nearest papers by Voyage embedding, never its own references) and ranks them.
       id_rank (1-5), id_correct (0/1)
  D. Pairwise (only with --against) — gold + this arm + another arm, judged in both orders.
       pair_score (1 = this arm better in both orders, 0.5 = split or tie, 0 = worse)

Two metrics need no judge (TF-IDF, the instrument of src/adapter/identify.py):
       tfidf_rank                       — rank of the own gold among all golds in the file
       ontarget_gold_sim, ontarget_max_ref_sim, ontarget_margin
                                        — similarity to the gold minus the highest similarity
                                          to any of the paper's reference abstracts

Requires OPENAI_API_KEY in .env_local (or environment). Resumes from --output if it exists.

Usage:
    python src/evaluation/evaluate_detailed.py --input generations.csv --output detailed.csv
    python src/evaluation/evaluate_detailed.py --input a.csv --output a_vs_b.csv --against b.csv
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from bert_score import score as bs_fn
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from rouge_score import rouge_scorer as rs_lib
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
from identify import ranks  # noqa: E402

# Papers judged in parallel; the output is saved every SAVE_EVERY papers so a crash loses little.
WORKERS = 8
SAVE_EVERY = 50
# Distractors more similar than this to the gold are near-duplicates (other versions of the
# same paper reach 0.996), which no reader could tell apart.
DUPLICATE_COSINE = 0.95
MIN_DISTRACTOR_WORDS = 50
N_DISTRACTORS = 4

# --- fixed label sets for group A -----------------------------------------------------------
FIELDS = [
    "machine learning", "computer vision", "natural language processing", "speech and audio",
    "robotics and control", "information retrieval and recommender systems",
    "security and privacy", "theory and algorithms", "networking and distributed systems",
    "databases and data mining", "software engineering and programming languages",
    "human-computer interaction", "graphics", "other",
]
CONTRIBUTIONS = [
    "new method or model", "dataset or benchmark", "empirical or comparative study",
    "survey or review", "theoretical result", "system, tool or library",
    "application of existing methods", "position or perspective",
]
MODALITIES = [
    "text", "images", "video", "audio or speech", "graphs or networks", "tabular data",
    "time series", "source code", "3D data or point clouds", "sensor or robot data", "none",
]
METHOD_FAMILIES = [
    "transformer or large language model", "other deep neural network",
    "reinforcement learning", "probabilistic or Bayesian model", "classical machine learning",
    "optimization method", "combinatorial or graph algorithm", "formal methods or logic",
    "signal processing", "systems engineering", "mathematical analysis without a model", "other",
]


def _options(labels):
    return "\n".join(f'    "{x}"' for x in labels)


FACETS_SYSTEM = f"""\
You are labelling a computer science paper abstract. Read the abstract and return valid JSON \
with exactly this structure:
{{
  "field":         <one label from FIELD>,
  "contribution":  <one label from CONTRIBUTION>,
  "modality":      <list of one or more labels from MODALITY>,
  "method_family": <one label from METHOD_FAMILY>
}}

Copy labels exactly as written. Choose the single best label where one is asked for.

FIELD — the research area the paper belongs to:
{_options(FIELDS)}

CONTRIBUTION — the main kind of contribution the paper makes:
{_options(CONTRIBUTIONS)}

MODALITY — every kind of data the paper works on ("none" if it works on no data):
{_options(MODALITIES)}

METHOD_FAMILY — the broad family of the paper's main technique:
{_options(METHOD_FAMILIES)}"""

COMPARE_SYSTEM = """\
You are a rigorous scientific peer reviewer. Given an original paper abstract and a generated \
abstract for the same paper, compare them. Return valid JSON with exactly this structure:
{
  "subfield":            <int 0-3>,
  "problem":             <int 0-3>,
  "task":                <int 0-3>,
  "claims_total":        <int>,
  "claims_supported":    <int>,
  "claims_contradicted": <int>,
  "coherence":           <int 1-10>,
  "language":            <int 1-10>
}

subfield, problem and task use one scale:
  0 = different, 1 = related but clearly different, 2 = the same with differences in detail, \
3 = the same.
- subfield: the specific research niche, narrower than the field (e.g. "graph contrastive \
learning", not "machine learning").
- problem: the underlying problem, gap or motivation the paper addresses.
- task: the concrete task or objective the paper takes on (e.g. detection vs segmentation, \
translation vs summarisation).
Do not let the methods, models, datasets or numbers named in either abstract influence these \
three scores.

Claims: a claim is a concrete, checkable statement in the GENERATED abstract about what the \
paper does or finds — a named method, model, dataset, task, setting, quantity or result. \
Generic statements ("deep learning has achieved great success") are not claims.
- claims_total: the number of claims in the generated abstract.
- claims_supported: how many of them the original states or clearly implies.
- claims_contradicted: how many of them the original states something incompatible with.
Claims that the original neither supports nor contradicts count in neither.

- coherence (1-10): Is the argument logically coherent? Does it progress smoothly from a \
background problem to a proposed solution, or does it feel like a disjointed list of ideas?
- language  (1-10): Does the text adhere to scientific publishing norms? Score on precise \
domain terminology, appropriate passive/active voice, and absence of generic conversational fluff."""

IDENTIFY_SYSTEM = """\
You are given a generated abstract and five real paper abstracts labelled A to E. The \
generated abstract was written to describe exactly one of the five papers. Rank all five \
papers from most to least likely to be the one described. Return valid JSON with exactly this \
structure:
{
  "ranking": [<five letters, most likely first>]
}"""

PAIR_SYSTEM = """\
You are a rigorous scientific peer reviewer. Given an original paper abstract and two \
generated abstracts for the same paper, decide which generated abstract describes the same \
paper as the original more accurately: the same research problem, the same task and the same \
kind of contribution. Ignore writing quality and length. Return valid JSON with exactly this \
structure:
{
  "better": <"1", "2" or "tie">
}"""

load_dotenv(".env_local")
client = None  # set in main(), so the prompts above can be imported without an API key


def call(system, user, model, retries=6):
    """-> parsed JSON, or None on failure (the row is retried on the next run)."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content)
        except RateLimitError:
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    LLM error: {e}", flush=True)
            return None
    return None


# --- group A ---------------------------------------------------------------------------------
def facets(text, model):
    """-> {"field", "contribution", "modality": [..], "method_family"}; off-list labels -> None."""
    r = call(FACETS_SYSTEM, f"ABSTRACT:\n{text}", model)
    if r is None:
        return None

    def one(value, labels):
        value = str(value).strip().lower()
        return value if value in labels else None

    modality = r.get("modality") or []
    modality = [m.strip().lower() for m in (modality if isinstance(modality, list) else [modality])]
    return {"field": one(r.get("field"), FIELDS),
            "contribution": one(r.get("contribution"), CONTRIBUTIONS),
            "modality": sorted(m for m in set(modality) if m in MODALITIES),
            "method_family": one(r.get("method_family"), METHOD_FAMILIES)}


def facet_scores(gold, gen):
    def match(key):
        if gold[key] is None or gen[key] is None:
            return np.nan
        return float(gold[key] == gen[key])

    a, b = set(gold["modality"]), set(gen["modality"])
    return {"gold_field": gold["field"], "gen_field": gen["field"], "field_match": match("field"),
            "gold_contribution": gold["contribution"], "gen_contribution": gen["contribution"],
            "contribution_match": match("contribution"),
            "gold_modality": "|".join(gold["modality"]), "gen_modality": "|".join(gen["modality"]),
            "modality_jaccard": len(a & b) / len(a | b) if a | b else np.nan,
            "gold_method_family": gold["method_family"], "gen_method_family": gen["method_family"],
            "method_family_match": match("method_family")}


# --- group B ---------------------------------------------------------------------------------
def compare(gold, gen, model):
    r = call(COMPARE_SYSTEM, f"ORIGINAL ABSTRACT:\n{gold}\n\nGENERATED ABSTRACT:\n{gen}", model)
    if r is None:
        return None
    total = int(r.get("claims_total", 0))
    supported = min(int(r.get("claims_supported", 0)), total)
    contradicted = min(int(r.get("claims_contradicted", 0)), total - supported)
    return {"subfield": r.get("subfield"), "problem": r.get("problem"), "task": r.get("task"),
            "claims_total": total, "claims_supported": supported,
            "claims_contradicted": contradicted,
            "claim_precision": supported / total if total else np.nan,
            "llm_coherence": r.get("coherence"), "llm_language": r.get("language")}


# --- group C ---------------------------------------------------------------------------------
def distractors(node_ids, nodes, references):
    """-> {node_id: [4 distractor node_ids]}: the gold's nearest papers by Voyage embedding,
    excluding itself, its references, near-duplicates and abstracts under 50 words. Depends
    only on the gold, so every arm gets the same options."""
    emb = np.load("data/processed/joint/node_embeddings.npy", mmap_mode="r")
    query = np.asarray(emb[node_ids])
    words = nodes["abstract"].fillna("").str.split().str.len().to_numpy()
    candidates = []
    for start in range(0, len(emb), 50_000):
        sim = query @ np.asarray(emb[start:start + 50_000]).T
        top = np.argpartition(-sim, 100, axis=1)[:, :100]
        candidates.append(np.stack([top + start, np.take_along_axis(sim, top, 1)], -1))
    candidates = np.concatenate(candidates, 1)
    out = {}
    for i, nid in enumerate(node_ids):
        order = candidates[i][np.argsort(-candidates[i][:, 1])]
        banned = references.get(nid, set()) | {nid}
        picked = [int(c) for c, s in order
                  if int(c) not in banned and s < DUPLICATE_COSINE
                  and words[int(c)] >= MIN_DISTRACTOR_WORDS]
        out[nid] = picked[:N_DISTRACTORS]
    return out


def identify(nid, gen, gold, options, nodes, model):
    """Options are shuffled with the node_id as seed, so the gold's position is fixed per paper
    and the same in every arm."""
    ids = [nid] + options
    order = np.random.default_rng(nid).permutation(len(ids))
    letters = "ABCDE"
    shown = {letters[k]: ids[j] for k, j in enumerate(order)}
    text = {i: (gold if i == nid else nodes.at[i, "abstract"]) for i in ids}
    user = f"GENERATED ABSTRACT:\n{gen}\n\n" + "\n\n".join(
        f"PAPER {k}:\n{text[i]}" for k, i in shown.items())
    r = call(IDENTIFY_SYSTEM, user, model)
    if r is None:
        return None
    ranking = [str(x).strip().upper()[:1] for x in r.get("ranking", [])]
    gold_letter = next(k for k, i in shown.items() if i == nid)
    rank = ranking.index(gold_letter) + 1 if gold_letter in ranking else len(ids)
    return {"id_rank": rank, "id_correct": float(rank == 1),
            "id_options": "|".join(str(i) for i in options)}


# --- group D ---------------------------------------------------------------------------------
def pairwise(gold, gen, other, model):
    """Judged in both orders to cancel position bias."""
    score = []
    for first, second, mine in [(gen, other, "1"), (other, gen, "2")]:
        r = call(PAIR_SYSTEM, f"ORIGINAL ABSTRACT:\n{gold}\n\nGENERATED ABSTRACT 1:\n{first}"
                              f"\n\nGENERATED ABSTRACT 2:\n{second}", model)
        if r is None:
            return None
        better = str(r.get("better", "")).strip().strip('"').lower()
        score.append(1.0 if better == mine else 0.5 if better == "tie" else 0.0)
    return {"pair_score": float(np.mean(score))}


# --- no judge, kept from evaluate_results_scores_only.py --------------------------------------
def rouge_bertscore(df):
    rouge = rs_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for gold, gen in zip(df["original_abstract"].astype(str), df["generated_abstract"].astype(str)):
        s = rouge.score(gold, gen)
        rows.append({"rouge1_f": round(s["rouge1"].fmeasure, 4),
                     "rouge2_f": round(s["rouge2"].fmeasure, 4),
                     "rougeL_f": round(s["rougeL"].fmeasure, 4)})
    out = pd.DataFrame(rows, index=df.index)
    P, R, F1 = bs_fn(df["generated_abstract"].astype(str).tolist(),
                     df["original_abstract"].astype(str).tolist(), lang="en", batch_size=32,
                     verbose=True)
    out["bertscore_p"] = P.numpy().round(4)
    out["bertscore_r"] = R.numpy().round(4)
    out["bertscore_f1"] = F1.numpy().round(4)
    return out


# --- no judge --------------------------------------------------------------------------------
def tfidf_metrics(df, nodes, references):
    """tfidf_rank as in identify.py; on-target margin with a vectoriser fitted on the golds and
    every reference abstract, so both similarities are on one scale."""
    gens = df["generated_abstract"].astype(str).tolist()
    golds = df["original_abstract"].astype(str).tolist()
    out = pd.DataFrame({"tfidf_rank": ranks(gens, golds)}, index=df.index)
    ref_texts = {nid: nodes.loc[sorted(references.get(nid, set())), "abstract"]
                 .dropna().astype(str).tolist() for nid in df["node_id"]}
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(
        golds + [t for ts in ref_texts.values() for t in ts])
    g, o = vec.transform(gens), vec.transform(golds)
    out["ontarget_gold_sim"] = np.asarray(g.multiply(o).sum(1)).ravel()
    max_ref = []
    for k, nid in enumerate(df["node_id"]):
        refs = ref_texts[nid]
        max_ref.append(float((vec.transform(refs) @ g[k].T).max()) if refs else np.nan)
    out["ontarget_max_ref_sim"] = max_ref
    out["ontarget_margin"] = out["ontarget_gold_sim"] - out["ontarget_max_ref_sim"]
    return out


def main():
    global client
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generations.csv")
    parser.add_argument("--output", default="detailed.csv")
    parser.add_argument("--against", default=None,
                        help="a second generations.csv on the same papers; adds pair_score")
    parser.add_argument("--gold-facets", default="data/frozen/gold_facets.json",
                        help="group-A labels of the gold abstracts, shared by every arm")
    parser.add_argument("--nodes", default="data/processed/joint/nodes.csv")
    parser.add_argument("--edges", default="data/processed/joint/edges.csv")
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    df = pd.read_csv(args.input)
    df = df[df["generated_abstract"].notna()
            & ~df["generated_abstract"].astype(str).str.startswith("ERROR:")].reset_index(drop=True)
    node_ids = df["node_id"].astype(int).tolist()
    print(f"Rows: {len(df)} valid")

    nodes = pd.read_csv(args.nodes, usecols=["node_id", "abstract"]).set_index("node_id")
    edges = pd.read_csv(args.edges)  # src cites dst: out-edges are the paper's references
    edges = edges[edges["src_node_id"].isin(node_ids)]
    references = edges.groupby("src_node_id")["dst_node_id"].agg(set).to_dict()
    df["num_of_neighbors"] = df["node_id"].map(lambda n: len(references.get(n, ())))

    print("ROUGE and BERTScore (first run downloads ~1 GB model) …")
    df = df.join(rouge_bertscore(df))
    print("TF-IDF metrics …")
    df = df.join(tfidf_metrics(df, nodes, references))
    print("Distractors …")
    options = distractors(node_ids, nodes, references)
    other = None
    if args.against:
        other = pd.read_csv(args.against).set_index("node_id")["generated_abstract"].astype(str)

    gold_facets = {}
    if os.path.exists(args.gold_facets):
        gold_facets = {int(k): v for k, v in json.load(open(args.gold_facets)).items()}
    done = {}
    if os.path.exists(args.output):
        done = pd.read_csv(args.output).set_index("node_id").to_dict("index")
        print(f"Resuming: {len(done)} rows in {args.output}")
    lock = threading.Lock()

    def has(nid, col):
        return nid in done and pd.notna(done[nid].get(col))

    def judge(row):
        nid, gen, gold = int(row["node_id"]), str(row["generated_abstract"]), str(row["original_abstract"])
        out = {k: v for k, v in done.get(nid, {}).items() if pd.notna(v)}
        if not has(nid, "field_match"):
            with lock:
                g = gold_facets.get(nid)
            if g is None:
                g = facets(gold, args.model)
                if g is not None:
                    with lock:
                        gold_facets[nid] = g
            f = facets(gen, args.model)
            if g is not None and f is not None:
                out.update(facet_scores(g, f))
        if not has(nid, "subfield"):
            out.update(compare(gold, gen, args.model) or {})
        if not has(nid, "id_rank") and len(options[nid]) == N_DISTRACTORS:
            out.update(identify(nid, gen, gold, options[nid], nodes, args.model) or {})
        if other is not None and not has(nid, "pair_score") and nid in other.index:
            out.update(pairwise(gold, gen, other[nid], args.model) or {})
        return nid, out

    def save():
        with lock:
            judged = pd.DataFrame.from_dict(done, orient="index")
            judged = judged.drop(columns=[c for c in judged.columns if c in df.columns], errors="ignore")
            df.join(judged, on="node_id").to_csv(args.output, index=False)
            os.makedirs(os.path.dirname(args.gold_facets) or ".", exist_ok=True)
            json.dump({str(k): v for k, v in gold_facets.items()}, open(args.gold_facets, "w"))

    print("Running LLM judge …")
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(judge, row) for _, row in df.iterrows()]
        for k, fut in enumerate(as_completed(futures), 1):
            nid, out = fut.result()
            with lock:
                done[nid] = out
            print(f"  [{k}/{len(df)}] node_id={nid}  subfield={out.get('subfield')}  "
                  f"task={out.get('task')}  id_rank={out.get('id_rank')}", flush=True)
            if k % SAVE_EVERY == 0:
                save()
    save()

    result = pd.read_csv(args.output)
    cols = ["field_match", "contribution_match", "modality_jaccard", "method_family_match",
            "subfield", "problem", "task", "claims_total", "claim_precision",
            "claims_contradicted", "llm_coherence", "llm_language", "id_correct", "id_rank", "pair_score", "tfidf_rank",
            "ontarget_margin"]
    print("\nMeans:")
    for c in cols:
        if c in result:
            v = pd.to_numeric(result[c], errors="coerce")
            print(f"  {c:<22} {v.mean():.3f}   (n={v.notna().sum()})")
    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
