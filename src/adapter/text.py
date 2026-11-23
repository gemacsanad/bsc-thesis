"""The corpus, the prompt of the reader and the construction of the frozen node sets."""

import argparse
import json
import os
import random
import re

import pandas as pd
from transformers import AutoTokenizer

from config import (BASE_MODEL, DATASET, FROZEN_DIR, GENERATION_RESERVE, JOINT_DIR, MAX_LENGTH,
                    MAX_OUT_DEGREE, MIN_OUT_DEGREE, NUM_PREFIX_TOKENS,
                    TRAIN_SUBSET, VALID_PER_DEGREE)
from neighbour_cap import fit_refs

FROZEN_REFS = os.path.join(FROZEN_DIR, "prompt_refs_capped.json")

FREEZE_PREFIX_TOKENS = 10

_frozen_refs = None


def frozen_refs():
    """-> {node_id: [ref_node_id, ...]}, the bibliography every model reads. Cached."""
    global _frozen_refs
    if _frozen_refs is None:
        with open(FROZEN_REFS) as f:
            _frozen_refs = {int(k): v for k, v in json.load(f).items()}
    return _frozen_refs


# --- The prompt of the reader -------------------------------------------------------------------
PROMPT = "graph-only"       # recorded in every checkpoint; the text-path runs had no such key

GRAPH_ONLY_SYSTEM = (
    "You are a scientific writing assistant. "
    "The context you have been given encodes the reference list of one paper, whose own "
    "abstract is missing. That paper is not among its references: it is a separate, more "
    "recent work that cites all of them. "
    "Reconstruct its abstract as continuous prose covering, in order: the problem it "
    "addresses; what the work it cites has already established about that problem; what it "
    "does about what remains open, described by the mechanism it uses the kind of model, "
    "data or analysis rather than by a name; and what it evaluates. "
    "Where the context leaves the paper's specific contribution undetermined, state what "
    "it does determine about its problem and approach, and stop there — a correct general "
    "statement is worth more than a specific guess. "
    "Write in the voice of the paper's own authors: never mention the cited works as a "
    "collection, how many there are, or the authors, and do not restate any single one "
    "of them. "
    "Write plain prose of 150-180 words, with no markup, mathematical notation or headings. "
    "Output ONLY the abstract text."
)

GRAPH_ONLY_USER = "Write the abstract."

if DATASET == "children":
    GRAPH_ONLY_SYSTEM = (
        "You are a children's book copywriter. "
        "The context you have been given encodes the books customers bought together with one "
        "children's book, whose own description is missing. That book is not among them: it is "
        "a separate book that was bought alongside all of them. "
        "Reconstruct its description as continuous prose covering: what kind of book it is and "
        "who it is for, its format and the age of its readers; what it is about, its subject, "
        "characters or story; and what makes it worth reading. "
        "Where the context leaves the book's specific content undetermined, state what it does "
        "determine about its kind, readers and subject, and stop there — a correct general "
        "statement is worth more than a specific guess. "
        "Write in the voice of the book's own description: never mention the other books, how "
        "many there are, or that customers bought them, and do not describe any single one of "
        "them. "
        "Write plain prose with no markup or headings. "
        "Output ONLY the description text."
    )
    GRAPH_ONLY_USER = "Write the description."

# --- The books' text arm: the neighbours as text, no prefix -----------------------------------
BOOK_TEXT_SYSTEM = (
    "You are a children's book copywriter. "
    "The books listed below are the books customers bought together with one children's "
    "book, whose own description is missing. That book is not among them: it is a separate "
    "book that was bought alongside all of them. "
    "Reconstruct its description as continuous prose covering: what kind of book it is and "
    "who it is for, its format and the age of its readers; what it is about, its subject, "
    "characters or story; and what makes it worth reading. "
    "Where the listed books leave the book's specific content undetermined, state what they "
    "do determine about its kind, readers and subject, and stop there — a correct general "
    "statement is worth more than a specific guess. "
    "Write in the voice of the book's own description: never mention the other books, how "
    "many there are, or that customers bought them, and do not describe any single one of "
    "them. "
    "Write plain prose with no markup or headings. "
    "Output ONLY the description text."
)
BOOK_TEXT_REFS = os.path.join(FROZEN_DIR, "prompt_refs_text.json")
_book_text_refs = None


def build_book_text_prompt(tokenizer, neighbours):
    """`neighbours` is a list of (title, description). ChatML, thinking disabled."""
    books = "\n\n".join(f"- Title: {t}\n  Description: {d}" for t, d in neighbours)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": BOOK_TEXT_SYSTEM},
         {"role": "user", "content": f"Books bought together with it:\n\n{books}\n\n"
                                     f"{GRAPH_ONLY_USER}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def book_text_example(tokenizer, corpus, node_id, refs=None):
    """-> (prompt, target) for the text arm. `refs` defaults to the frozen list for the node."""
    global _book_text_refs
    if refs is None:
        if _book_text_refs is None:
            with open(BOOK_TEXT_REFS) as f:
                _book_text_refs = {int(k): v for k, v in json.load(f).items()}
        refs = _book_text_refs[node_id]
    return (build_book_text_prompt(tokenizer, [(corpus.title[n], corpus.abstract[n])
                                               for n in refs]),
            corpus.abstract[node_id])


def build_graph_only_prompt(tokenizer):
    """The one prompt every node gets, in Qwen3's ChatML with thinking disabled."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": GRAPH_ONLY_SYSTEM},
         {"role": "user", "content": GRAPH_ONLY_USER}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def graph_only_example(tokenizer, corpus, node_id):
    """-> (prompt, target) for one node. The single builder train.py and generate.py share."""
    return build_graph_only_prompt(tokenizer), corpus.abstract[node_id]


SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


INSTRUCTION = (
    "You are a scientific writing assistant. "
    "Based on the source publications listed below, write a concise abstract "
    "for a new application paper that builds on and combines these works. "
    "Output ONLY the abstract text, without any heading or explanation."
)


def user_message(neighbours):
    """The experimental content: the instruction and the reference block."""
    sources = "\n\n".join(f"- Title: {t}\n  Abstract: {a}" for t, a in neighbours)
    return (f"{INSTRUCTION}\n\n"
            f"Source publications:\n{sources or '(no neighbouring sources available)'}\n\n"
            f"Generated abstract:")


def build_prompt(tokenizer, neighbours):
    """The user message wrapped in the model's own chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_message(neighbours)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build_prompt_vicuna(neighbours):
    """The Vicuna chat wrapper, under which the frozen reference lists were computed."""
    return f"{SYSTEM} USER: {user_message(neighbours)} ASSISTANT:"


def encode(tokenizer, prompt, target):
    """-> (input_ids, labels). Labels are -100 over the prompt, so only the abstract trains."""
    prompt_ids = tokenizer(prompt).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids


def fits(input_ids):
    """The prefix occupies real attention positions, so it counts against the context."""
    return len(input_ids) + NUM_PREFIX_TOKENS <= MAX_LENGTH


# --- Target cleaning ----------------------------------------------------------------------------
_UNWRAP = re.compile(r"\\(?:textit|textbf|emph|texttt|textsc|textrm|mathrm|mathbf|mathit|text)"
                     r"\s*\{([^{}]*)\}")
# The old switch form, "{\em text}".
_UNWRAP_SWITCH = re.compile(r"\{\\(?:em|it|bf|tt|sl|sc|rm)\s+([^{}]*)\}")
_MATH_DELIM = re.compile(r"(?<!\\)\$+")
# Maths delimiters are removed and the maths inside stays.
_ESCAPES = {r"\%": "%", r"\&": "&", r"\_": "_", r"\$": "$", r"\#": "#", r"\\": " ",
            r"\ ": " ", r"\(": " ", r"\)": " ", r"\[": " ", r"\]": " "}
# Accent escapes: "Verd\'u" -> "Verdu".
_ACCENT = re.compile(r"""\\['"`^~=.]""")
_BRACES = re.compile(r"[{}]")
_COMMAND = re.compile(r"\\([a-zA-Z]+)\s*")
_WHITESPACE = re.compile(r"\s+")


def clean_abstract(text):
    r"""Strip LaTeX markup and maths delimiters, keeping their contents. -> str"""
    text = str(text)
    for _ in range(3):                       # nested \textbf{\textit{x}}; 3 covers the corpus
        text, n = _UNWRAP.subn(r"\1", text)
        text, m = _UNWRAP_SWITCH.subn(r"\1", text)
        if not n and not m:
            break
    text = _MATH_DELIM.sub("", text)         # delimiters only; the maths itself stays
    for src, dst in _ESCAPES.items():
        text = text.replace(src, dst)
    text = _ACCENT.sub("", text)
    text = _COMMAND.sub(r"\1 ", text)
    text = _BRACES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


class Corpus:
    """Node text and reference lists, for turning a node_id into a training example."""

    def __init__(self, nodes=None, edges=None, clean=True):
        nodes = nodes or os.path.join(JOINT_DIR, "nodes.csv")
        edges = edges or os.path.join(JOINT_DIR, "edges.csv")

        df = pd.read_csv(nodes, usecols=["node_id", "title", "abstract"]).set_index("node_id")
        self.title = df["title"].fillna("")
        self.abstract = df["abstract"].fillna("")
        if clean:
            self.abstract = self.abstract.map(clean_abstract)
        self.refs = pd.read_csv(edges).groupby("src_node_id")["dst_node_id"].apply(list)

    def example(self, tokenizer, node_id):
        """-> (prompt, target_abstract) for one node, with every reference shown."""
        refs = self.refs.get(node_id, [])
        return (build_prompt(tokenizer, [(self.title[n], self.abstract[n]) for n in refs]),
                self.abstract[node_id])

    def out_degree(self, node_id):
        return len(self.refs.get(node_id, []))


def capped_example(tokenizer, corpus, embeddings, node_id, frozen=True):
    """-> (prompt, target, kept_refs) for one node, trimmed to fit the context."""
    target = corpus.abstract[node_id]

    if frozen:
        refs = frozen_refs()[node_id]
    else:
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        reserve = max(GENERATION_RESERVE, len(target_ids) + 1)   # +1 for the EOS token
        budget = MAX_LENGTH - FREEZE_PREFIX_TOKENS - reserve

        all_refs = corpus.refs.get(node_id, [])

        def measure(kept):
            prompt = build_prompt_vicuna([(corpus.title[n], corpus.abstract[n]) for n in kept])
            return len(tokenizer(prompt).input_ids)

        refs = fit_refs(all_refs, embeddings, measure, budget)
        return (build_prompt_vicuna([(corpus.title[n], corpus.abstract[n]) for n in refs]),
                target, refs)

    return (build_prompt(tokenizer, [(corpus.title[n], corpus.abstract[n]) for n in refs]),
            target, refs)


def collect(tokenizer, corpus, candidates, wanted, rng):
    """Walk shuffled candidates, keeping those that fit, until `wanted` are found."""
    pool = list(candidates)
    rng.shuffle(pool)
    kept, lengths, seen = [], [], 0
    for node_id in pool:
        seen += 1
        input_ids, _ = encode(tokenizer, *corpus.example(tokenizer, node_id))
        if fits(input_ids):
            kept.append(int(node_id))
            lengths.append(len(input_ids))
            if len(kept) == wanted:
                break
    return kept, lengths, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=BASE_MODEL, help="tokenizer to measure with")
    ap.add_argument("--nodes", default=os.path.join(JOINT_DIR, "nodes.csv"))
    ap.add_argument("--edges", default=os.path.join(JOINT_DIR, "edges.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=FROZEN_DIR)
    args = ap.parse_args()

    print(f"Loading tokenizer {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("Loading corpus …")
    corpus = Corpus(args.nodes, args.edges)

    with open(os.path.join(FROZEN_DIR, "splits_eligible.json")) as f:
        splits = json.load(f)

    rng = random.Random(args.seed)

    print(f"\nCollecting {TRAIN_SUBSET:,} training nodes …")
    train, train_len, train_seen = collect(tokenizer, corpus, splits["train"], TRAIN_SUBSET, rng)
    assert len(train) == TRAIN_SUBSET, f"only {len(train)} training nodes fit the context"
    print(f"  {len(train):,} kept from {train_seen:,} examined "
          f"({1 - len(train) / train_seen:.1%} overflowed)")

    print(f"\nCollecting {VALID_PER_DEGREE} validation nodes at each out-degree "
          f"{MIN_OUT_DEGREE}-{MAX_OUT_DEGREE} …")
    by_degree = {}
    for node_id in splits["valid"]:
        by_degree.setdefault(corpus.out_degree(node_id), []).append(node_id)

    valid, valid_len = [], []
    for degree in range(MIN_OUT_DEGREE, MAX_OUT_DEGREE + 1):
        kept, lengths, seen = collect(tokenizer, corpus, by_degree.get(degree, []),
                                      VALID_PER_DEGREE, rng)
        assert len(kept) == VALID_PER_DEGREE, \
            f"only {len(kept)} valid nodes at out-degree {degree} fit the context"
        valid += kept
        valid_len += lengths
        print(f"  degree {degree:>2}: {len(kept)} kept from {seen} examined, "
              f"tokens mean {sum(lengths) / len(lengths):,.0f}")

    # --- checks -----------------------------------------------------------
    with open(os.path.join(FROZEN_DIR, "eval_nodes.json")) as f:
        eval_nodes = json.load(f)
    assert len(set(train)) == len(train) and len(set(valid)) == len(valid), "duplicates"
    assert not set(train) & set(valid), "train and valid overlap"
    for name, ids in (("train", train), ("valid", valid)):
        assert not set(ids) & set(eval_nodes), f"{name} overlaps the frozen eval set"
        assert set(ids) <= set(splits[name]), f"{name} contains ineligible nodes"

    # --- write ------------------------------------------------------------
    path = os.path.join(args.out_dir, "training_sets.json")
    with open(path, "w") as f:
        json.dump({"train": sorted(train), "valid": sorted(valid)}, f)

    # --- report -----------------------------------------------------------
    for name, lengths in (("train", train_len), ("valid", valid_len)):
        lengths = sorted(lengths)
        print(f"\n{name}: {len(lengths):,} nodes, sequence length "
              f"mean {sum(lengths) / len(lengths):,.0f}, "
              f"median {lengths[len(lengths) // 2]:,}, "
              f"p99 {lengths[int(len(lengths) * 0.99)]:,}, max {lengths[-1]:,}")

    # The eval set is frozen and is not filtered.
    over = sum(not fits(encode(tokenizer, *corpus.example(tokenizer, n))[0]) for n in eval_nodes)
    print(f"\neval set: {over}/{len(eval_nodes)} nodes overflow the context "
          f"({over / len(eval_nodes):.1%}) — left in place, Stage 5's problem")

    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
