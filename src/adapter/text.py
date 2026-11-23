"""
Usage:
    python adapter/text.py          # freeze the subsets (needs the Vicuna tokenizer)
"""

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

# Which references a prompt shows was decided once, under the Vicuna tokenizer, and written
# here. It used to be recomputed per call, which made the bibliography a function of the
# *tokenizer*: Qwen3's BPE is more efficient on English than Vicuna's SentencePiece, so more
# references would survive the same budget and the two models would silently read different
# evidence.
FROZEN_REFS = os.path.join(FROZEN_DIR, "prompt_refs_capped.json")

# The freeze is defined by four things, all of which have to be pinned for it to stay
# re-derivable: the Vicuna tokenizer, the Vicuna chat
# wrapper (build_prompt_vicuna), GENERATION_RESERVE = 320, and this — the prefix width the
# budget was computed against. NUM_PREFIX_TOKENS moved to 32 for Qwen3's GQA, and recomputing
# with it drops a reference on some nodes (train node 32196: 14 -> 13). That does not make the
# frozen lists wrong; it makes them *historical*, which is the point.
#
# The lists were therefore sized for a narrower prefix than the run now uses. That is safe but
# not free, and it was checked rather than assumed: a check over all 9,501
# nodes gives a largest prompt of 3,627 against a 3,744 budget at T = 32 under Qwen's
# tokenizer — 117 to spare, because the BPE saves more than the 22 extra prefix tokens cost.
FREEZE_PREFIX_TOKENS = 10

_frozen_refs = None


def frozen_refs():
    """-> {node_id: [ref_node_id, ...]}, the bibliography every model reads. Cached."""
    global _frozen_refs
    if _frozen_refs is None:
        with open(FROZEN_REFS) as f:
            _frozen_refs = {int(k): v for k, v in json.load(f).items()}
    return _frozen_refs


# --- The live prompt: soft-prompt-only (2026-09-16) ------------------------------------------
# No reference block. Everything the model learns about the target paper arrives through the
# graph prefix, so the text is identical for every node and the prefix is the only thing that
# differs between two papers.
#
# The system message is the era-4 instruction,
# the best of the four under Qwen3-14B (5.333 vs era 1's 3.875 on 48 nodes), with the phrases
# that pointed at a listed bibliography re-pointed at the context the prefix supplies.
#
# Everything below this block — SYSTEM, INSTRUCTION, build_prompt, capped_example — is the old
# text path. train.py and generate.py no longer call it.
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

# DYTAG_DATASET=children: the same prompt, line for line, for a children's book whose
# description is reconstructed from the books it was bought with. No word count: descriptions
# run from 80 to over 700 words (paper abstracts sit near 170), so the length is left to
# what training shows.
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
# The comparison arm for the books. Unlike the papers' 09-10 text baseline, whose era-1 wording
# differed from the graph prompt, this is the
# graph prompt with "the context encodes" re-pointed at a listed block: the evidence is the only
# difference. Which neighbours are listed is frozen in BOOK_TEXT_REFS.
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
    """The one prompt every node gets, in Qwen3's ChatML with thinking disabled.

    A short user turn is kept rather than ending on the system message: a chat-tuned model
    answering with no user turn at all is off-distribution, and the frozen arm would pay for it.
    """
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
    """The experimental content: the instruction and the reference block.

    `neighbours` is a list of (title, abstract) for the papers the target cites. This is the
    part that must not change between the Vicuna and Qwen runs — only the chat delimiters
    around it do. Kept separate for exactly that reason.
    """
    sources = "\n\n".join(f"- Title: {t}\n  Abstract: {a}" for t, a in neighbours)
    return (f"{INSTRUCTION}\n\n"
            f"Source publications:\n{sources or '(no neighbouring sources available)'}\n\n"
            f"Generated abstract:")


def build_prompt(tokenizer, neighbours):
    """The user message wrapped in the model's own chat template.

    Built through `apply_chat_template` rather than hand-rolled, so the delimiters come from
    the tokenizer that will read them.

    `enable_thinking=False` is mandatory. Qwen3 is a hybrid thinking model, and a `<think>`
    block would poison both the training targets and the generations. Note it does not remove
    the block: the template emits an *empty, already closed* `<think>\n\n</think>\n\n`,
    which is how Qwen3 is told to answer directly.
    """
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_message(neighbours)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build_prompt_vicuna(neighbours):
    """The Vicuna wrapper, kept because the frozen reference lists were computed under it.

    Not used by training or generation any more. `capped_example(frozen=False)` builds with
    this, so data/frozen/prompt_refs_capped.json can still be re-derived
    byte for byte — the freeze is defined by the Vicuna tokenizer *and* the Vicuna format,
    and a script that cannot reproduce its own output is not much of a freeze.
    """
    return f"{SYSTEM} USER: {user_message(neighbours)} ASSISTANT:"


def encode(tokenizer, prompt, target):
    """-> (input_ids, labels). Labels are -100 over the prompt, so only the abstract trains.

    Qwen3 has no BOS (`bos_token is None`) and its chat template does not prepend one, so
    `add_special_tokens` adds nothing here — unlike Vicuna, where this call carried BOS.
    The appended `eos_token_id` is `<|im_end|>` (151645), which is what ends an assistant
    turn in ChatML, so the target is terminated the way the model expects.
    """
    prompt_ids = tokenizer(prompt).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids


def fits(input_ids):
    """The prefix occupies real attention positions, so it counts against the context."""
    return len(input_ids) + NUM_PREFIX_TOKENS <= MAX_LENGTH


# --- Run 8: target cleaning ------------------------------------------------------------------
# Audited over the 7,750 capped training abstracts: inline maths in 13.5%, LaTeX commands in
# 7.0%, super/subscripts in 6.2%, whitespace runs in 2.1%, escaped characters in 1.2%. No HTML,
# no entities, no non-Latin text. So the corpus is clean apart from LaTeX and maths, which touch
# roughly a fifth of it, and those documents score measurably worse on the eval set
# (bertscore_f1 -0.0110, llm_accuracy -0.0839 across the 93 of 945 golds that carry markup).
#
# The stronger reason is a contradiction rather than a correlation: GRAPH_ONLY_SYSTEM instructs
# "no markup, mathematical notation or headings", and 13.5% of the supervision contains exactly
# that. The instruction and the targets have been disagreeing on a fifth of the training data.
#
# The delimiters go, the content stays: "$O(n^2)$" becomes "O(n^2)", not nothing. Deleting the
# maths would delete the method, which is the thing the run exists to recover.
#
# Declared limitation: the voyage embeddings were computed on the RAW text and re-embedding
# 384,756 documents is not on the table, so cleaning the targets introduces a mismatch between
# the feature and the target. It is mild — the features are an input representation, not a
# label — and it is stated rather than hidden. For the same reason generate.py builds its
# Corpus with clean=False, so the evaluation golds stay byte-identical to all eight earlier
# runs and the scores remain comparable.
_UNWRAP = re.compile(r"\\(?:textit|textbf|emph|texttt|textsc|textrm|mathrm|mathbf|mathit|text)"
                     r"\s*\{([^{}]*)\}")
# The old switch form, "{\em text}" rather than "\emph{text}", which is common in abstracts
# written before the 2000s. Without it the braces survive into the target as "{em text}".
_UNWRAP_SWITCH = re.compile(r"\{\\(?:em|it|bf|tt|sl|sc|rm)\s+([^{}]*)\}")
_MATH_DELIM = re.compile(r"(?<!\\)\$+")
# The last four are the \( \) \[ \] inline- and display-maths delimiters, which are the same
# class of thing as $ and go the same way: the delimiter is removed, the maths inside stays.
_ESCAPES = {r"\%": "%", r"\&": "&", r"\_": "_", r"\$": "$", r"\#": "#", r"\\": " ",
            r"\ ": " ", r"\(": " ", r"\)": " ", r"\[": " ", r"\]": " "}
# Accent escapes: "Verd\'u" -> "Verdu", "na\"ive" -> "naive". The mark goes rather than being
# kept as punctuation, which would leave a stray quote in the middle of a word.
_ACCENT = re.compile(r"""\\['"`^~=.]""")
_BRACES = re.compile(r"[{}]")
_COMMAND = re.compile(r"\\([a-zA-Z]+)\s*")
_WHITESPACE = re.compile(r"\s+")


def clean_abstract(text):
    r"""Strip LaTeX markup and maths delimiters, keeping their contents. -> str

    The order of the six steps is load-bearing and each one is here because the previous one
    would otherwise do damage:

      1. unwrap the known text-styling commands, innermost first, in both the modern form
         ("\emph{x}") and the old switch form ("{\em x}");
      2. strip unescaped maths delimiters, BEFORE step 3 — otherwise a price written "US\$49"
         becomes "US$49" and then loses its dollar sign along with the real delimiters;
      3. resolve the remaining escapes to their literals;
      4. un-backslash any control word that is left, keeping its NAME. Deleting it instead was
         tried and is wrong: "$\Omega(N \log\log N)$" becomes "(N N)" under deletion and
         "Omega (N log log N)" under this — the stray space before the bracket is accepted,
         because every rule that removes it also glues "\log\log" into "loglog";
      5. drop accent escapes and the grouping braces the commands leave behind
         ("\tilde{O}" -> "tilde O");
      6. collapse whitespace.

    There is deliberately NO rule turning a literal "\n" or "\t" two-character sequence into
    whitespace. It was in the first version and it was wrong: across 60,000 abstracts this
    corpus has **10** bare "\n"/"\r" escapes and **5,172** LaTeX commands beginning with those
    letters, so the rule fired almost exclusively on "\times", "\tilde" and "\rho" — turning
    "a Z \times Z" into "a Z imes Z". Step 4 handles all of them.
    """
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
        """-> (prompt, target_abstract) for one node, with every reference shown.

        The legacy (uncapped) path. `tokenizer` is needed because the chat wrapper now comes
        from the model's own template.
        """
        refs = self.refs.get(node_id, [])
        return (build_prompt(tokenizer, [(self.title[n], self.abstract[n]) for n in refs]),
                self.abstract[node_id])

    def out_degree(self, node_id):
        return len(self.refs.get(node_id, []))


def capped_example(tokenizer, corpus, embeddings, node_id, frozen=True):
    """-> (prompt, target, kept_refs) for one node, trimmed to fit the context.

    The single capped-prompt path: train.py builds training prompts with it, and generate.py builds eval prompts with
    it. They must agree exactly — a node was selected on the assumption that these
    references are the ones shown, and the failure mode of a mismatched budget is silent
    (a fluent abstract written from the wrong sources, not an error).

    `frozen=True` reads the reference list from FROZEN_REFS and raises KeyError if the node
    is not in it. It deliberately does not fall back to recomputing: a node recomputed under
    a different tokenizer would be shown a different bibliography and would still produce a
    fluent abstract. `tokenizer` and `embeddings` go unused on this path, and stay in the
    signature because every caller has both and the two paths must remain interchangeable.

    `frozen=False` is the original computation, and is how FROZEN_REFS itself is built.
    The budget leaves room for the prefix and for the abstract: the
    node's own abstract when it is longer than GENERATION_RESERVE, so the training sequence
    is complete, and never less than GENERATION_RESERVE, so generation has somewhere to write.

    The two branches deliberately build **different prompt formats**. `frozen=True` is the
    live path and uses the current model's chat template; `frozen=False` exists only to
    re-derive the freeze, which was computed under the Vicuna tokenizer and the Vicuna
    wrapper, so it pins both. They agree on the reference list, which is the thing the
    freeze actually records — not on the string.
    """
    target = corpus.abstract[node_id]

    if frozen:
        refs = frozen_refs()[node_id]
    else:
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        reserve = max(GENERATION_RESERVE, len(target_ids) + 1)   # +1 for the EOS token
        # FREEZE_PREFIX_TOKENS, not NUM_PREFIX_TOKENS: this branch re-derives a historical
        # artifact and must use the width it was built under. See FREEZE_PREFIX_TOKENS above.
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

    # The eval set is frozen and is not filtered — Stage 5 has to handle its overflow.
    over = sum(not fits(encode(tokenizer, *corpus.example(tokenizer, n))[0]) for n in eval_nodes)
    print(f"\neval set: {over}/{len(eval_nodes)} nodes overflow the context "
          f"({over / len(eval_nodes):.1%}) — left in place, Stage 5's problem")

    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
