"""
Turn the CS-TAG Books-Children CSV (data/children_books/Children.csv) into the same
nodes.csv / edges.csv pair the citation pipeline reads (data/processed/joint/), so the
adapter code can load the books by changing only the paths.

Input columns: category, text ("Description: <blurb>; Title: <title>"), label, node_id,
neighbour (a Python list of co-purchased node_ids).

Output
------
nodes.csv   node_id, cstag_id, title, abstract, category, label, words, seller_note,
            shared_description
            Books with an empty description are dropped (nothing to generate or compare
            against), and the rest renumbered 0..N-1 so row i of the embedding matrix is
            node_id i; `cstag_id` is the original node_id.
            `abstract` holds the book description, so text.Corpus reads it unchanged.
            `words` is the description's word count. `seller_note` flags descriptions that
            describe a copy's condition ("Good condition. Ships same day.") rather than the
            book; nearly all are under 80 words, so the length filter drops them anyway.
            `shared_description` flags a description identical (after normalising case and
            punctuation) to another book's: mostly publisher boilerplate ("DK was founded in
            London in 1974...", 257 books). Such a text describes the publisher, not the
            book, and its copies sit among the neighbours, so these books must not be targets.
edges.csv   src_node_id, dst_node_id, both directions of every co-purchase pair.
            The CSV's neighbour lists are not symmetric; co-purchase is, so each node
            aggregates every book it was bought with. Self-loops, duplicates and edges to
            dropped books removed.

Cleaning: HTML entities are unescaped (18.5% of rows have them; there are no tags) and
whitespace is collapsed. Nothing else is changed: reviewer quotes and School Library
Journal grade tags ("PreS-Gr 1In this picture book...") are part of the blurbs as sold.

Usage:
    python build_children.py
"""

import argparse
import ast
import html
import os
import re

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(ROOT, "data", "children_books")

# Matched against the first 150 characters only: a seller note leads with the condition,
# while a blurb can mention "highlighting" or "ships" anywhere.
SELLER_NOTE = re.compile(
    r"(?i)(?:\b(?:very good|good|fair|acceptable|like new|excellent|used|great|new)\s*(?:\+\s*)?"
    r"condition\b|pages are (?:intact|clean|unmarked)|ex-?library|ships? (?:within|fast|quickly|"
    r"same|daily)|(?:signs of|minor|light|shelf) wear|remainder mark|(?:\bused|pre-owned|"
    r"second-?hand) (?:book|item|copy)|satisfaction guaranteed|may contain (?:notes|highlighting))")
_WHITESPACE = re.compile(r"\s+")


def clean(text):
    return _WHITESPACE.sub(" ", html.unescape(text)).strip()


def normalised(text):
    """Lower case, letters and digits only: the key for "the same text" and "the same title"."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join(DATA_DIR, "Children.csv"))
    ap.add_argument("--out-dir", default=DATA_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    assert (df["node_id"] == range(len(df))).all(), "node_id is not 0..N-1 in row order"
    assert (df["text"].str.count("; Title: ") == 1).all(), "a row without exactly one '; Title: '"

    desc, title = zip(*(t.split("; Title: ") for t in df["text"]))
    nodes = pd.DataFrame({
        "cstag_id": df["node_id"],
        "title": [clean(t) for t in title],
        "abstract": [clean(d.removeprefix("Description:")) for d in desc],
        "category": df["category"],
        "label": df["label"],
    })
    nodes["words"] = nodes["abstract"].str.split().str.len()
    empty = int((nodes["words"] == 0).sum())
    nodes = nodes[nodes["words"] > 0].reset_index(drop=True)
    nodes.insert(0, "node_id", nodes.index)
    nodes["seller_note"] = nodes["abstract"].str[:150].str.contains(SELLER_NOTE)
    nodes["shared_description"] = nodes["abstract"].map(normalised).duplicated(keep=False)

    new_id = dict(zip(nodes["cstag_id"], nodes["node_id"]))
    pairs = [(new_id[i], new_id[j]) for i, lst in zip(df["node_id"], df["neighbour"])
             for j in ast.literal_eval(lst) if i != j and i in new_id and j in new_id]
    one_way = pd.DataFrame(pairs, columns=["src_node_id", "dst_node_id"])
    edges = (pd.concat([one_way, one_way.rename(columns={"src_node_id": "dst_node_id",
                                                         "dst_node_id": "src_node_id"})])
             .drop_duplicates().sort_values(["src_node_id", "dst_node_id"]))
    assert edges.values.min() >= 0 and edges.values.max() < len(nodes)

    os.makedirs(args.out_dir, exist_ok=True)
    nodes.to_csv(os.path.join(args.out_dir, "nodes.csv"), index=False)
    edges.to_csv(os.path.join(args.out_dir, "edges.csv"), index=False)

    deg = edges.groupby("src_node_id").size().reindex(nodes["node_id"], fill_value=0)
    long_enough = (nodes["words"] >= 80) & ~nodes["seller_note"]
    usable = long_enough & (deg.values >= 3) & ~nodes["shared_description"]
    w = nodes["words"]
    print(f"nodes            {len(nodes):,}  ({empty} with an empty description dropped)")
    print(f"edges            {len(edges):,} directed rows = {len(edges) // 2:,} undirected pairs")
    print(f"degree           median {deg.median():.0f}, mean {deg.mean():.1f}, "
          f"isolated {(deg == 0).sum():,}")
    print(f"description      median {w.median():.0f} words, mean {w.mean():.1f}, "
          f"p90 {w.quantile(.9):.0f}; >= 80 words {(w >= 80).mean():.1%}")
    print(f"seller notes     {nodes['seller_note'].sum():,} "
          f"({(nodes['seller_note'] & (w >= 80)).sum()} of them >= 80 words)")
    print(f"shared descr.    {nodes['shared_description'].sum():,} books "
          f"({(nodes['shared_description'] & long_enough).sum():,} of them >= 80 words)")
    print(f"usable           {usable.sum():,}  "
          f"(>= 80 words, >= 3 neighbours, no seller note, description not shared)")


if __name__ == "__main__":
    main()
