"""Does the judge actually give 10/10 when the two abstracts are identical?

Every rubric in this folder scores a generated abstract against the original. That only means
something if the scale is anchored: a perfect reproduction has to score at the top. This is
the cheapest possible check — hand the judge the *same* abstract twice, as both ORIGINAL and
GENERATED, and see whether it returns 10s.

If it does not, every score in the folder is being read against a ceiling that is lower than
10 and not known, and gaps between arms are compressed by whatever that gap is.

10 nodes, R1, `gpt-5.4-mini`, `temperature=0.0` — identical to `score.py` in every respect
except that `generated_abstract` is replaced by `original_abstract`.

Usage:
    conda activate DyTAG
    python experiments/rubric_sensitivity/identity_ceiling.py

Writes `identity_ceiling.csv` and prints the summary.
"""

import csv
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from rubrics import RUBRICS                      # noqa: E402
from runs import generations                     # noqa: E402

load_dotenv(ROOT / ".env_local")

N = 10
SEED = 0
MODEL = "gpt-5.4-mini"


def main():
    source = pd.read_csv(generations("baseline")).set_index("node_id")
    nodes = random.Random(SEED).sample(sorted(source.index), N)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rows = []
    for i, node_id in enumerate(nodes, start=1):
        abstract = source.loc[node_id, "original_abstract"]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": RUBRICS["R1"]},
                {"role": "user",
                 "content": f"ORIGINAL ABSTRACT:\n{abstract}\n\nGENERATED ABSTRACT:\n{abstract}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content)
        rows.append({"node_id": node_id, **{k: result.get(k) for k in
                                            ("accuracy", "coherence", "language", "reason")}})
        print(f"[{i}/{N}] node {node_id}: "
              f"accuracy={result.get('accuracy')} coherence={result.get('coherence')} "
              f"language={result.get('language')}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "identity_ceiling.csv", index=False, quoting=csv.QUOTE_ALL)

    print()
    for metric in ("accuracy", "coherence", "language"):
        col = frame[metric]
        print(f"{metric:9s} mean {col.mean():.2f}  min {col.min()}  max {col.max()}  "
              f"perfect 10s: {(col == 10).sum()}/{N}")


if __name__ == "__main__":
    main()
