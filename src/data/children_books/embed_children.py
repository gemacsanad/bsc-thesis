"""
Embed every book with voyage-4-large, exactly as the papers were embedded
(notebooks/adapter/embed_nodes.ipynb): `title + "\\n\\n" + description`, input_type
"document", 128 texts per request.

Output: data/children_books/node_embeddings.npy, float32 (N, 1024), row i == node_id i.
Rows are written straight into a memory-mapped .npy and node_embeddings.progress.json
records how many are finished, so re-running resumes instead of starting over. A batch that
keeps failing after all retries raises, with everything done so far already on disk.

Needs VOYAGE_API_KEY in the repository's .env.

Usage:
    python embed_children.py
    python prepare_sets.py        # afterwards: applies the cosine > 0.95 duplicate filter
"""

import json
import os
import time

import numpy as np
import pandas as pd
import voyageai
from dotenv import load_dotenv

from build_children import DATA_DIR, ROOT

MODEL = "voyage-4-large"
INPUT_TYPE = "document"
BATCH_SIZE = 128          # Voyage max texts per request
MAX_RETRIES = 6

OUT_PATH = os.path.join(DATA_DIR, "node_embeddings.npy")
PROG_PATH = os.path.join(DATA_DIR, "node_embeddings.progress.json")


def embed_batch(client, batch):
    """Embed one batch, retrying with exponential backoff. Raises if it never succeeds."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.embed(batch, model=MODEL, input_type=INPUT_TYPE).embeddings
        except Exception as e:
            wait = 2 ** attempt
            print(f"    {type(e).__name__}: {e} — retrying in {wait}s "
                  f"({attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"batch failed after {MAX_RETRIES} attempts")


def main():
    load_dotenv(os.path.join(ROOT, ".env"))
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    nodes = pd.read_csv(os.path.join(DATA_DIR, "nodes.csv"),
                        usecols=["node_id", "title", "abstract"], keep_default_na=False)
    assert (nodes["node_id"].to_numpy() == np.arange(len(nodes))).all(), \
        "node_id is not 0..N-1 in row order"
    texts = (nodes["title"] + "\n\n" + nodes["abstract"]).tolist()
    n = len(texts)

    if os.path.exists(PROG_PATH):
        with open(PROG_PATH) as f:
            prog = json.load(f)
        assert prog["model"] == MODEL, f"existing run used {prog['model']}, not {MODEL}"
        rows_done = prog["rows_done"]
        emb = np.lib.format.open_memmap(OUT_PATH, mode="r+")
        assert emb.shape[0] == n, "node_embeddings.npy does not match nodes.csv"
        print(f"resuming: {rows_done:,}/{n:,} rows already embedded")
    else:
        rows_done, emb = 0, None
        print(f"starting fresh: {n:,} books to embed")

    session_start, t0 = rows_done, time.time()
    for start in range(rows_done, n, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        vectors = np.asarray(embed_batch(client, batch), dtype=np.float32)
        if emb is None:   # the dimension is only known after the first response
            emb = np.lib.format.open_memmap(OUT_PATH, mode="w+", dtype=np.float32,
                                            shape=(n, vectors.shape[1]))
        emb[start:start + len(batch)] = vectors
        emb.flush()
        rows_done = start + len(batch)
        with open(PROG_PATH, "w") as f:
            json.dump({"rows_done": rows_done, "model": MODEL}, f)

        if (start // BATCH_SIZE + 1) % 10 == 0 or rows_done == n:
            elapsed = time.time() - t0
            rate = (rows_done - session_start) / max(elapsed, 1e-9)
            print(f"  {rows_done:,}/{n:,} ({rows_done / n:.1%})  elapsed {elapsed / 60:.1f} min  "
                  f"ETA {(n - rows_done) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    saved = np.load(OUT_PATH, mmap_mode="r")
    norms = np.linalg.norm(saved, axis=1)
    print(f"done: {OUT_PATH}  shape {saved.shape}  "
          f"zero rows {int((norms == 0).sum())}  norm range {norms.min():.4f}-{norms.max():.4f}")


if __name__ == "__main__":
    main()
