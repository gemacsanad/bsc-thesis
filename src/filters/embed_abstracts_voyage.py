"""
Generate abstract embeddings for nodes_subset.csv using Voyage AI voyage-4.

Saves node_subset_features.npy with shape (N, D) where N = number of nodes
in nodes_subset.csv and D = embedding dimension of voyage-4.
Row order matches the row order of nodes_subset.csv.

Requirements:
    pip install voyageai python-dotenv pandas numpy

Usage:
    python src/filters/embed_abstracts_voyage.py
"""

import os
import time
import numpy as np
import pandas as pd
import voyageai
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'ogbn_subset')
NODES_PATH = os.path.join(DATA_DIR, 'nodes_subset.csv')
OUT_PATH = os.path.join(DATA_DIR, 'node_subset_features.npy')

MODEL = "voyage-4"
BATCH_SIZE = 128  # Voyage AI max texts per request
INPUT_TYPE = "document"


def batch_embed(client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        for attempt in range(6):
            try:
                result = client.embed(batch, model=MODEL, input_type=INPUT_TYPE)
                all_embeddings.extend(result.embeddings)
                print(f"  batch {batch_num}/{total_batches} done ({len(batch)} texts)", flush=True)
                break
            except voyageai.error.RateLimitError:
                wait = 2 ** attempt
                print(f"  Rate limit — retrying in {wait}s …", flush=True)
                time.sleep(wait)
            except Exception as e:
                raise RuntimeError(f"Embedding failed on batch {batch_num}: {e}") from e
        else:
            raise RuntimeError(f"Max retries exceeded on batch {batch_num}")

    return all_embeddings


def main():
    nodes_df = pd.read_csv(NODES_PATH)
    print(f"Loaded {len(nodes_df)} nodes from {NODES_PATH}")

    abstracts = nodes_df['abstract'].fillna('').astype(str).tolist()

    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    print(f"Embedding {len(abstracts)} abstracts with {MODEL} …")
    embeddings = batch_embed(client, abstracts)

    matrix = np.array(embeddings, dtype=np.float32)
    print(f"Embedding matrix shape: {matrix.shape}")

    np.save(OUT_PATH, matrix)
    print(f"Saved → {OUT_PATH}")


if __name__ == '__main__':
    main()
