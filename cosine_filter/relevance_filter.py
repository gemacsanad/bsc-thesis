import os
import time
import pandas as pd
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'ogbn_arxiv_dataset')
EDGES_PATH = os.path.join(DATA_DIR, 'edges_subset.csv')
NODES_PATH = os.path.join(DATA_DIR, 'nodes_subset.csv')
MODEL = "gpt-5.4-mini"
DELAY = 0.1  # seconds between API calls


def build_node_lookup(nodes_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in nodes_df.iterrows():
        lookup[int(row['node_id'])] = {
            'title': row['title'],
            'abstract': row['abstract'],
        }
    return lookup


SYSTEM_PROMPT = (
    "You are a scientific relevance classifier. "
    "Given a source paper and a reference paper, decide whether the reference is "
    "DIRECTLY relevant to the source paper — meaning it contributes meaningfully to "
    "the source paper's topic, methodology, or core contributions. "
    "Respond with exactly one word: True or False."
)


def is_relevant(client: OpenAI, src: dict, dst: dict) -> bool:
    user_msg = (
        f"SOURCE PAPER\nTitle: {src['title']}\nAbstract: {src['abstract']}\n\n"
        f"REFERENCE PAPER\nTitle: {dst['title']}\nAbstract: {dst['abstract']}\n\n"
        "Is the reference paper directly relevant to the source paper? "
        "Reply with True or False only."
    )
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=5,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().lower()
            return answer.startswith("true")
        except RateLimitError:
            wait = 2 ** attempt
            print(f"  Rate limit — retrying in {wait}s …", flush=True)
            time.sleep(wait)
    raise RuntimeError("max retries exceeded")


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    edges_df = pd.read_csv(EDGES_PATH)
    nodes_df = pd.read_csv(NODES_PATH)
    node_lookup = build_node_lookup(nodes_df)

    if 'relevant' not in edges_df.columns:
        edges_df['relevant'] = None

    total = len(edges_df)
    for i, (idx, row) in enumerate(edges_df.iterrows()):
        if pd.notna(edges_df.at[idx, 'relevant']):
            continue

        src_id = int(row['src_node_id'])
        dst_id = int(row['dst_node_id'])

        src = node_lookup.get(src_id)
        dst = node_lookup.get(dst_id)

        if src is None or dst is None:
            edges_df.at[idx, 'relevant'] = False
            print(f"[{i+1}/{total}] edge ({src_id} -> {dst_id}): missing node data, marked False")
            continue

        try:
            result = is_relevant(client, src, dst)
            edges_df.at[idx, 'relevant'] = result
            print(f"[{i+1}/{total}] edge ({src_id} -> {dst_id}): {result}")
        except Exception as e:
            print(f"[{i+1}/{total}] edge ({src_id} -> {dst_id}): ERROR - {e}")
            edges_df.at[idx, 'relevant'] = None

        if (i + 1) % 100 == 0:
            edges_df.to_csv(EDGES_PATH, index=False)
            print(f"  -> checkpoint saved at {i+1} rows")

        time.sleep(DELAY)

    edges_df['relevant'] = edges_df['relevant'].astype('boolean')
    edges_df.to_csv(EDGES_PATH, index=False)
    print(f"\nDone. Results saved to {EDGES_PATH}")
    print(edges_df['relevant'].value_counts())


if __name__ == '__main__':
    main()
