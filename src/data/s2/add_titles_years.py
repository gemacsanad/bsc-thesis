"""Add the title and year of every paper to papers_filtered.parquet, from the raw
Semantic Scholar 'papers' shards.

Usage:
    python add_titles_years.py --raw-dir <raw/s2> --papers-filtered <processed/s2/papers_filtered.parquet>
"""

import argparse
import os
import duckdb

SHARD_GLOB_PATTERN = "*.jsonl.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True,
                     help="Directory containing raw/papers produced by download_shards.py "
                          "(i.e. pass <out-dir>/raw from that script)")
    ap.add_argument("--papers-filtered", required=True,
                     help="Path to papers_filtered.parquet (from filter_ogbn_references.py); "
                          "overwritten in place with title/year columns added")
    ap.add_argument("--tmp-dir", default=None,
                     help="Directory for DuckDB's disk spill (should have plenty of free space); "
                          "defaults to the parent directory of --papers-filtered + /duckdb_tmp")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory-limit", default="32GB")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.papers_filtered))
    tmp_dir = args.tmp_dir or os.path.join(out_dir, "duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")

    papers_glob = os.path.join(args.raw_dir, "papers", SHARD_GLOB_PATTERN)

    print("[1/3] Reading title/year from raw 'papers' shards...")
    con.execute(f"""
        CREATE OR REPLACE TABLE mag_title_map AS
        SELECT
            TRY_CAST(externalids.MAG AS BIGINT) AS mag_id,
            ANY_VALUE(title) AS title,
            ANY_VALUE(year) AS year
        FROM read_ndjson(
            '{papers_glob}',
            columns={{'corpusid': 'BIGINT', 'externalids': 'STRUCT(MAG VARCHAR)',
                      'title': 'VARCHAR', 'year': 'BIGINT'}}
        )
        WHERE externalids.MAG IS NOT NULL
        GROUP BY mag_id
    """)
    n_map = con.execute("SELECT COUNT(*) FROM mag_title_map").fetchone()[0]
    print(f"      {n_map:,} distinct mag_ids with title/year info")

    print("[2/3] Joining onto papers_filtered.parquet...")
    con.execute(f"""
        CREATE OR REPLACE TABLE papers_with_titles AS
        SELECT
            p.mag_id AS mag_id,
            p.corpusid AS corpusid,
            p.abstract AS abstract,
            t.title AS title,
            t.year AS year
        FROM read_parquet('{args.papers_filtered}') p
        LEFT JOIN mag_title_map t USING (mag_id)
    """)
    n_total = con.execute("SELECT COUNT(*) FROM papers_with_titles").fetchone()[0]
    n_matched = con.execute("SELECT COUNT(*) FROM papers_with_titles WHERE title IS NOT NULL").fetchone()[0]
    print(f"      {n_matched:,}/{n_total:,} rows matched a title (rest left as NULL)")

    print("[3/3] Writing updated papers_filtered.parquet...")
    tmp_out = args.papers_filtered + ".tmp"
    con.execute(f"COPY papers_with_titles TO '{tmp_out}' (FORMAT PARQUET)")
    os.replace(tmp_out, args.papers_filtered)
    print(f"      wrote {args.papers_filtered} ({n_total:,} rows, columns: "
          f"mag_id, corpusid, abstract, title, year)")


if __name__ == "__main__":
    main()
