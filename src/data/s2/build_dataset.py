"""Join the raw Semantic Scholar shards into papers.parquet (mag_id, corpusid, abstract) and
references.parquet (citing_mag_id, cited_mag_id), out-of-core with DuckDB.

Usage:
    python build_dataset.py --raw-dir <raw/s2> --out-dir <processed/s2>
"""

import argparse
import os
import duckdb

SHARD_GLOB_PATTERN = "*.jsonl.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True,
                     help="Directory containing raw/{papers,abstracts,citations} produced by download_shards.py "
                          "(i.e. pass <out-dir>/raw from that script)")
    ap.add_argument("--out-dir", required=True, help="Where to write papers.parquet / references.parquet")
    ap.add_argument("--tmp-dir", default=None,
                     help="Directory for DuckDB's disk spill (should have plenty of free space); "
                          "defaults to <out-dir>/duckdb_tmp")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory-limit", default="32GB")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(args.out_dir, "duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")

    papers_glob = os.path.join(args.raw_dir, "papers", SHARD_GLOB_PATTERN)
    abstracts_glob = os.path.join(args.raw_dir, "abstracts", SHARD_GLOB_PATTERN)
    citations_glob = os.path.join(args.raw_dir, "citations", SHARD_GLOB_PATTERN)

    print("[1/3] Building mag_id <-> corpusid mapping from 'papers' shards...")
    con.execute(f"""
        CREATE OR REPLACE TABLE mag_map AS
        SELECT
            corpusid,
            TRY_CAST(externalids.MAG AS BIGINT) AS mag_id
        FROM read_ndjson(
            '{papers_glob}',
            columns={{'corpusid': 'BIGINT', 'externalids': 'STRUCT(MAG VARCHAR)'}}
        )
        WHERE externalids.MAG IS NOT NULL
    """)
    n_map = con.execute("SELECT COUNT(*) FROM mag_map").fetchone()[0]
    print(f"      {n_map:,} papers have a MAG id")

    print("[2/3] Joining 'abstracts' shards with mag_id -> papers.parquet ...")
    con.execute(f"""
        CREATE OR REPLACE TABLE papers AS
        SELECT
            m.mag_id AS mag_id,
            a.corpusid AS corpusid,
            a.abstract AS abstract
        FROM read_ndjson(
            '{abstracts_glob}',
            columns={{'corpusid': 'BIGINT', 'abstract': 'VARCHAR'}}
        ) a
        JOIN mag_map m USING (corpusid)
        WHERE a.abstract IS NOT NULL
    """)
    papers_out = os.path.join(args.out_dir, "papers.parquet")
    con.execute(f"COPY papers TO '{papers_out}' (FORMAT PARQUET)")
    n_papers = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"      wrote {papers_out} ({n_papers:,} rows)")

    print("[3/3] Joining 'citations' shards with mag_id (both endpoints) -> references.parquet ...")
    con.execute(f"""
        CREATE OR REPLACE TABLE references_tbl AS
        SELECT DISTINCT
            mc.mag_id AS citing_mag_id,
            mr.mag_id AS cited_mag_id
        FROM read_ndjson(
            '{citations_glob}',
            columns={{'citingcorpusid': 'BIGINT', 'citedcorpusid': 'BIGINT'}}
        ) c
        JOIN mag_map mc ON mc.corpusid = c.citingcorpusid
        JOIN mag_map mr ON mr.corpusid = c.citedcorpusid
    """)
    refs_out = os.path.join(args.out_dir, "references.parquet")
    con.execute(f"COPY references_tbl TO '{refs_out}' (FORMAT PARQUET)")
    n_refs = con.execute("SELECT COUNT(*) FROM references_tbl").fetchone()[0]
    print(f"      wrote {refs_out} ({n_refs:,} rows)")

    print("\nDone. Final outputs:")
    print(f"  {papers_out}")
    print(f"  {refs_out}")


if __name__ == "__main__":
    main()
