"""Keep the ogbn-arxiv papers and every paper they cite, and the references between them.

Usage:
    python filter_ogbn_references.py --nodes-csv <ogbn_arxiv/nodes.csv> \
        --papers <processed/s2/papers.parquet> --references <processed/s2/references.parquet> \
        --out-dir <processed/s2>
"""

import argparse
import os
import duckdb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes-csv", required=True,
                     help="Path to data/raw/ogbn_arxiv/nodes.csv (needs a 'mag_id' column)")
    ap.add_argument("--papers", required=True,
                     help="Path to papers.parquet (from build_dataset.py)")
    ap.add_argument("--references", required=True,
                     help="Path to references.parquet (from build_dataset.py)")
    ap.add_argument("--out-dir", required=True,
                     help="Where to write papers_filtered.parquet / references_subset.parquet")
    ap.add_argument("--tmp-dir", default=None,
                     help="Directory for DuckDB's disk spill; defaults to <out-dir>/duckdb_tmp")
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

    con.execute(f"CREATE OR REPLACE VIEW papers AS SELECT * FROM read_parquet('{args.papers}')")
    con.execute(f"CREATE OR REPLACE VIEW references_tbl AS SELECT * FROM read_parquet('{args.references}')")

    print("[1/5] Loading ogbn-arxiv mag_ids from nodes.csv...")
    con.execute(f"""
        CREATE OR REPLACE TABLE ogbn_ids AS
        SELECT DISTINCT TRY_CAST(mag_id AS BIGINT) AS mag_id
        FROM read_csv_auto('{args.nodes_csv}')
        WHERE mag_id IS NOT NULL
    """)
    n_ogbn = con.execute("SELECT COUNT(*) FROM ogbn_ids").fetchone()[0]
    n_ogbn_in_papers = con.execute(
        "SELECT COUNT(*) FROM ogbn_ids WHERE mag_id IN (SELECT mag_id FROM papers)"
    ).fetchone()[0]
    print(f"      {n_ogbn:,} distinct ogbn-arxiv mag_ids "
          f"({n_ogbn_in_papers:,} of them found in papers.parquet)")

    print("[2/5] Finding all references made by ogbn-arxiv nodes...")
    con.execute("""
        CREATE OR REPLACE TABLE new_ref_ids AS
        SELECT DISTINCT r.cited_mag_id AS mag_id
        FROM references_tbl r
        JOIN ogbn_ids o ON r.citing_mag_id = o.mag_id
    """)
    n_new_refs = con.execute("SELECT COUNT(*) FROM new_ref_ids").fetchone()[0]
    n_new_refs_not_in_ogbn = con.execute(
        "SELECT COUNT(*) FROM new_ref_ids WHERE mag_id NOT IN (SELECT mag_id FROM ogbn_ids)"
    ).fetchone()[0]
    print(f"      {n_new_refs:,} distinct referenced mag_ids "
          f"({n_new_refs_not_in_ogbn:,} not already in the ogbn-arxiv node set)")

    print("[3/5] Building combined node set (ogbn-arxiv nodes + their references)...")
    con.execute("""
        CREATE OR REPLACE TABLE combined_ids AS
        SELECT mag_id FROM ogbn_ids
        UNION
        SELECT mag_id FROM new_ref_ids
    """)
    n_combined = con.execute("SELECT COUNT(*) FROM combined_ids").fetchone()[0]
    print(f"      {n_combined:,} distinct mag_ids in the combined set")

    print("[4/5] Writing papers_filtered.parquet...")
    con.execute("""
        CREATE OR REPLACE TABLE papers_filtered AS
        SELECT p.mag_id, p.corpusid, p.abstract
        FROM papers p
        WHERE p.mag_id IN (SELECT mag_id FROM combined_ids)
    """)
    papers_out = os.path.join(args.out_dir, "papers_filtered.parquet")
    con.execute(f"COPY papers_filtered TO '{papers_out}' (FORMAT PARQUET)")
    n_papers_filtered = con.execute("SELECT COUNT(*) FROM papers_filtered").fetchone()[0]
    print(f"      wrote {papers_out} ({n_papers_filtered:,} rows)")

    print("[5/5] Writing references_subset.parquet (all edges between papers_filtered nodes)...")
    con.execute("""
        CREATE OR REPLACE TABLE references_subset AS
        SELECT DISTINCT citing_mag_id, cited_mag_id
        FROM references_tbl
        WHERE citing_mag_id IN (SELECT mag_id FROM papers_filtered)
          AND cited_mag_id IN (SELECT mag_id FROM papers_filtered)
    """)
    refs_out = os.path.join(args.out_dir, "references_subset.parquet")
    con.execute(f"COPY references_subset TO '{refs_out}' (FORMAT PARQUET)")
    n_refs_subset = con.execute("SELECT COUNT(*) FROM references_subset").fetchone()[0]
    print(f"      wrote {refs_out} ({n_refs_subset:,} rows)")

    print("\nDone. Final outputs:")
    print(f"  {papers_out}")
    print(f"  {refs_out}")


if __name__ == "__main__":
    main()
