# Semantic Scholar full-dataset download

Downloads the full Semantic Scholar bulk datasets and joins them into two
consolidated, `mag_id`-keyed files (instead of 30 raw shard files per
dataset) - no full text is downloaded, only abstracts and the reference
(citation) graph.

## Why these three datasets

| Dataset      | Used for                                              |
|--------------|--------------------------------------------------------|
| `papers`     | `externalids.MAG` -> builds the `mag_id <-> corpusid` mapping |
| `abstracts`  | `corpusid -> abstract` text                             |
| `citations`  | `citingcorpusid -> citedcorpusid` edges (the reference graph) |

Semantic Scholar keys everything internally by `corpusid`; `mag_id` only
appears inside `papers.externalids.MAG`. So we always need `papers` as the
mapping table even though we don't keep any of its other fields.

## Pipeline

1. `download_shards.py` - downloads every shard of `papers`, `abstracts`,
   and `citations` to `<out-dir>/raw/<dataset>/NNN.jsonl.gz`. Resumable:
   re-running skips shards already fully downloaded and resumes partial
   ones via HTTP `Range`. Only the release lookup + one shard-listing call
   per dataset go through the rate-limited API key (1 req/s, cumulative
   across all S2 endpoints); the shard files are pre-signed URLs and are
   not throttled.

2. `build_dataset.py` - uses DuckDB to read the raw `.jsonl.gz` shards
   directly (out-of-core, so it doesn't need to fit ~500GB in RAM) and
   produces:
   - `papers.parquet` - columns `mag_id, corpusid, abstract`
   - `references.parquet` - columns `citing_mag_id, cited_mag_id`

## Usage

```bash
pip install -r requirements.txt

# 1. Download all raw shards (resumable - safe to re-run if interrupted)
python download_shards.py --out-dir /path/to/s2_data

# 2. Join into the two final consolidated files
python build_dataset.py --raw-dir /path/to/s2_data/raw --out-dir /path/to/s2_data/processed
```

Requires a `.env` file (or exported env var) with `S2_API_KEY` - the same
one used by `test_bulk_download.py` / `fetch_paper_by_mag_id.py` at the
repo root.

## Notes / assumptions

- The raw `papers` dataset is large but only `corpusid` and
  `externalids.MAG` are ever parsed out of it (via DuckDB column
  projection) - it is never materialized in full.
- Papers without a MAG id are dropped from `papers.parquet`; citation
  edges where either endpoint lacks a MAG id are dropped from
  `references.parquet` (both are inner joins against `mag_map`), since the
  whole point of this pipeline is a `mag_id`-keyed view consistent with
  the rest of this repo (`nodes.csv` / `edges.csv` also use `mag_id`).
- `build_dataset.py --tmp-dir` should point at a directory with a lot of
  free space (the `citations` join is the largest step); on the HPC this
  should be somewhere under `/project/...`, not the home directory.
