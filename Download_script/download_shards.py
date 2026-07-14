"""
Download all shards of one or more Semantic Scholar bulk datasets to disk.

Only the two metadata calls per dataset (release lookup is shared, then
one shard-listing call) go through the rate-limited Datasets API. The
shard files themselves are pre-signed URLs and are downloaded directly,
with byte-level resume support so an interrupted run can just be
re-executed.

We need three datasets to build a mag_id-keyed (paper -> abstract,
references) view without any full text:
  - papers     : per-paper metadata, incl. externalids.MAG -> used to
                 build the mag_id <-> corpusid mapping.
  - abstracts  : corpusid -> abstract text.
  - citations  : citingcorpusid -> citedcorpusid edges (the reference graph).

Usage:
    python download_shards.py --out-dir /path/to/s2_data
    python download_shards.py --out-dir /path/to/s2_data --datasets papers abstracts
"""

import argparse
import os

from s2_common import get_latest_release, get_dataset_files, download_file_resumable

DEFAULT_DATASETS = ["papers", "abstracts", "citations"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True,
                     help="Root output directory; shards are written to <out-dir>/raw/<dataset>/")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                     help=f"Dataset names to download (default: {DEFAULT_DATASETS})")
    ap.add_argument("--release", default=None,
                     help="Release id to use; defaults to the latest release")
    args = ap.parse_args()

    release_id = args.release or get_latest_release()
    print(f"Using release: {release_id}")

    for dataset in args.datasets:
        print(f"\n--- {dataset} ---")
        files = get_dataset_files(release_id, dataset)
        raw_dir = os.path.join(args.out_dir, "raw", dataset)
        os.makedirs(raw_dir, exist_ok=True)
        print(f"  {len(files)} shards -> {raw_dir}")

        for i, url in enumerate(files):
            dest = os.path.join(raw_dir, f"{i:03d}.jsonl.gz")
            download_file_resumable(url, dest)

    print("\nAll requested datasets downloaded.")


if __name__ == "__main__":
    main()
