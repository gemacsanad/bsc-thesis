"""
Download all shards of one or more Semantic Scholar bulk datasets to disk.

Only the two metadata calls per dataset (release lookup is shared, then
one shard-listing call) go through the rate-limited Datasets API. The
shard files themselves are pre-signed URLs and are downloaded directly,
with byte-level resume support so an interrupted run can just be
re-executed.

Shard URLs expire (pre-signed S3 URLs, commonly after roughly an hour).
For datasets with many/large shards, downloading them one after another
can easily exceed that window, so a fresh URL list is fetched right
before each shard (not once for the whole dataset up front), and any
403 (expired URL) encountered mid-download triggers another refetch.

We need three datasets to build a mag_id-keyed (paper -> abstract,
references) view without any full text:
  - papers     : per-paper metadata, incl. externalids.MAG -> used to
                 build the mag_id <-> corpusid mapping.
  - abstracts  : corpusid -> abstract text.
  - citations  : citingcorpusid -> citedcorpusid edges (the reference graph).

Usage:
    python download_shards.py --out-dir /path/to/data/raw/s2
    python download_shards.py --out-dir /path/to/data/raw/s2 --datasets papers abstracts
"""

import argparse
import os
import time

from s2_common import get_latest_release, get_dataset_files, download_file_resumable

DEFAULT_DATASETS = ["papers", "abstracts", "citations"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True,
                     help="Output directory; shards are written to <out-dir>/<dataset>/")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                     help=f"Dataset names to download (default: {DEFAULT_DATASETS})")
    ap.add_argument("--release", default=None,
                     help="Release id to use; defaults to the latest release")
    args = ap.parse_args()

    release_id = args.release or get_latest_release()
    print(f"Using release: {release_id}", flush=True)

    for dataset in args.datasets:
        print(f"\n--- {dataset} ---", flush=True)
        files = get_dataset_files(release_id, dataset)
        n_files = len(files)
        raw_dir = os.path.join(args.out_dir, dataset)
        os.makedirs(raw_dir, exist_ok=True)
        print(f"  {n_files} shards -> {raw_dir}", flush=True)

        dataset_start = time.monotonic()
        for i in range(n_files):
            dest = os.path.join(raw_dir, f"{i:03d}.jsonl.gz")
            label = f"{dataset} {i + 1:03d}/{n_files:03d}"

            # Fetch a fresh URL right before downloading this shard rather
            # than reusing the list fetched at the top of the dataset loop,
            # since that list can be stale/expired by the time we get here.
            url = get_dataset_files(release_id, dataset)[i]
            refresher = lambda i=i: get_dataset_files(release_id, dataset)[i]  # noqa: E731

            download_file_resumable(url, dest, label=label, url_refresher=refresher)

            elapsed = time.monotonic() - dataset_start
            done = i + 1
            avg_per_shard = elapsed / done
            eta = avg_per_shard * (n_files - done)
            print(f"  {dataset}: {done}/{n_files} shards done "
                  f"(elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))}, "
                  f"ETA {time.strftime('%H:%M:%S', time.gmtime(eta))})", flush=True)

        print(f"  {dataset}: all {n_files} shards complete "
              f"in {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - dataset_start))}", flush=True)

    print("\nAll requested datasets downloaded.", flush=True)


if __name__ == "__main__":
    main()
