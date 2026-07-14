"""
Shared helpers for talking to the Semantic Scholar Datasets API
(https://api.semanticscholar.org/datasets/v1).

Rate limiting
-------------
The API key is limited to 1 request/second, CUMULATIVE ACROSS ALL
ENDPOINTS (Graph, Recommendations, Datasets) - not per endpoint. Every
metadata call in this module goes through `get_with_retry`, which
throttles globally and retries on 429 with exponential backoff, honoring
`Retry-After` when present.

The actual shard files (returned as `files` URLs) are pre-signed
CloudFront/S3 links. They are NOT subject to the API key or its rate
limit, so downloading them is not throttled here.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("S2_API_KEY")
if not API_KEY:
    raise ValueError("Set S2_API_KEY in the environment or a .env file first")

HEADERS = {"x-api-key": API_KEY}
BASE_URL = "https://api.semanticscholar.org/datasets/v1"

MIN_REQUEST_INTERVAL = 1.1  # seconds; a bit over 1/s to stay safely under the limit
_last_request_time = 0.0


def _throttle():
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


def get_with_retry(url, max_retries=8, initial_delay=8, **kwargs):
    """GET `url` with global throttling + exponential backoff on 429."""
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        _throttle()
        r = requests.get(url, headers=HEADERS, **kwargs)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        retry_after = r.headers.get("Retry-After")
        wait = float(retry_after) if retry_after else delay
        print(f"    429 Too Many Requests (attempt {attempt}/{max_retries}). "
              f"Waiting {wait:.0f}s before retrying...")
        time.sleep(wait)
        delay *= 1.6

    # Final attempt: let it raise if still failing.
    _throttle()
    r = requests.get(url, headers=HEADERS, **kwargs)
    r.raise_for_status()
    return r


def get_latest_release():
    r = get_with_retry(f"{BASE_URL}/release/latest")
    return r.json()["release_id"]


def get_dataset_files(release_id, dataset_name):
    """Return the list of pre-signed shard URLs for a dataset."""
    r = get_with_retry(f"{BASE_URL}/release/{release_id}/dataset/{dataset_name}")
    return r.json()["files"]


def download_file_resumable(url, dest_path, chunk_size=1 << 20, max_retries=5):
    """
    Stream `url` to `dest_path`, resuming from a partial `.part` file via
    HTTP Range if a previous attempt was interrupted. Skips entirely if
    `dest_path` already exists and matches the remote Content-Length.
    Shard URLs are pre-signed and not rate-limited, so this is not
    throttled through `get_with_retry`.
    """
    tmp_path = dest_path + ".part"

    head = requests.head(url, timeout=30)
    head.raise_for_status()
    remote_size = int(head.headers.get("Content-Length", 0))

    if os.path.exists(dest_path) and (
        remote_size == 0 or os.path.getsize(dest_path) == remote_size
    ):
        print(f"    {os.path.basename(dest_path)}: already downloaded, skipping")
        return

    resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    if remote_size and resume_from == remote_size:
        os.rename(tmp_path, dest_path)
        return

    for attempt in range(1, max_retries + 1):
        try:
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            mode = "ab" if resume_from else "wb"
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            resume_from += len(chunk)
            break
        except requests.exceptions.RequestException as e:
            print(f"    retry {attempt}/{max_retries} after error: {e}")
            time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")

    os.rename(tmp_path, dest_path)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"    {os.path.basename(dest_path)}: {size_mb:.1f} MB")
