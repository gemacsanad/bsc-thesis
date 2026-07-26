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

Retries are intentionally very generous everywhere in this module (long
backoff caps, high retry counts): this is meant for an unattended HPC job
where waiting out a transient outage is preferable to the whole job
failing and having to be resubmitted.
"""

import os
import time
import requests
from dotenv import load_dotenv

# Load the .env file that lives next to this script (Download_script/.env),
# regardless of the caller's current working directory.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)

API_KEY = os.environ.get("S2_API_KEY")
if not API_KEY:
    raise ValueError(f"Set S2_API_KEY in {_ENV_PATH} first")

HEADERS = {"x-api-key": API_KEY}
BASE_URL = "https://api.semanticscholar.org/datasets/v1"

MIN_REQUEST_INTERVAL = 1.1  # seconds; a bit over 1/s to stay safely under the limit
MAX_BACKOFF_DELAY = 300  # seconds; cap so a single retry never waits more than this
_last_request_time = 0.0


def _throttle():
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


def _format_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def get_with_retry(url, max_retries=200, initial_delay=8, **kwargs):
    """GET `url` with global throttling + exponential backoff on 429.

    max_retries is intentionally very high with a capped delay
    (MAX_BACKOFF_DELAY) so this keeps retrying for a long time rather
    than giving up and killing the job.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        _throttle()
        r = requests.get(url, headers=HEADERS, **kwargs)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        retry_after = r.headers.get("Retry-After")
        wait = float(retry_after) if retry_after else delay
        wait = min(wait, MAX_BACKOFF_DELAY)
        print(f"    429 Too Many Requests (attempt {attempt}/{max_retries}). "
              f"Waiting {wait:.0f}s before retrying...", flush=True)
        time.sleep(wait)
        delay = min(delay * 1.6, MAX_BACKOFF_DELAY)

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


def _print_progress(tag, downloaded, total, elapsed, initial_bytes):
    rate = (downloaded - initial_bytes) / elapsed if elapsed > 0 else 0
    pct = (downloaded / total * 100) if total else 0
    bar_width = 20
    filled = int(bar_width * pct / 100) if total else 0
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"    [{tag}] [{bar}] {pct:5.1f}%  "
          f"{_format_bytes(downloaded)}/{_format_bytes(total)}  "
          f"{_format_bytes(rate)}/s", flush=True)


class _ExpiredUrlError(Exception):
    """Raised when a pre-signed shard URL responds with 403 (e.g. expired)."""


def _write_chunks(r, tmp_path, mode, resume_from, remote_size, tag, chunk_size,
                   progress_interval_s, progress_interval_bytes):
    """Write the streamed response body of `r` to `tmp_path`, printing periodic progress."""
    start_time = time.monotonic()
    last_report_time = start_time
    last_report_bytes = resume_from
    initial_bytes = resume_from

    with open(tmp_path, mode) as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            resume_from += len(chunk)
            now = time.monotonic()
            if (now - last_report_time >= progress_interval_s
                    or resume_from - last_report_bytes >= progress_interval_bytes):
                _print_progress(tag, resume_from, remote_size, now - start_time, initial_bytes)
                last_report_time = now
                last_report_bytes = resume_from
    return resume_from


def _attempt_download(url, tmp_path, resume_from, tag, chunk_size,
                       progress_interval_s, progress_interval_bytes):
    """
    Try to (continue) downloading `url` into `tmp_path`, starting at byte
    `resume_from`. Returns (new_resume_from, remote_size) once the response
    stream ends. Raises `_ExpiredUrlError` on a 403 response, or a
    `requests.exceptions.RequestException` on other request errors (timeouts,
    connection resets, etc).

    Deliberately never issues a HEAD request to check the remote size ahead
    of time: S3 pre-signed URLs are only signed for one specific HTTP
    method (typically GET), so a HEAD against a GET-signed URL fails with
    403 regardless of whether the URL has actually expired. The remote size
    is instead read off the real GET response's Content-Length /
    Content-Range headers.
    """
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 403:
            raise _ExpiredUrlError(f"403 Forbidden for {tag}")
        r.raise_for_status()

        if resume_from and r.status_code == 200:
            # Server ignored our Range request (e.g. doesn't support it) and
            # is sending the full object again - restart the file from scratch.
            resume_from = 0
            mode = "wb"
        else:
            mode = "ab" if resume_from else "wb"

        content_range = r.headers.get("Content-Range")
        if content_range and "/" in content_range:
            remote_size = int(content_range.rsplit("/", 1)[-1])
        else:
            remote_size = resume_from + int(r.headers.get("Content-Length", 0))

        resume_from = _write_chunks(r, tmp_path, mode, resume_from, remote_size, tag,
                                     chunk_size, progress_interval_s, progress_interval_bytes)

    return resume_from, remote_size


def _log_retry_wait(tag, kind, attempt, max_retries, wait, resume_from):
    print(f"    [{tag}] {kind} (attempt {attempt}/{max_retries}); "
          f"retrying from byte {resume_from} in {wait:.0f}s...", flush=True)


class _DownloadRetryState:
    """Mutable retry state (current url/resume_from/delay) for one shard download attempt loop."""

    def __init__(self, url, resume_from, initial_delay):
        self.url = url
        self.resume_from = resume_from
        self.delay = initial_delay

    def bump_delay(self):
        self.delay = min(self.delay * 1.6, MAX_BACKOFF_DELAY)

    def wait_time(self):
        return min(self.delay, MAX_BACKOFF_DELAY)


def _run_download_attempt(state, tmp_path, tag, chunk_size, progress_interval_s,
                           progress_interval_bytes, attempt, max_retries, url_refresher):
    """
    Run one `_attempt_download` call, updating `state` in place. Returns True
    if the download is complete, False if the caller should retry.
    """
    try:
        state.resume_from, remote_size = _attempt_download(
            state.url, tmp_path, state.resume_from, tag, chunk_size,
            progress_interval_s, progress_interval_bytes,
        )
        if remote_size and state.resume_from < remote_size:
            # Connection closed early without raising - treat as incomplete.
            print(f"    [{tag}] stream ended early at {_format_bytes(state.resume_from)}/"
                  f"{_format_bytes(remote_size)}, retrying...", flush=True)
            return False
        return True
    except _ExpiredUrlError:
        wait = state.wait_time()
        _log_retry_wait(tag, "URL expired/forbidden", attempt, max_retries, wait, state.resume_from)
        if url_refresher is not None:
            state.url = url_refresher()
        time.sleep(wait)
        state.bump_delay()
        return False
    except requests.exceptions.RequestException as e:
        wait = state.wait_time()
        _log_retry_wait(tag, f"connection error ({e})", attempt, max_retries, wait, state.resume_from)
        time.sleep(wait)
        state.bump_delay()
        return False


def download_file_resumable(url, dest_path, chunk_size=1 << 20, max_retries=1000,
                             label=None, progress_interval_s=15, progress_interval_bytes=50 << 20,
                             url_refresher=None, initial_retry_delay=10):
    """
    Stream `url` to `dest_path`, resuming from a partial `.part` file via
    HTTP Range if a previous attempt was interrupted. Skips entirely if
    `dest_path` already exists (i.e. a prior run completed it fully).

    Retries are extremely generous (very high max_retries, capped
    exponential backoff at MAX_BACKOFF_DELAY) - for an unattended HPC job,
    waiting out a transient outage or a burst of expired URLs is strongly
    preferred over letting the whole job fail and having to be resubmitted.

    Progress is reported as plain newline-terminated print()s (not
    carriage-return based) at most every `progress_interval_s` seconds or
    `progress_interval_bytes` bytes, whichever comes first, so it is
    friendly to `tail -f` on a log file.

    `label` is an optional short prefix (e.g. "papers 007/030") used in
    progress/log messages to identify which shard this is.

    `url_refresher`, if given, is a zero-argument callable returning a
    fresh URL for the same shard. Semantic Scholar shard URLs are
    pre-signed and expire (commonly after roughly an hour); on a 403
    Forbidden, `url_refresher` is called to get a new URL and the download
    is retried with it, instead of failing the job.
    """
    tag = label or os.path.basename(dest_path)
    tmp_path = dest_path + ".part"

    if os.path.exists(dest_path):
        print(f"    [{tag}] already downloaded ({_format_bytes(os.path.getsize(dest_path))}), skipping", flush=True)
        return

    resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    if resume_from:
        print(f"    [{tag}] resuming from {_format_bytes(resume_from)}", flush=True)
    else:
        print(f"    [{tag}] starting download", flush=True)

    state = _DownloadRetryState(url, resume_from, initial_retry_delay)
    for attempt in range(1, max_retries + 1):
        if _run_download_attempt(state, tmp_path, tag, chunk_size, progress_interval_s,
                                  progress_interval_bytes, attempt, max_retries, url_refresher):
            break
    else:
        raise RuntimeError(f"Failed to download {tag} after {max_retries} attempts")

    os.rename(tmp_path, dest_path)
    final_size = os.path.getsize(dest_path)
    print(f"    [{tag}] done: {_format_bytes(final_size)}", flush=True)

