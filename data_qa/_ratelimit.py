"""GitHub rate-limit handling shared by every data-qa API caller.

GitHub has two limits.  The PRIMARY limit (5000 requests/hour per token) is visible in the
``X-RateLimit-Remaining`` / ``X-RateLimit-Reset`` headers.  The SECONDARY ("abuse") limit is not
visible at all: GitHub documents it only as rough guidance (about 80 content-creating requests per
minute, a few hundred per hour, at most 100 concurrent requests, all shared by every client acting
for the same user) and signals it after the fact with a 403 or 429, usually carrying
``Retry-After``.  The refresh array runs up to ``QA_ARRAY_THROTTLE`` elements at once, each posting
up to ~15 figures, and at 10 elements 45 of 157 elements failed on the secondary limit (array
45412405, 2026-10-10).

Two defences, both cluster-wide because every array element shares the user's limit:

* **Write budget.**  Every content-creating request (anything but GET/HEAD) first takes a token
  from a bucket kept in a small JSON state file under an exclusive POSIX lock, so all elements on
  all nodes draw from one budget (``QA_WRITE_RATE_PER_MIN``, default 30/min, burst
  ``QA_WRITE_BURST``, default 10).  ``QA_WRITE_BUCKET=off`` disables it.
* **Back off and retry.**  A rate-limited response is retried after ``Retry-After`` (else until
  ``X-RateLimit-Reset`` when the primary limit is spent, else an exponential 60 s, 120 s, ... with
  jitter).  The wait is also written into the shared state as ``pause_until`` so every other
  element holds its writes too, instead of each one discovering the limit separately.
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import time

_READ_METHODS = ("GET", "HEAD")
MAX_RETRIES = int(os.environ.get("QA_RATELIMIT_RETRIES", "6"))
MAX_TOTAL_WAIT_S = float(os.environ.get("QA_RATELIMIT_MAX_WAIT_S", "1800"))
_BACKOFF_BASE_S = 60.0
_BACKOFF_CAP_S = 900.0


# The state file lives in $HOME, which is NFS on HiPerGator: POSIX locks there held across nodes
# (4 tasks on 2 nodes x 300 locked increments -> exactly 1200), while the same test on /blue lost
# updates (1018 of 1200), so do not point QA_WRITE_BUCKET at /blue or /orange.
def _bucket_path():
    p = os.environ.get("QA_WRITE_BUCKET",
                       os.path.join(os.path.expanduser("~"), ".cache", "data-qa", "write_bucket.json"))
    return None if p.lower() in ("", "off", "none", "0") else p


def _rate_per_s():
    return float(os.environ.get("QA_WRITE_RATE_PER_MIN", "30")) / 60.0


def _burst():
    return float(os.environ.get("QA_WRITE_BURST", "10"))


def _hdr(headers, name):
    """Case-insensitive header lookup (urllib keeps the server's capitalisation)."""
    for k, v in (headers or {}).items():
        if k.lower() == name:
            return v
    return None


def is_rate_limited(status, payload, headers):
    """True for a primary or secondary rate-limit refusal.  A plain 403 (bad permissions) is not
    one: it carries neither ``Retry-After``, a spent ``X-RateLimit-Remaining``, nor the message."""
    if status not in (403, 429):
        return False
    if _hdr(headers, "retry-after") is not None or _hdr(headers, "x-ratelimit-remaining") == "0":
        return True
    msg = payload.get("message", "") if isinstance(payload, dict) else ""
    return "rate limit" in str(msg).lower()


def retry_wait(headers, attempt, now=None):
    """Seconds to wait before retry number ``attempt`` (0-based)."""
    now = time.time() if now is None else now
    ra = _hdr(headers, "retry-after")
    if ra is not None:
        try:
            return max(1.0, float(ra))
        except ValueError:
            pass                                # an HTTP-date form; fall through to backoff
    if _hdr(headers, "x-ratelimit-remaining") == "0":
        try:
            return max(1.0, float(_hdr(headers, "x-ratelimit-reset")) - now + 1.0)
        except (TypeError, ValueError):
            pass
    return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2 ** attempt) * random.uniform(0.8, 1.2)


class _Locked:
    """Exclusive POSIX lock on the bucket state file; yields the parsed state and writes it back."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, "a+")
        fcntl.lockf(self.fh, fcntl.LOCK_EX)
        self.fh.seek(0)
        try:
            self.state = json.loads(self.fh.read() or "{}")
        except ValueError:
            self.state = {}
        return self.state

    def __exit__(self, *exc):
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(json.dumps(self.state))
        self.fh.flush()
        fcntl.lockf(self.fh, fcntl.LOCK_UN)
        self.fh.close()
        return False


def _take(state, now, rate, burst):
    """Take one token from ``state`` at time ``now``; returns 0 on success, else seconds to wait
    (nothing taken).  Honours a shared ``pause_until`` set by a rate-limited element."""
    pause = float(state.get("pause_until", 0.0))
    if now < pause:
        return pause - now
    tokens = min(burst, float(state.get("tokens", burst)) + (now - float(state.get("t", now))) * rate)
    state["t"] = now
    if tokens >= 1.0:
        state["tokens"] = tokens - 1.0
        return 0.0
    state["tokens"] = tokens
    return (1.0 - tokens) / rate


def acquire_write(sleep=time.sleep, clock=time.time):
    """Block until the shared write budget allows one more content-creating request."""
    path = _bucket_path()
    if path is None:
        return
    rate, burst = _rate_per_s(), _burst()
    while True:
        try:
            with _Locked(path) as state:
                wait = _take(state, clock(), rate, burst)
        except OSError as e:                     # unwritable home/cache: run unthrottled, say so
            print(f"[ratelimit] write budget unavailable ({e}); not throttling", flush=True)
            return
        if wait <= 0:
            return
        sleep(min(wait, 60.0) + random.uniform(0.0, 0.5))


def note_rate_limited(wait, clock=time.time):
    """Record a rate-limit wait in the shared state so every element holds its writes."""
    path = _bucket_path()
    if path is None:
        return
    try:
        with _Locked(path) as state:
            state["pause_until"] = max(float(state.get("pause_until", 0.0)), clock() + wait)
            state["tokens"] = 0.0
            state["t"] = clock()
    except OSError as e:
        print(f"[ratelimit] could not record the pause ({e})", flush=True)


def call(method, send, sleep=time.sleep):
    """Run ``send() -> (status, payload, headers)`` under the write budget, retrying rate-limit
    refusals.  Returns the last response; a refusal that outlasts the retries is returned as is so
    the caller reports it."""
    waited = 0.0
    for attempt in range(MAX_RETRIES + 1):
        if method.upper() not in _READ_METHODS:
            acquire_write(sleep=sleep)
        status, payload, headers = send()
        if not is_rate_limited(status, payload, headers) or attempt == MAX_RETRIES:
            return status, payload, headers
        wait = retry_wait(headers, attempt)
        if waited + wait > MAX_TOTAL_WAIT_S:
            return status, payload, headers
        print(f"[ratelimit] {method} refused ({status}); waiting {wait:.0f}s "
              f"(retry {attempt + 1}/{MAX_RETRIES})", flush=True)
        note_rate_limited(wait)
        sleep(wait)
        waited += wait
    return status, payload, headers
