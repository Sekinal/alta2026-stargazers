"""Download AU/UK subreddit comments + posts from the Arctic Shift API into raw parquet files.

Source: https://arctic-shift.photon-reddit.com (API docs: github.com/ArthurHeitmann/arctic_shift).
Terms: Reddit content, redistributed by Arctic Shift for research; we keep it local and unlabelled.

Sampling: for every subreddit and month in [--start, --end), the month is cut into 4 windows and
at most cap/4 items are taken from the start of each window (ascending), so a month's sample is
spread over the month instead of being the first N items. Posts get cap * POST_FRAC.

    uv run python experiments/data/fetch_reddit_arctic.py --out /workspace/raw/reddit --workers 4
Resumable: one parquet per (subreddit, kind); existing files are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

API = "https://arctic-shift.photon-reddit.com/api"
UA = {"User-Agent": "alta2026-research-corpus/0.1 (unlabelled domain-adaptive pretraining)"}
WINDOWS = 4
POST_FRAC = 0.25

# subreddit -> max comments per month (posts get POST_FRAC of that)
SUBS = {
    "en-UK": {
        "unitedkingdom": 4000, "ukpolitics": 4000, "CasualUK": 3000, "AskUK": 3000, "london": 2500,
        "britishproblems": 1500, "manchester": 1500, "UKPersonalFinance": 2000, "AskBrits": 1500,
        "Scotland": 1000, "glasgow": 1000, "Edinburgh": 1000, "bristol": 1000, "brum": 1000,
        "Leeds": 1000, "LabourUK": 1000, "uktravel": 600, "GreatBritishMemes": 600,
    },
    "en-AU": {
        "australia": 4000, "AustralianPolitics": 3000, "aussie": 1500, "melbourne": 3000,
        "sydney": 3000, "brisbane": 2000, "perth": 2000, "adelaide": 2000, "AusFinance": 3000,
        "AskAnAustralian": 2000, "canberra": 1000, "hobart": 800, "AusPropertyChat": 1000,
        "AusLegal": 800, "GoldCoast": 600, "tasmania": 600, "AustraliaTravel": 600, "straya": 600,
    },
}
COMMENT_FIELDS = "id,body,author,created_utc,score,parent_id,link_id,subreddit"
POST_FIELDS = "id,title,selftext,author,created_utc,score,num_comments,subreddit,link_flair_text"

_lock = threading.Lock()


def log(*a):
    with _lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(path: str, params: dict) -> list[dict]:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                d = json.load(r)
            if d.get("error"):
                raise RuntimeError(d["error"])
            return d["data"] or []
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("X-RateLimit-Reset", 0) or 0) + 1 if e.code == 429 else 2 ** attempt
            log(f"HTTP {e.code} {path} {params.get('subreddit')} {params.get('after')} -> sleep {wait}s")
            time.sleep(wait)
        except Exception as e:  # timeouts, resets, server-side errors
            log(f"ERR {type(e).__name__}: {e} {params.get('subreddit')} {params.get('after')}")
            time.sleep(2 ** attempt)
    log(f"GAVE UP {url}")
    return []


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) < (ye, me):
        a = datetime(y, m, 1, tzinfo=timezone.utc)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        yield int(a.timestamp()), int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())


def fetch_window(kind: str, sub: str, lo: int, hi: int, cap: int) -> list[dict]:
    out, after = [], lo
    path, fields = ("comments/search", COMMENT_FIELDS) if kind == "comments" else ("posts/search", POST_FIELDS)
    while len(out) < cap:
        batch = get(path, {"subreddit": sub, "after": after, "before": hi, "sort": "asc",
                           "limit": "auto", "fields": fields})
        if not batch:
            break
        out.extend(batch)
        last = max(int(b["created_utc"]) for b in batch)
        if last <= after:  # >1 page of items in the same second: step past it
            last = after + 1
        after = last
        if len(batch) < 100:
            break
    return out[:cap]


def fetch_sub(variety: str, sub: str, cap: int, kind: str, args) -> None:
    dst = Path(args.out) / f"{sub}_{kind}.parquet"
    if dst.exists():
        log(f"skip {dst.name} (exists)")
        return
    t0, rows = time.time(), []
    cap = cap if kind == "comments" else max(20, int(cap * POST_FRAC))
    for lo, hi in months(args.start, args.end):
        step = (hi - lo) // WINDOWS
        for w in range(WINDOWS):
            rows.extend(fetch_window(kind, sub, lo + w * step, lo + (w + 1) * step if w < WINDOWS - 1 else hi,
                                     cap // WINDOWS))
    if not rows:
        log(f"EMPTY {sub} {kind}")
        return
    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.lit(variety).alias("variety"), pl.lit(kind).alias("kind"))
    df = df.unique("id", keep="first")
    tmp = dst.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.rename(dst)
    log(f"done {sub} {kind}: {df.height} rows in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/raw/reddit")
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default="2026-01")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="restrict to these subreddits")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(v, s, c, k) for v, subs in SUBS.items() for s, c in subs.items() for k in ("comments", "posts")
            if not args.only or s in args.only]
    jobs.sort(key=lambda j: -j[2])  # big subs first
    log(f"{len(jobs)} jobs, {args.start}..{args.end}")
    with ThreadPoolExecutor(args.workers) as ex:
        for f in [ex.submit(fetch_sub, v, s, c, k, args) for v, s, c, k in jobs]:
            f.result()
    log("ALL DONE")


if __name__ == "__main__":
    sys.exit(main())
