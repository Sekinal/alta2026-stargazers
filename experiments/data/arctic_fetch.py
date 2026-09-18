"""Fetch /s-marked (weakly sarcastic) and unmarked comments from UK/AU subreddits via Arctic Shift.

Source: https://arctic-shift.photon-reddit.com (public Reddit archive API, no auth).
Raw JSONL goes to data/external/raw/arctic/ (git-ignored); arctic_build.py turns it into parquet.

Stages (each resumable; re-running skips finished work):
  pos      every comment matching body "/s" per subreddit, paged by created_utc
  neg      unmarked comments from the same subreddit, sampled per month in proportion to positives
  parents  text of the parent comment/post for every kept comment (context, both classes)

  uv run python experiments/data/arctic_fetch.py pos neg parents [--subs unitedkingdom,...] [--workers 3]
"""

from __future__ import annotations

import argparse
import calendar
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "external" / "raw" / "arctic"
API = "https://arctic-shift.photon-reddit.com/api"
FIELDS = "id,author,body,created_utc,link_id,parent_id,score,subreddit"
NEG_PER_POS = 2.0

SUBS = {
    "en-UK": ["unitedkingdom", "ukpolitics", "CasualUK", "AskUK", "britishproblems", "london", "Scotland",
              "LabourUK", "uknews", "britishpolitics", "GreatBritishMemes", "UKPersonalFinance", "england",
              "manchester", "brexit", "BritishSuccess", "Wales", "northernireland", "bristol", "leeds",
              "glasgow", "edinburgh", "birmingham", "tories", "unitedkingdomcirclejerk", "LibDem"],
    "en-AU": ["australia", "AustralianPolitics", "melbourne", "sydney", "brisbane", "perth", "Adelaide",
              "canberra", "AusFinance", "AskAnAustralian", "aussie", "straya", "AusPropertyChat", "auslaw",
              "queensland", "tasmania", "GoldCoast", "AusPol", "circlejerkaustralia",
              "australian", "darwin", "AusEcon", "centrelink", "ausjobs"],
}
VARIETY = {s: v for v, subs in SUBS.items() for s in subs}

_lock = threading.Lock()
_last = [0.0]
_gap = [1.0]  # seconds between requests across all threads; doubles on 429, decays on success
GAP_FLOOR = [0.5]


def get(path: str, **params) -> list[dict]:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(12):
        with _lock:
            wait = _last[0] + _gap[0] - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "alta-research-sarcasm/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            if out.get("error"):
                raise RuntimeError(out["error"])
            with _lock:
                _gap[0] = max(GAP_FLOOR[0], _gap[0] * 0.98)
            return out.get("data") or []
        except urllib.error.HTTPError as e:
            if e.code == 400:
                print(f"400 {url}: {e.read()[:200]!r}", flush=True)
                return []
            print(f"http {e.code} {path} {params.get('subreddit', '')}: {e.read()[:120]!r}", flush=True)
            delay = 5 * (attempt + 1)
            if e.code == 429:
                with _lock:
                    _gap[0] = min(8.0, _gap[0] * 2)
                delay = float(e.headers.get("x-ratelimit-reset") or 30) + 1
        except Exception as e:  # timeouts, resets, server-side query timeouts
            print(f"retry {attempt} {path} {params.get('subreddit', '')}: {e}", flush=True)
            delay = 5 * (attempt + 1)
        time.sleep(delay)
    raise RuntimeError(f"gave up on {url}")


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def _months(start: str = "2008-01") -> list[str]:
    y, m = map(int, start.split("-"))
    now = datetime.now(timezone.utc)
    out = []
    while (y, m) <= (now.year, now.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_pos(sub: str) -> None:
    """Month-bounded windows: open-ended full-text queries time out server-side."""
    out, done, prog = RAW / f"{sub}.pos.jsonl", RAW / f"{sub}.pos.done", RAW / f"{sub}.pos.months"
    if done.exists():
        return
    finished = set(prog.read_text().split()) if prog.exists() else set()
    seen = {r["id"] for r in _read(out)}
    for ym in _months():
        if ym in finished:
            continue
        lo, hi = _month_bounds(ym)
        after = lo - 1
        while True:
            page = get("comments/search", subreddit=sub, body='"/s"', limit=100, sort="asc",
                       after=after, before=hi + 1, fields=FIELDS)
            new = [r for r in page if r["id"] not in seen]
            if new:
                seen.update(r["id"] for r in new)
                _append(out, new)
            if len(page) < 100:
                break
            nxt = max(r["created_utc"] for r in page)
            after = nxt if new else nxt + 1  # a full page sharing one timestamp: step past it
        with prog.open("a") as f:
            f.write(ym + "\n")
        if ym.endswith("-12"):
            print(f"{sub} pos {len(seen)} through {ym}", flush=True)
    done.touch()
    print(f"{sub} pos DONE {len(seen)}", flush=True)


def _month_bounds(ym: str) -> tuple[int, int]:
    y, m = map(int, ym.split("-"))
    start = calendar.timegm((y, m, 1, 0, 0, 0))
    return start, calendar.timegm((y, m, calendar.monthrange(y, m)[1], 23, 59, 59))


def fetch_neg(sub: str) -> None:
    """Per month with k positives, draw ~NEG_PER_POS*k unmarked comments from random start times."""
    out, done = RAW / f"{sub}.neg.jsonl", RAW / f"{sub}.neg.done"
    if done.exists():
        return
    pos = _read(RAW / f"{sub}.pos.jsonl")
    if not pos:
        done.touch()
        return
    per_month = Counter(datetime.fromtimestamp(r["created_utc"], timezone.utc).strftime("%Y-%m") for r in pos)
    have = Counter(datetime.fromtimestamp(r["created_utc"], timezone.utc).strftime("%Y-%m") for r in _read(out))
    rng = random.Random(f"{sub}-neg")
    for ym, k in sorted(per_month.items()):
        target = int(round(NEG_PER_POS * k))
        lo, hi = _month_bounds(ym)
        tries = 0
        while have[ym] < target and tries < 4 + target // 20:
            tries += 1
            t0 = rng.randint(lo, hi)
            page = get("comments/search", subreddit=sub, limit=100, sort="asc", after=t0, before=hi,
                       fields=FIELDS)
            # take a small contiguous slice per draw so one thread doesn't dominate a month
            page = [r for r in page if r.get("author") not in ("AutoModerator", "[deleted]")
                    and r.get("body") not in ("[deleted]", "[removed]")][:25]
            if not page:
                continue
            _append(out, page)
            have[ym] += len(page)
    done.touch()
    print(f"{sub} neg DONE {sum(have.values())}", flush=True)


def fetch_parents(sub: str) -> None:
    out, done = RAW / f"{sub}.parents.jsonl", RAW / f"{sub}.parents.done"
    if done.exists():
        return
    want = {r["parent_id"] for kind in ("pos", "neg") for r in _read(RAW / f"{sub}.{kind}.jsonl")
            if r.get("parent_id")}
    have = {r["pid"] for r in _read(out)}
    todo = sorted(want - have)
    for prefix, endpoint, fields in (("t1_", "comments/ids", "id,body"), ("t3_", "posts/ids", "id,title,selftext")):
        ids = [p[3:] for p in todo if p.startswith(prefix)]
        for i in range(0, len(ids), 100):
            page = get(endpoint, ids=",".join(ids[i:i + 100]), fields=fields)
            rows = []
            for r in page:
                text = r.get("body") if prefix == "t1_" else "\n\n".join(
                    x for x in (r.get("title"), r.get("selftext")) if x and x not in ("[deleted]", "[removed]"))
                rows.append({"pid": prefix + r["id"], "text": text})
            _append(out, rows)
    done.touch()
    print(f"{sub} parents DONE {len(want)} wanted", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["pos", "neg", "parents"])
    ap.add_argument("--subs", default="")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    subs = a.subs.split(",") if a.subs else list(VARIETY)
    fn = {"pos": fetch_pos, "neg": fetch_neg, "parents": fetch_parents}
    for stage in a.stages:
        with ThreadPoolExecutor(a.workers) as ex:
            for sub, exc in zip(subs, ex.map(lambda s: _safe(fn[stage], s), subs)):
                if exc:
                    print(f"{stage} {sub} FAILED: {exc}", file=sys.stderr, flush=True)


def _safe(f, sub):
    try:
        f(sub)
    except Exception as e:  # keep other subreddits going
        return repr(e)
    return None


if __name__ == "__main__":
    main()
