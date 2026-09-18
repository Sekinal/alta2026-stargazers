"""Bulk-download whole days of UK/AU subreddit comments (+ posts) from Arctic Shift; /s is found locally.

Arctic Shift's full-text search for "/s" times out server-side (it tokenises to "s"), while plain
time-window listings are cheap, so we page through complete random days per subreddit instead.
Whole days give both classes from the same distribution and most parents (same-day comments/posts).

  uv run python experiments/data/arctic_bulk.py [--workers 2] [--hours 10]

Output: data/external/raw/arctic_bulk/<sub>/<YYYY-MM-DD>.jsonl(.posts.jsonl); a day is written to a temp
file and renamed when complete, so the run can be killed and resumed at any time.
Days are visited newest-era first (2023-2026, the BESSTIE-like period, then 2019-2022), subs round-robin.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from arctic_fetch import GAP_FLOOR, get

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "raw" / "arctic_bulk"
CFIELDS = "id,author,body,created_utc,link_id,parent_id,score,subreddit"
PFIELDS = "id,title,selftext,created_utc"

# (subreddit, first year to sample). The HF UK-city dump already covers CasualUK/london/... to 2022.
SUBS = [
    ("unitedkingdom", 2019), ("australia", 2019), ("ukpolitics", 2019), ("AustralianPolitics", 2019),
    ("CasualUK", 2023), ("melbourne", 2019), ("AskUK", 2019), ("sydney", 2019),
    ("britishproblems", 2019), ("brisbane", 2019), ("Scotland", 2019), ("perth", 2019),
    ("LabourUK", 2019), ("Adelaide", 2019), ("london", 2023), ("AskAnAustralian", 2019),
    ("GreatBritishMemes", 2019), ("canberra", 2019), ("UKPersonalFinance", 2019), ("AusFinance", 2019),
    ("britishpolitics", 2019), ("aussie", 2019), ("Wales", 2019), ("circlejerkaustralia", 2019),
    ("AusPol", 2019), ("manchester", 2023), ("queensland", 2019),
]


def _days(sub: str, first_year: int) -> list[str]:
    end = date(2026, 9, 15)
    rng = random.Random(f"days-{sub}")
    recent = [date(2023, 1, 1) + timedelta(d) for d in range((end - date(2023, 1, 1)).days)]
    older = [date(first_year, 1, 1) + timedelta(d) for d in range((date(2023, 1, 1) - date(first_year, 1, 1)).days)] \
        if first_year < 2023 else []
    rng.shuffle(recent)
    rng.shuffle(older)
    return [d.isoformat() for d in recent + older]


def _page_all(endpoint: str, sub: str, lo: int, hi: int, fields: str) -> list[dict]:
    rows, seen, after = [], set(), lo - 1
    while True:
        page = get(endpoint, subreddit=sub, limit="auto", sort="asc", after=after, before=hi, fields=fields)
        new = [r for r in page if r["id"] not in seen]
        seen.update(r["id"] for r in new)
        rows += new
        if not page:
            break
        last = max(int(r["created_utc"]) for r in page)
        if not new:
            last += 1
        if last >= hi:
            break
        after = last
    return rows


def fetch_day(sub: str, day: str) -> int:
    d = OUT / sub
    out = d / f"{day}.jsonl"
    if out.exists():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    lo = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())
    hi = lo + 86400
    com = _page_all("comments/search", sub, lo, hi, CFIELDS)
    posts = _page_all("posts/search", sub, lo - 86400, hi, PFIELDS)  # posts from the day before too
    for path, rows in ((d / f"{day}.posts.jsonl", posts), (out, com)):  # comments last = completion marker
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.rename(path)
    return len(com)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--subs", default="")
    ap.add_argument("--min-gap", type=float, default=0.5, help="floor on seconds between requests")
    a = ap.parse_args()
    GAP_FLOOR[0] = a.min_gap
    subs = [s for s in SUBS if not a.subs or s[0] in a.subs.split(",")]
    queues = {s: _days(s, y) for s, y in subs}
    stop = time.time() + a.hours * 3600
    lock = threading.Lock()
    order = [(s, day) for i in range(max(len(q) for q in queues.values()))
             for s, _ in subs if i < len(queues[s]) for day in [queues[s][i]]]
    it = iter(order)
    total = [0, 0]

    def worker():
        while time.time() < stop:
            with lock:
                task = next(it, None)
            if task is None:
                return
            try:
                n = fetch_day(*task)
            except Exception as e:
                print(f"FAILED {task}: {e}", flush=True)
                continue
            with lock:
                total[0] += n
                total[1] += 1
                if total[1] % 50 == 0:
                    print(f"{time.strftime('%H:%M:%S')} days={total[1]} comments={total[0]} last={task}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for _ in range(a.workers):
            ex.submit(worker)


if __name__ == "__main__":
    main()
