"""Recover the Reddit context (parent comment / submission) of BESSTIE-style Reddit texts.

BESSTIE ships only the comment text. The comments are public, so we find each text verbatim in
Reddit dumps and pull its parent (or, for top-level comments, the submission title + selftext).

Stages:

  fetch   (optional) download whole days of comments + posts for chosen subreddits from the Arctic Shift
          API into data/external/raw/context_days/<sub>/<day>.jsonl (newest day first, resumable). BESSTIE's
          Reddit texts are the latest comments of 7 subreddits before ~2024-06-02 (BESSTIE_SUBS), so ~700
          subreddit-days (~340 MB) cover 95% of the pool. Without --subs it fetches exactly those.

      uv run python experiments/data/context_match.py fetch            # BESSTIE_SUBS, ~1 h
      uv run python experiments/data/context_match.py fetch --subs Britain --start 2024-01-01 --end 2024-02-15

  index   scan Reddit dumps (Pushshift-style per-subreddit .zst, Arctic Shift .jsonl / .parquet) and
          write a compact hash index of every comment/post: hashes of the normalised text plus id,
          subreddit, created_utc, score, parent_id, link_id. Raw dumps can be deleted afterwards.

      uv run --with zstandard,orjson python experiments/data/context_match.py index \\
          --src data/external/raw/context_days --out data/external/context/index/days --procs 8

  match   normalise the Reddit rows of a CSV (test.csv / valid.csv; uids as in alta.common.load_test)
          or the pool (--pool), join them to the index, then fetch the context of every match by id
          from the Arctic Shift API.

      uv run --with orjson python experiments/data/context_match.py match --input data/test.csv \\
          --out data/external/context/test_context.parquet

Output (one row per input row incl. Google rows, which stay unmatched; NO labels): uid, matched,
match_type (exact/fuzzy/null), match_score (difflib ratio; 1 for exact), n_candidates, in_window, kind,
comment_id, subreddit, created_utc, score, is_top_level, parent_id, link_id, parent_text, submission_title,
submission_selftext (context texts cut to 512 words). parent_text is null for top-level comments and for
deleted/removed parents; ~70% of BESSTIE's comments are top-level, so the submission is the usual context.

Normalisation (both sides): HTML unescape, markdown links -> link text, URLs and standalone "/s"
dropped, lowercase, keep [a-z0-9] only (whitespace too is dropped: BESSTIE texts lose line breaks and
spaces around removed links). Exact = equal key; fuzzy = the first or last 40 key chars agree and the
full keys have difflib ratio >= 0.85 (edited comments, truncated texts).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import io
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alta.common import load_pool, load_test  # noqa: E402

INDEX = ROOT / "data" / "external" / "context" / "index"
DAYS = ROOT / "data" / "external" / "raw" / "context_days"
EDGE = 40  # key chars used for the prefix/suffix (fuzzy) hashes
MIN_FUZZY_RATIO = 0.85
MIN_COVER, MIN_LEN_RATIO = 0.9, 0.5  # or: 90% of the shorter key matches and lengths within 2x
MAX_FUZZY = 20  # fuzzy candidates verified per text
WINDOW = "2024-02-15:2024-06-04"  # BESSTIE's Reddit crawl (pool matches: 2024-02-18 .. 2024-06-03 00:24 UTC)
# Subreddits of the pool matches, with the first day each needs (the latest comments before ~2024-06-02 were
# taken per subreddit, so small subs reach further back). fetch --subs/--start these to rebuild the index.
BESSTIE_SUBS = {"AustralianPolitics": "2024-04-01", "melbourne": "2024-05-10", "AskAnAustralian": "2024-04-01",
                "unitedkingdom": "2024-05-20", "england": "2024-03-01", "GreatBritishMemes": "2024-02-15",
                "Britain": "2024-02-10"}
BESSTIE_END = "2024-06-04"
MIN_UNIQUE = 25  # shorter keys only count as matched when they occur exactly once in the index
MAX_WORDS = 512
CHUNK = 2_000_000

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_SARC = re.compile(r"(?<![\w/])/s(?![\w/])")
_NON = re.compile(r"[^a-z0-9]+")


def norm_key(text: str | None) -> str:
    if not text:
        return ""
    t = _URL.sub(" ", _LINK.sub(r"\1", html.unescape(text))).lower()
    return _NON.sub("", _SARC.sub(" ", t))


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def key_hashes(key: str) -> tuple[int, int | None, int | None]:
    if len(key) < EDGE:
        return h64(key), None, None
    return h64(key), h64(key[:EDGE]), h64(key[-EDGE:])


# ---------------------------------------------------------------------------------------- fetch

def cmd_fetch(args) -> None:
    import threading
    from datetime import date, timedelta

    import arctic_bulk
    from arctic_fetch import GAP_FLOOR
    arctic_bulk.OUT = Path(args.out)  # fetch_day writes <OUT>/<sub>/<day>.jsonl (+ .posts.jsonl)
    GAP_FLOOR[0] = args.min_gap
    starts = ({s: args.start for s in args.subs.split(",")} if args.subs else BESSTIE_SUBS)
    d1 = date.fromisoformat(args.end)
    first = min(date.fromisoformat(d) for d in starts.values())
    days = [(d1 - timedelta(i + 1)).isoformat() for i in range((d1 - first).days)]  # newest first
    tasks = iter([(s, d) for d in days for s, d0 in starts.items() if d >= d0])
    lock, n = threading.Lock(), [0]

    def worker():
        while True:
            with lock:
                task = next(tasks, None)
            if task is None:
                return
            try:
                c = arctic_bulk.fetch_day(*task)
            except Exception as e:  # noqa: BLE001 - a failed day is retried on the next run
                print(f"FAILED {task}: {e}", flush=True)
                continue
            with lock:
                n[0] += 1
                print(f"{time.strftime('%H:%M:%S')} {task[0]} {task[1]}: {c} comments ({n[0]} days)", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ---------------------------------------------------------------------------------------- index

def _records(path: Path):
    """Yield raw dicts from a Pushshift .zst, an Arctic Shift .jsonl or a .parquet file."""
    if path.suffix == ".parquet":
        yield from pl.read_parquet(path).iter_rows(named=True)
        return
    import orjson
    if path.suffix == ".zst":
        import zstandard
        fh = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(path.open("rb"))
        lines = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
    else:
        lines = path.open(encoding="utf-8", errors="replace")
    for line in lines:
        try:
            yield orjson.loads(line)
        except orjson.JSONDecodeError:
            continue


def _int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def index_file(path: Path, out_dir: Path, since: int, until: int) -> tuple[str, int, int]:
    name = f"{path.parent.name}__{path.name}"  # Arctic Shift day files share names across subreddits
    done = out_dir / f"{name}.done"
    if done.exists():
        return name, 0, 0
    rows, n_seen, part = [], 0, 0

    def flush():
        nonlocal rows, part
        if rows:
            pl.DataFrame(rows, schema=SCHEMA, orient="row").write_parquet(out_dir / f"{name}.{part:03d}.parquet")
            part += 1
            rows = []

    for r in _records(path):
        n_seen += 1
        ts = _int(r.get("created_utc"))
        if ts is None or not since <= ts < until:
            continue
        if "body" in r:
            texts = [("comment", r.get("body"))]
        elif "title" in r or "selftext" in r:
            texts = [("post", r.get("selftext")), ("title", r.get("title"))]
        else:
            continue
        for kind, text in texts:
            key = norm_key(text)
            if not key or text in ("[deleted]", "[removed]"):
                continue
            hk, hp, hs = key_hashes(key)
            rows.append((hk, hp, hs, len(key), kind, str(r.get("id")), r.get("subreddit"), ts, _int(r.get("score")),
                         r.get("parent_id"), r.get("link_id")))
        if len(rows) >= CHUNK:
            flush()
    flush()
    done.write_text(str(n_seen))
    return name, n_seen, part


SCHEMA = {"h_key": pl.Int64, "h_pre": pl.Int64, "h_suf": pl.Int64, "key_len": pl.Int32, "kind": pl.String,
          "id": pl.String, "subreddit": pl.String, "created_utc": pl.Int64, "score": pl.Int64,
          "parent_id": pl.String, "link_id": pl.String}


def cmd_index(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for src in args.src:
        p = Path(src)
        files += [p] if p.is_file() else sorted(q for q in p.rglob("*")
                                                   if q.suffix in (".zst", ".jsonl", ".parquet") and q.is_file())
    files.sort(key=lambda p: -p.stat().st_size)  # big files first
    ts = lambda d: int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())  # noqa: E731
    since, until = ts(args.since), ts(args.until)
    print(f"{len(files)} files -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.procs) as ex:
        for name, n, parts in ex.map(index_file, files, [out] * len(files), [since] * len(files),
                                     [until] * len(files)):
            print(f"{time.time() - t0:7.0f}s {name}: {n} records, {parts} parts", flush=True)


# ---------------------------------------------------------------------------------------- match

def _load_input(args) -> pl.DataFrame:
    if args.pool:
        return load_pool().select("uid", "source", "variety", "text")
    return load_test(args.input).select("uid", "source", "variety", "text")


def _candidates(q: pl.DataFrame, index_dirs: list[Path]) -> pl.DataFrame:
    """Index rows whose full-key hash equals a query's (exact), or, for texts without an exact hit, whose
    prefix or suffix hash does (fuzzy candidates, verified on the full text later)."""
    files = [str(f) for d in index_dirs for f in sorted(d.rglob("*.parquet"))]
    idx = pl.scan_parquet(files)
    cols = ["id", "kind", "subreddit", "created_utc", "score", "parent_id", "link_id", "key_len"]
    q = q.lazy().with_columns(pl.col("qkey").str.len_chars().alias("qlen"))
    exact = q.join(idx.select("h_key", *cols), on="h_key").select("uid", "qlen", *cols).with_columns(
        pl.lit("exact").alias("match_type")).collect()
    fuzzy = pl.concat([
        q.filter(pl.col(h).is_not_null()).join(idx.filter(pl.col(h).is_not_null()).select(h, *cols), on=h)
        .select("uid", "qlen", *cols) for h in ("h_pre", "h_suf")]).with_columns(
        pl.lit("fuzzy").alias("match_type")).collect()
    fuzzy = fuzzy.filter(~pl.col("uid").is_in(exact["uid"].implode()))
    return pl.concat([exact, fuzzy]).unique(["uid", "id", "kind"], keep="first", maintain_order=True)


def _similarity(a: str, b: str) -> tuple[float, float]:
    """difflib ratio, and the share of the shorter key covered by matching blocks (edits that append or
    cut text, e.g. BESSTIE scraped an edited comment while the archive holds the original)."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = sum(m.size for m in sm.get_matching_blocks())
    return sm.ratio(), blocks / max(1, min(len(a), len(b)))


def _chunks(xs: list[str], n: int = 100):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _fetch(kind: str, ids: list[str], fields: str) -> dict[str, dict]:
    from arctic_fetch import get
    out = {}
    for ch in _chunks(sorted(set(ids))):
        for r in get(f"{kind}/ids", ids=",".join(ch), fields=fields):
            out[r["id"]] = r
    return out


def _clean(text: str | None) -> str | None:
    if text is None or text.strip() in ("", "[deleted]", "[removed]"):
        return None
    words = text.split()
    return text if len(words) <= MAX_WORDS else " ".join(words[:MAX_WORDS])


def cmd_match(args) -> None:
    rows = _load_input(args)
    red = rows.filter(pl.col("source") == "reddit")
    keys = [norm_key(t) for t in red["text"]]
    hs = [key_hashes(k) for k in keys]
    q = red.select("uid").with_columns(
        pl.Series("qkey", keys, pl.String), pl.Series("h_key", [h[0] for h in hs], pl.Int64),
        pl.Series("h_pre", [h[1] for h in hs], pl.Int64), pl.Series("h_suf", [h[2] for h in hs], pl.Int64))
    index_dirs = [Path(d) for d in args.index] if args.index else [INDEX]
    cand = _candidates(q.filter(pl.col("qkey") != ""), index_dirs)
    print(f"{cand.height} candidate rows for {cand['uid'].n_unique()} / {red.height} reddit texts", flush=True)

    lo, hi = (int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp()) for d in args.window.split(":"))
    cand = cand.with_columns(pl.col("created_utc").is_between(lo, hi, closed="left").alias("in_window"))

    # fuzzy candidates are verified on the full text, fetched by id (comments and posts)
    fz = cand.filter(pl.col("match_type") == "fuzzy").sort("uid", ~pl.col("in_window"), "created_utc").group_by(
        "uid", maintain_order=True).head(MAX_FUZZY)
    if fz.height:
        bodies = _fetch("comments", fz.filter(pl.col("kind") == "comment")["id"].to_list(), "id,body")
        posts = _fetch("posts", fz.filter(pl.col("kind") != "comment")["id"].to_list(), "id,title,selftext")
        qk = dict(zip(q["uid"], q["qkey"]))

        def sim(uid, cid, kind):
            src = (bodies.get(cid, {}).get("body") if kind == "comment"
                   else posts.get(cid, {}).get("selftext" if kind == "post" else "title"))
            k = norm_key(src)
            if not k:
                return 0.0, 0.0, 0.0
            r, cov = _similarity(qk[uid], k)
            return r, cov, min(len(k), len(qk[uid])) / max(len(k), len(qk[uid]))
        sims = [sim(*r) for r in fz.select("uid", "id", "kind").iter_rows()]
        fz = fz.with_columns(pl.Series("match_score", [x[0] for x in sims], pl.Float64),
                             pl.Series("_cov", [x[1] for x in sims], pl.Float64),
                             pl.Series("_lenr", [x[2] for x in sims], pl.Float64)).filter(
            (pl.col("match_score") >= MIN_FUZZY_RATIO)
            | ((pl.col("_cov") >= MIN_COVER) & (pl.col("_lenr") >= MIN_LEN_RATIO))).drop("_cov", "_lenr")
    else:
        fz = fz.with_columns(pl.lit(None, pl.Float64).alias("match_score"))
    cand = pl.concat([cand.filter(pl.col("match_type") == "exact").with_columns(pl.lit(1.0).alias("match_score")),
                      fz])

    # BESSTIE-era copies win: if a text has candidates inside --window, the others are dropped. Then one
    # match per uid: prefer exact, comments, the best score, then the earliest copy (reposts come later).
    # Short texts ("lol") only count when a single candidate is left.
    cand = cand.filter(pl.col("in_window") | ~pl.col("in_window").any().over("uid"))
    cand = cand.with_columns(pl.len().over("uid").alias("n_candidates")).sort(
        "uid", pl.col("match_type") != "exact", pl.col("kind") != "comment", -pl.col("match_score"),
        "created_utc").unique("uid", keep="first", maintain_order=True)
    cand = cand.filter((pl.col("qlen") >= MIN_UNIQUE) | (pl.col("n_candidates") == 1))
    print(cand.group_by("match_type", "kind").len(), flush=True)

    com = cand.filter(pl.col("kind") == "comment")
    parent_c = [p[3:] for p in com["parent_id"].drop_nulls() if p.startswith("t1_")]
    links = [p[3:] for p in com["link_id"].drop_nulls()]
    pc = _fetch("comments", parent_c, "id,body")
    pp = _fetch("posts", links, "id,title,selftext")
    print(f"fetched {len(pc)}/{len(set(parent_c))} parent comments, {len(pp)}/{len(set(links))} submissions",
          flush=True)

    def ctx(r):
        par, link = r["parent_id"] or "", (r["link_id"] or "")[3:]
        post = pp.get(link, {})
        return {"is_top_level": par.startswith("t3_") if par else None,
                "parent_text": _clean(pc.get(par[3:], {}).get("body")) if par.startswith("t1_") else None,
                "submission_title": _clean(post.get("title")), "submission_selftext": _clean(post.get("selftext"))}
    extra = pl.DataFrame([ctx(r) for r in cand.iter_rows(named=True)],
                         schema={"is_top_level": pl.Boolean, "parent_text": pl.String,
                                 "submission_title": pl.String, "submission_selftext": pl.String})
    m = pl.concat([cand, extra], how="horizontal_extend").select(
        "uid", pl.lit(True).alias("matched"), "match_type", "match_score", "n_candidates", "in_window", "kind",
        pl.col("id").alias("comment_id"), "subreddit", "created_utc", "score", "is_top_level", "parent_id", "link_id",
        "parent_text", "submission_title", "submission_selftext")
    out = rows.select("uid").join(m, on="uid", how="left", maintain_order="left").with_columns(
        pl.col("matched").fill_null(False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    rep = rows.join(out, on="uid").filter(pl.col("source") == "reddit").group_by("variety").agg(
        pl.len(), pl.col("matched").mean().round(3).alias("match_rate"),
        (pl.col("match_type") == "fuzzy").sum().alias("fuzzy"),
        pl.col("parent_text").is_not_null().sum().alias("with_parent_text"),
        pl.col("is_top_level").sum().alias("top_level"),
        pl.from_epoch("created_utc").min().alias("first"), pl.from_epoch("created_utc").max().alias("last"))
    print(rep, flush=True)
    print(f"wrote {args.out} ({out.height} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("index")
    a.add_argument("--src", nargs="+", required=True, help="dump files or directories")
    a.add_argument("--out", required=True)
    a.add_argument("--since", default="2005-01-01")
    a.add_argument("--until", default="2100-01-01")
    a.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1))
    f = sp.add_parser("fetch")
    f.add_argument("--subs", help="comma-separated subreddits (default: BESSTIE_SUBS with their start days)")
    f.add_argument("--start", help="first day (inclusive), YYYY-MM-DD; required with --subs")
    f.add_argument("--end", default=BESSTIE_END, help="last day (exclusive)")
    f.add_argument("--out", default=str(DAYS))
    f.add_argument("--workers", type=int, default=3)
    f.add_argument("--min-gap", type=float, default=0.4, help="floor on seconds between API requests")
    b = sp.add_parser("match")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="CSV with source/variety/text columns (test.csv layout)")
    g.add_argument("--pool", action="store_true", help="match the labelled pool (train+valid)")
    b.add_argument("--out", required=True)
    b.add_argument("--index", nargs="*", help=f"index directories (default {INDEX})")
    b.add_argument("--window", default=WINDOW, help="preferred created_utc range START:END (ISO dates)")
    args = ap.parse_args()
    if args.cmd == "fetch" and args.subs and not args.start:
        ap.error("--start is required with --subs")
    {"fetch": cmd_fetch, "index": cmd_index, "match": cmd_match}[args.cmd](args)


if __name__ == "__main__":
    main()
