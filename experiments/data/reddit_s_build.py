"""/s-labelled reddit comments from UK/AU subreddits -> data/external/sarcasm/reddit_s_*.parquet.

Positives: comments whose author tagged them "/s" (tag stripped, meta-mentions and quoted /s excluded;
see ext_common.strip_s_tag). Negatives: untagged comments from the same subreddit and year, sampled at
NEG_RATIO per positive (weak: some unmarked comments are sarcastic too). parent_text is given for both
classes when the parent is in the dump/archive, so its presence does not leak the label.

Inputs (any mix):
  data/external/raw/usuk/<Sub>_{comments,submissions}.zst    Pushshift-format per-subreddit dumps
                                                              (alujjdnd/Reddit-US-UK on HF, to 2022-12)
  data/external/raw/arctic/<sub>.{pos,neg,parents}.jsonl       arctic_fetch.py output (any date)
  data/external/raw/arctic_bulk/<sub>/<day>.jsonl              arctic_bulk.py output (whole days, 2019-2026)
  data/external/raw/convokit/<sub>.corpus.zip                  ConvoKit subreddit corpora (to 2018-10)

  uv run --with zstandard,orjson,pyahocorasick python experiments/data/reddit_s_build.py zst convokit  # dumps -> candidates
  uv run --with zstandard,orjson,pyahocorasick python experiments/data/reddit_s_build.py build  # candidates -> datasets
"""

from __future__ import annotations

import html
import io
import random
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path

import orjson
import polars as pl
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arctic_fetch import VARIETY  # noqa: E402
from ext_common import OUT as ROOT_SARC, RAW, finalize, has_s_tag, strip_s_tag, write  # noqa: E402

USUK = RAW / "usuk"
ARCTIC = RAW / "arctic"
BULK = RAW / "arctic_bulk"
CONVOKIT = RAW / "convokit"
CAND = RAW / "reddit_s_candidates"
NEG_RATIO = 3
NEG_KEEP_P = 0.08  # first-pass sampling rate for negatives (subsampled to NEG_RATIO later)
BOTS = {"AutoModerator", "[deleted]", "Bot", "RemindMeBot", "sneakpeekbot", "WikiTextBot", "B0tRank"}
LICENSE = "Reddit user content; research use (Reddit API/archive terms apply)"
# r/newcastle is ambiguous (Newcastle NSW vs Newcastle upon Tyne): mapped to "unknown" and so excluded
VAR = {k.lower(): v for k, v in VARIETY.items()} | {"liverpool": "en-UK", "southampton": "en-UK",
                                                     "nottingham": "en-UK", "sheffield": "en-UK"}


def _lines(path: Path):
    with path.open("rb") as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        with dctx.stream_reader(fh) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8", errors="replace"):
                yield line


def _ok(body: str | None, author: str | None) -> bool:
    return bool(body) and body not in ("[deleted]", "[removed]") and author not in BOTS \
        and not (author or "").lower().endswith("bot") and 10 <= len(body) <= 3000


_MD_LINK = re.compile(r"\[([^\]\n]+)\]\((?!/s\))[^)\s]*\)")


def clean_body(body: str | None) -> str | None:
    """HTML entities (&amp; &gt;) as in the dumps, markdown links -> their text; keeps "[x](/s)" for the tagger."""
    if not body:
        return body
    body = _MD_LINK.sub(r"\1", html.unescape(body))
    return body.replace("&#x200B;", "").replace("\u200b", "")


def _unbracket(text: str) -> str:
    m = re.fullmatch(r"\[([^\]]+)\]", text.strip())
    return m.group(1) if m else text


def _candidate(c: dict, rng: random.Random) -> dict | None:
    body, author = clean_body(c.get("body")), c.get("author")
    if not _ok(body, author):
        return None
    if has_s_tag(body):
        text, y = strip_s_tag(body), 1
    elif "/s" in body.lower() or "\\s" in body or rng.random() >= NEG_KEEP_P:
        return None
    else:  # drop quoted lines, as strip_s_tag does for positives
        text, y = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith(">")).strip(), 0
    if not text or len(text) < 8:
        return None
    return {"id": c.get("id"), "text": _unbracket(text), "sarcasm": y, "parent_id": c.get("parent_id"),
            "created_utc": int(float(c.get("created_utc") or 0)), "subreddit": c.get("subreddit"),
            "score": c.get("score")}


def zst_candidates() -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    for cpath in sorted(USUK.glob("*_comments.zst")):
        sub = cpath.name.removesuffix("_comments.zst")
        out = CAND / f"usuk_{sub}.parquet"
        if out.exists():
            continue
        rng = random.Random(sub)
        rows, n = [], 0
        for line in _lines(cpath):
            n += 1
            try:
                c = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            r = _candidate(c, rng)
            if r:
                rows.append(r)
        df = pl.DataFrame(rows, schema={"id": pl.String, "text": pl.String, "sarcasm": pl.Int8, "parent_id": pl.String,
                                        "created_utc": pl.Int64, "subreddit": pl.String, "score": pl.Int64},
                          strict=False)
        # second pass: parent texts (comments, then submission title + selftext)
        want = set(df["parent_id"].drop_nulls().to_list())
        par = {}
        for line in _lines(cpath):
            if '"id"' not in line:
                continue
            c = orjson.loads(line)
            pid = "t1_" + str(c.get("id"))
            if pid in want and c.get("body") not in (None, "[deleted]", "[removed]"):
                par[pid] = clean_body(c["body"])
        spath = USUK / f"{sub}_submissions.zst"
        if spath.exists():
            for line in _lines(spath):
                s = orjson.loads(line)
                pid = "t3_" + str(s.get("id"))
                if pid in want:
                    par[pid] = clean_body("\n\n".join(x for x in (s.get("title"), s.get("selftext"))
                                                        if x and x not in ("[deleted]", "[removed]")))
        df = df.with_columns(pl.col("parent_id").replace_strict(par, default=None, return_dtype=pl.String)
                             .alias("parent_text"), pl.lit("usuk").alias("origin"))
        df.write_parquet(out)
        print(f"{sub}: {n} comments, {int(df['sarcasm'].sum())} /s, {df.height} candidates, "
              f"{len(par)}/{len(want)} parents", flush=True)


def _convokit_one(zpath: Path) -> str:
    sub = zpath.name.removesuffix(".corpus.zip")
    out = CAND / f"convokit_{sub}.parquet"
    if out.exists():
        return f"{sub}: cached"
    rng = random.Random(sub)
    rows, n = [], 0
    with zipfile.ZipFile(zpath) as z:
        name = next(x for x in z.namelist() if x.endswith("utterances.jsonl"))
        with z.open(name) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                u = orjson.loads(line)
                if not u.get("reply_to"):
                    continue
                n += 1
                meta = u.get("meta") or {}
                pref = "t3_" if u["reply_to"] == u.get("root") else "t1_"
                r = _candidate({"id": u["id"], "body": u.get("text"), "author": u.get("user"),
                                "parent_id": pref + u["reply_to"], "created_utc": u.get("timestamp"),
                                "subreddit": meta.get("subreddit") or sub, "score": meta.get("score")}, rng)
                if r:
                    rows.append(r)
        df = pl.DataFrame(rows, schema={"id": pl.String, "text": pl.String, "sarcasm": pl.Int8, "parent_id": pl.String,
                                        "created_utc": pl.Int64, "subreddit": pl.String, "score": pl.Int64},
                          strict=False)
        want = set(df["parent_id"].drop_nulls().to_list())
        par = {}
        cname = next((x for x in z.namelist() if x.endswith("conversations.json")), None)
        titles = orjson.loads(z.read(cname)) if cname else {}
        with z.open(name) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                u = orjson.loads(line)
                is_post = not u.get("reply_to")
                pid = ("t3_" if is_post else "t1_") + u["id"]
                if pid not in want:
                    continue
                text = u.get("text") if u.get("text") not in ("[deleted]", "[removed]") else ""
                if is_post:
                    text = "\n\n".join(x for x in ((titles.get(u["id"]) or {}).get("title"), text) if x)
                if text:
                    par[pid] = clean_body(text)
        del titles
    df = df.with_columns(pl.col("parent_id").replace_strict(par, default=None, return_dtype=pl.String)
                         .alias("parent_text"), pl.lit("convokit").alias("origin"))
    df.write_parquet(out)
    return f"{sub}: {n} comments, {int(df['sarcasm'].sum())} /s, {df.height} candidates, {len(par)}/{len(want)} parents"


def convokit_candidates(workers: int = 4) -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    zips = sorted(CONVOKIT.glob("*.corpus.zip")) if CONVOKIT.exists() else []
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(_convokit_one, zips):
            print(msg, flush=True)


def arctic_candidates() -> pl.DataFrame:
    frames = []
    rng = random.Random(0)
    for pos in sorted(ARCTIC.glob("*.pos.jsonl")):
        sub = pos.name.removesuffix(".pos.jsonl")
        rows = []
        for kind in ("pos", "neg"):
            p = ARCTIC / f"{sub}.{kind}.jsonl"
            if not p.exists():
                continue
            for line in p.open():
                c = orjson.loads(line)
                body = clean_body(c.get("body"))
                if not _ok(body, c.get("author")):
                    continue
                if has_s_tag(body):
                    text, y = strip_s_tag(body), 1
                elif kind == "neg" and "/s" not in body.lower():
                    text, y = body.strip(), 0
                else:
                    continue
                if text and len(text) >= 8:
                    rows.append({"id": c["id"], "text": text, "sarcasm": y, "parent_id": c.get("parent_id"),
                                 "created_utc": int(c.get("created_utc") or 0), "subreddit": c.get("subreddit") or sub,
                                 "score": c.get("score")})
        if not rows:
            continue
        df = pl.DataFrame(rows, strict=False).unique("id")
        par = {}
        pp = ARCTIC / f"{sub}.parents.jsonl"
        if pp.exists():
            for line in pp.open():
                r = orjson.loads(line)
                if r.get("text") and r["text"] not in ("[deleted]", "[removed]"):
                    par[r["pid"]] = r["text"]
        df = df.with_columns(pl.col("parent_id").replace_strict(par, default=None, return_dtype=pl.String)
                             .alias("parent_text"), pl.lit("arctic").alias("origin"))
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def bulk_candidates() -> pl.DataFrame:
    """Whole days: /s comments + sampled untagged ones; parents looked up among the same days' items."""
    frames = []
    for d in sorted(p for p in BULK.iterdir() if p.is_dir()) if BULK.exists() else []:
        rng = random.Random(d.name)
        rows, par = [], {}
        for f in sorted(d.glob("*.jsonl")):
            if f.name.endswith(".posts.jsonl"):
                for line in f.open():
                    s = orjson.loads(line)
                    par["t3_" + s["id"]] = clean_body("\n\n".join(
                        x for x in (s.get("title"), s.get("selftext")) if x and x not in ("[deleted]", "[removed]")))
                continue
            for line in f.open():
                c = orjson.loads(line)
                if c.get("body") not in (None, "[deleted]", "[removed]"):
                    par["t1_" + c["id"]] = clean_body(c["body"])
                r = _candidate(c, rng)
                if r:
                    rows.append(r)
        if not rows:
            continue
        df = pl.DataFrame(rows, schema={"id": pl.String, "text": pl.String, "sarcasm": pl.Int8, "parent_id": pl.String,
                                        "created_utc": pl.Int64, "subreddit": pl.String, "score": pl.Int64},
                          strict=False)
        want = set(df["parent_id"].drop_nulls().to_list())
        par = {k: v for k, v in par.items() if k in want}
        frames.append(df.with_columns(pl.col("parent_id").replace_strict(par, default=None, return_dtype=pl.String)
                                      .alias("parent_text"), pl.lit("arctic_bulk").alias("origin")))
        print(f"bulk {d.name}: {len(list(d.glob('*.posts.jsonl')))} days, {int(df['sarcasm'].sum())} /s", flush=True)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def build() -> None:
    frames = [pl.read_parquet(p) for p in sorted(CAND.glob("usuk_*.parquet")) + sorted(CAND.glob("convokit_*.parquet"))]
    for extra in (arctic_candidates(), bulk_candidates()):
        if extra.height:
            frames.append(extra)
    df = pl.concat(frames, how="diagonal_relaxed").with_columns(
        pl.from_epoch("created_utc").dt.year().alias("year"))
    # the same comment can come from several origins: prefer dumps over the API
    rank = {"usuk": 0, "convokit": 1, "arctic_bulk": 2, "arctic": 3}
    df = df.sort(pl.col("origin").replace_strict(rank, default=9)).unique("id", keep="first")
    df = df.with_columns(pl.col("subreddit").str.to_lowercase().replace_strict(VAR, default="unknown")
                         .alias("variety_guess"))
    # negatives: NEG_RATIO per positive within subreddit x year
    pos = df.filter(pl.col("sarcasm") == 1)
    quota = pos.group_by("subreddit", "year").agg((pl.len() * NEG_RATIO).alias("q"))
    neg = (df.filter(pl.col("sarcasm") == 0).join(quota, on=["subreddit", "year"], how="inner")
           .with_columns(pl.int_range(pl.len()).shuffle(seed=7).over("subreddit", "year").alias("r"))
           .filter(pl.col("r") < pl.col("q")).drop("q", "r"))
    df = pl.concat([pos, neg], how="diagonal_relaxed")
    # SARC (2009-2017) covers some of the same subreddits: drop exact overlaps so the sets can be stacked
    sarc = ROOT_SARC / "sarc_ukau.parquet"
    if sarc.exists():
        from dedupe import normalise
        seen = set(pl.read_parquet(sarc, columns=["text"]).select(normalise(pl.col("text")))["text"].to_list())
        n0 = df.height
        df = df.filter(~normalise(pl.col("text")).is_in(list(seen)))
        print(f"dropped {n0 - df.height} rows already in sarc_ukau", flush=True)
    print(df.group_by("variety_guess", "sarcasm").len().sort("variety_guess", "sarcasm"), flush=True)
    print(df.group_by("subreddit").agg(pl.len(), pl.col("sarcasm").sum()).sort("len", descending=True).head(40),
          flush=True)
    from dedupe import Matcher, reference_texts
    matcher = Matcher(reference_texts())
    for var, name in (("en-UK", "reddit_s_uk"), ("en-AU", "reddit_s_au")):
        part = finalize(df.filter(pl.col("variety_guess") == var), dataset=name, source="reddit",
                        license=LICENSE, label_quality="weak")
        write(part, name, url="https://arctic-shift.photon-reddit.com ; https://huggingface.co/datasets/alujjdnd/Reddit-US-UK",
              license=LICENSE, matcher=matcher,
              notes=f"Positives = author /s tag (stripped); negatives = untagged comments, same subreddit x year, "
                    f"{NEG_RATIO}:1. Weak labels both ways. parent_text for both classes where available. "
                    "Covers the BESSTIE collection subreddits/years: re-run dedupe.py --test on the test CSV.")


if __name__ == "__main__":
    cmds = sys.argv[1:] or ["zst", "convokit", "build"]
    if "zst" in cmds:
        zst_candidates()
    if "convokit" in cmds:
        convokit_candidates()
    if "build" in cmds:
        build()
