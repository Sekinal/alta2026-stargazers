"""Build external labelled sentiment corpora for intermediate pre-fine-tuning.

Run on the remote (CPU only), from the project root:
    uv run --with huggingface_hub --with pyahocorasick python experiments/data/sentiment_build.py [--only a,b] [--cap N]

For each dataset, writes to data/external/sentiment/:
    <name>.parquet           binary rows after cleaning, dedupe and cap (exactly label-balanced, domain-stratified)
    <name>__full.parquet     every binary row after dedupe, before the cap (so any other sample is recoverable)
    <name>__neutral.parquet  rows with no binary label (neutral / mixed / ambiguous / 3 stars), sentiment null
and updates MANIFEST.json. Columns: text, sentiment, orig_label, source, variety_guess, domain,
dataset, license, label_quality.

Never add BESSTIE or anything from its authors (competition rule: no resource may contain test labels).
"""

from __future__ import annotations

import argparse
import ast
import gzip
import html
import io
import json
import re
import sys
import time
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedupe_sentiment import build_index, norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "sentiment"
RAW = OUT / "raw"
SEED = 0
COLS = ["text", "sentiment", "orig_label", "source", "variety_guess", "domain", "dataset", "license", "label_quality"]

# --------------------------------------------------------------------------- text cleaning
# The competition text is ASCII-only, with no URLs, emoji or HTML entities; make external text look the same.
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION = re.compile(r"(?<!\w)@\w+")
_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                        "…": "...", " ": " "})
_WS = re.compile(r"\s+")


def clean(s: str | None) -> str:
    if not s:
        return ""
    s = html.unescape(s).replace("\\n", " ").replace("\\\"", '"')
    s = _URL.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = s.translate(_TRANS)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return _WS.sub(" ", s).strip()


_EN_WORDS = set(
    "the a an and or but is are was were be been it this that i you he she we they my your our not no "
    "to of in on for with at by from very so good great bad nice place food service staff friendly".split()
)


def looks_english(s: str) -> bool:
    toks = re.findall(r"[a-z']+", s.lower())
    if not toks:
        return False
    hits = sum(t in _EN_WORDS for t in toks)
    return hits >= 1 if len(toks) <= 4 else hits / len(toks) >= 0.12


# --------------------------------------------------------------------------- download helpers
def hf(repo: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type="dataset"))


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def frame(text, sentiment, orig, *, source, variety, domain, dataset, license, quality) -> pl.DataFrame:
    df = pl.DataFrame({"text": text, "sentiment": sentiment, "orig_label": [str(o) for o in orig]},
                      schema_overrides={"sentiment": pl.Int8})
    for k, v in {"source": source, "variety_guess": variety, "domain": domain}.items():
        df = df.with_columns(pl.Series(k, v, dtype=pl.String) if isinstance(v, list) else pl.lit(v).alias(k))
    return df.with_columns(pl.lit(dataset).alias("dataset"), pl.lit(license).alias("license"),
                           pl.lit(quality).alias("label_quality")).select(COLS)


def stars(r) -> int | None:
    return 0 if r <= 2 else 1 if r >= 4 else None


# --------------------------------------------------------------------------- sources
# Each returns (frame, meta). sentiment null = no binary label (goes to __neutral).
GOEMO_POS = {"admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love",
             "optimism", "pride", "relief"}
GOEMO_NEG = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief",
             "nervousness", "remorse", "sadness"}
GOEMO_NAMES = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
               "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
               "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
               "remorse", "sadness", "surprise", "neutral"]
AU_SUBS = {"australia", "sydney", "melbourne", "brisbane", "perth", "adelaide", "canberra", "tasmania", "straya",
           "ausfinance", "australianpolitics", "afl", "nrl", "queensland", "newcastle", "aussie"}
# UK football-club subs are UK-leaning but not reliably UK authors; variety_guess stays a guess
UK_SUBS = {"unitedkingdom", "casualuk", "ukpolitics", "london", "britishproblems", "scotland", "wales", "manchester",
           "premierleague", "reddevils", "gunners", "coys", "liverpoolfc", "chelseafc", "lufc", "glasgow", "edinburgh",
           "britishsuccess", "askuk", "brexit"}


def go_emotions():
    simp = pl.concat([pl.read_parquet(hf("google-research-datasets/go_emotions", f"simplified/{s}-00000-of-00001.parquet"))
                      for s in ("train", "validation", "test")])
    raw = pl.read_parquet(hf("google-research-datasets/go_emotions", "raw/train-00000-of-00001.parquet"),
                          columns=["id", "subreddit"]).unique("id")
    df = simp.join(raw, on="id", how="left")
    sent, orig = [], []
    for labs in df["labels"].to_list():
        names = {GOEMO_NAMES[i] for i in labs}
        p, n = bool(names & GOEMO_POS), bool(names & GOEMO_NEG)
        sent.append(1 if p and not n else 0 if n and not p else None)
        orig.append("|".join(sorted(names)))
    subs = [(s or "unknown").lower() for s in df["subreddit"].to_list()]
    var = ["en-AU" if s in AU_SUBS else "en-UK" if s in UK_SUBS else "unknown"
           for s in subs]
    out = frame(df["text"].to_list(), sent, orig, source="reddit", variety=var, domain=subs,
                dataset="go_emotions", license="Apache-2.0",
                quality="human; emotion labels kept when >=2 of 3-5 raters agree; pos/neg via Ekman-style grouping")
    return out, {"url": "https://huggingface.co/datasets/google-research-datasets/go_emotions",
                 "recommended_use": "core Reddit sentiment source; pool-reddit model AUC 0.77 on the binary labels. Neutral "
                                    "bucket scores like average pool reddit (26% predicted positive; questions 17%), so if "
                                    "used, soft-label it at the segment prior (~0.3), not as positive. Text has [NAME] placeholders",
                 "neutral_handling": "positive emotions only -> 1; negative emotions only -> 0; neutral-only, "
                                     "ambiguous-only (surprise/realization/curiosity/confusion) and mixed pos+neg -> "
                                     "__neutral (orig_label keeps the emotion set)"}


def tweet_eval():
    df = pl.concat([pl.read_parquet(hf("cardiffnlp/tweet_eval", f"sentiment/{s}-00000-of-00001.parquet"))
                    for s in ("train", "validation", "test")])
    m = {0: 0, 1: None, 2: 1}
    out = frame(df["text"].to_list(), [m[x] for x in df["label"]], [["negative", "neutral", "positive"][x] for x in df["label"]], source="twitter", variety="unknown",
                domain="tweets", dataset="tweet_eval_sentiment",
                license="unknown (SemEval-2017 Task 4 release; Twitter ToS; research use)", quality="human crowd (SemEval-2017)")
    return out, {"url": "https://huggingface.co/datasets/cardiffnlp/tweet_eval (config sentiment = SemEval-2017 Task 4A)",
                 "neutral_handling": "neutral -> __neutral"}


def sentiment140():
    z = fetch("https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip", RAW / "sentiment140.zip")
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.startswith("training."))
        raw = zf.read(name)
    df = pl.read_csv(io.BytesIO(raw), has_header=False, encoding="utf8-lossy",
                     new_columns=["polarity", "id", "date", "query", "user", "text"], schema_overrides={"id": pl.String})
    out = frame(df["text"].to_list(), [0 if p == 0 else 1 for p in df["polarity"]], df["polarity"].to_list(),
                source="twitter", variety="unknown", domain="tweets", dataset="sentiment140",
                license="unknown (Go, Bhayani & Huang 2009; academic use)", quality="distant (emoticons, removed from text)")
    return out, {"url": "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip (training set only; 2009 tweets)",
                 "neutral_handling": "training set has no neutral class"}


def yelp_polarity():
    df = pl.concat([pl.read_parquet(hf("fancyzhx/yelp_polarity", f"plain_text/{s}-00000-of-00001.parquet"))
                    for s in ("train", "test")])
    out = frame(df["text"].to_list(), df["label"].to_list(), [["1-2 stars", "4-5 stars"][x] for x in df["label"]], source="review", variety="en-US",
                domain="yelp", dataset="yelp_polarity", license="Yelp Dataset Terms (academic, non-commercial)",
                quality="author star rating (1-2 neg, 4-5 pos; 3 excluded upstream)")
    return out, {"url": "https://huggingface.co/datasets/fancyzhx/yelp_polarity (Zhang et al. 2015)",
                 "neutral_handling": "3-star already excluded upstream"}


def amazon_polarity():
    parts = [hf("fancyzhx/amazon_polarity", f"amazon_polarity/train-0000{i}-of-00004.parquet") for i in range(4)]
    parts.append(hf("fancyzhx/amazon_polarity", "amazon_polarity/test-00000-of-00001.parquet"))
    # 4M rows: pre-sample 5x the cap per label to keep dedupe cheap; __full is this pre-sample
    df = pl.concat([pl.read_parquet(p) for p in parts]).sample(fraction=1.0, shuffle=True, seed=SEED)
    df = pl.concat([df.filter(pl.col("label") == k).head(500_000) for k in (0, 1)])
    text = [f"{t}. {c}" if t and not c.startswith(t) else c for t, c in zip(df["title"], df["content"])]
    out = frame(text, df["label"].to_list(), [["1-2 stars", "4-5 stars"][x] for x in df["label"]], source="review",
                variety="en-US", domain="amazon", dataset="amazon_polarity", license="Apache-2.0 (HF card)",
                quality="author star rating (1-2 neg, 4-5 pos; 3 excluded upstream)")
    return out, {"url": "https://huggingface.co/datasets/fancyzhx/amazon_polarity (Zhang et al. 2015); "
                        "random 500k/label pre-sample of 4M",
                 "neutral_handling": "3-star already excluded upstream; text = title + '. ' + body"}


def imdb():
    df = pl.concat([pl.read_parquet(hf("stanfordnlp/imdb", f"plain_text/{s}-00000-of-00001.parquet")) for s in ("train", "test")])
    text = [t.replace("<br />", " ") for t in df["text"]]
    out = frame(text, df["label"].to_list(), [["<=4/10", ">=7/10"][x] for x in df["label"]], source="review",
                variety="unknown", domain="imdb", dataset="imdb",
                license="other (Maas et al. 2011; research use, cite)", quality="author rating (<=4 neg, >=7 pos)")
    return out, {"url": "https://huggingface.co/datasets/stanfordnlp/imdb", "neutral_handling": "5-6/10 excluded upstream"}


def sst2():
    df = pl.read_parquet(hf("stanfordnlp/sst2", "data/train-00000-of-00001.parquet"))
    df = pl.concat([df, pl.read_parquet(hf("stanfordnlp/sst2", "data/validation-00000-of-00001.parquet"))])
    out = frame(df["sentence"].to_list(), df["label"].to_list(), df["label"].to_list(), source="other",
                variety="unknown", domain="movie_phrases", dataset="sst2",
                license="unknown (Socher et al. 2013; research use)", quality="human (SST phrase treebank, binarised)")
    return out, {"url": "https://huggingface.co/datasets/stanfordnlp/sst2 (GLUE; train = phrases incl. fragments)",
                 "neutral_handling": "neutral phrases excluded upstream"}


def rotten_tomatoes():
    df = pl.concat([pl.read_parquet(hf("cornell-movie-review-data/rotten_tomatoes", f"{s}.parquet"))
                    for s in ("train", "validation", "test")])
    out = frame(df["text"].to_list(), df["label"].to_list(), df["label"].to_list(), source="review", variety="unknown",
                domain="movie_snippets", dataset="rotten_tomatoes", license="unknown (Pang & Lee 2005; research use)",
                quality="critic fresh/rotten flag")
    return out, {"url": "https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes",
                 "neutral_handling": "none (binary)"}


def dynasent():
    z = fetch("https://github.com/cgpotts/dynasent/raw/main/dynasent-v1.1.zip", RAW / "dynasent-v1.1.zip")
    rows = []
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.endswith(".jsonl") and "__MACOSX" not in n:
                rnd = "r1" if "round01" in n else "r2"
                for line in zf.read(n).decode().splitlines():
                    d = json.loads(line)
                    rows.append((d["sentence"], d["gold_label"], rnd))
    m = {"positive": 1, "negative": 0}
    out = frame([r[0] for r in rows], [m.get(r[1]) for r in rows], [r[1] for r in rows], source="review",
                variety="en-US", domain=[f"yelp_sentences_{r[2]}" for r in rows], dataset="dynasent",
                license="CC-BY-4.0", quality="human, 5 raters; gold = majority (>=3/5)")
    return out, {"url": "https://github.com/cgpotts/dynasent (v1.1, rounds 1+2, all splits)",
                 "neutral_handling": "neutral, mixed and no-majority -> __neutral"}


def qut_ausvotes2025():
    d = RAW / "qut_ausvotes2025" / "2025 Federal Election Dataset"
    if not d.exists():
        z = fetch("https://data.researchdatafinder.qut.edu.au/dataset/63c54cdc-2a79-49cf-a1ef-91414a8710d5/resource/"
                  "a29b725e-dc05-4ccd-a960-5a043a8bee77/download/2025-federal-election-dataset.zip",
                  RAW / "qut_ausvotes2025" / "ds.zip")
        zipfile.ZipFile(z).extractall(z.parent)
    rd = pl.read_parquet(d / "reddit_comments.parquet")
    nt = pl.read_parquet(d / "newstalk_comments.parquet")
    df = pl.concat([
        rd.select(pl.col("comment_body").alias("text"), "support", "party", pl.lit("reddit").alias("src"),
                  ("r/" + pl.col("subreddit").str.to_lowercase()).alias("dom")),
        nt.select(pl.col("comment_body").alias("text"), "support", "party", pl.lit("other").alias("src"),
                  ("newstalk/" + pl.col("source")).alias("dom")),
    ])
    m = {"OPPOSE": 0, "SUPPORT": 1}
    sup = df["support"].to_list()
    out = frame(df["text"].to_list(), [m.get(x) for x in sup],
                [f"{x}:{p}" for x, p in zip(sup, df["party"])], source=df["src"].to_list(), variety="en-AU",
                domain=df["dom"].to_list(), dataset="qut_ausvotes2025", license="CC-BY-NC-4.0 (QUT Digital Observatory)",
                quality="LLM-labelled party stance (support/oppose), used as a sentiment proxy")
    return out, {"url": "https://researchdatafinder.qut.edu.au/display/n22735 (2025 Australian Federal Election "
                        "NewsTalk/Reddit dataset; Reddit = 20 AU political/city subreddits, NewsTalk = AU news-site comments)",
                 "recommended_use": "text only (domain-adaptive MLM / pseudo-labelling). Stance labels are a poor proxy for the "
                                    "task's sentiment: a pool-reddit TF-IDF model gets AUC 0.56 on them and scores even SUPPORT "
                                    "comments as negative (7.8% predicted positive)",
                 "neutral_handling": "support OPPOSE -> 0, SUPPORT -> 1 (stance toward a party, not overall tone); "
                                     "NONE/null/other -> __neutral",
                 "stratify": ["source", "domain"]}


_GPS = re.compile(r"'gps': \[(-?[\d.]+), (-?[\d.]+)\]")
_PID = re.compile(r"'gPlusPlaceId': u?'(\d+)'")
_ADDR = re.compile(r"'address': \[([^\]]*)\]")


def _country(line: str) -> str | None:
    a = _ADDR.search(line)
    g = _GPS.search(line)
    if not a or not g:
        return None
    addr, (lat, lon) = a.group(1), map(float, g.groups())
    if re.search(r"Australia'\s*$", addr) and -45 < lat < -9 and 112 < lon < 155:
        return "en-AU"
    if re.search(r"United Kingdom'\s*$", addr) and 49.5 < lat < 61 and -8.8 < lon < 2:
        return "en-UK"
    return None


def google_local_2018():
    d = RAW / "googlelocal2018"
    places = {}
    with gzip.open(d / "places.clean.json.gz", "rt") as f:
        for line in f:
            c = _country(line)
            if c:
                places[_PID.search(line).group(1)] = c
    print(f"  google_local: {len(places)} AU/UK places", flush=True)
    text, rating, var, cat = [], [], [], []
    with gzip.open(d / "reviews.clean.json.gz", "rt") as f:
        for line in f:
            m = _PID.search(line)
            if not m or m.group(1) not in places or "'reviewText': None" in line:
                continue
            r = ast.literal_eval(line)
            t = r.get("reviewText")
            if not t or r.get("rating") is None:
                continue
            text.append(t)
            rating.append(float(r["rating"]))
            var.append(places[m.group(1)])
            cats = r.get("categories") or []
            cat.append(cats[0] if cats else "unknown")
    out = frame(text, [stars(x) for x in rating], [f"{x:g} stars" for x in rating], source="review", variety=var,
                domain=cat, dataset="google_local_2018_au_uk",
                license="UCSD McAuley lab public dataset (research use, cite He, Kang & McAuley 2017)",
                quality="author star rating (1-2 neg, 4-5 pos); variety_guess = place country, not reviewer")
    return out, {"url": "https://mcauleylab.ucsd.edu/public_datasets/data/googlelocal/ (reviews/places.clean.json.gz, "
                        "Google Local 2018 worldwide; AU/UK places by address country + GPS box)",
                 "recommended_use": "AU/UK review robustness; 3-star rows score 69% positive under a pool-reddit model, "
                                    "so the task would likely call most mixed reviews positive",
                 "neutral_handling": "3 stars -> __neutral; English-only filter (stopword heuristic)",
                 "hold_until_test_dedupe": True,
                 "hold_reason": "5 exact matches with the pool show BESSTIE's Google reviews overlap this source, so "
                                "hidden test reviews are probably in it with star ratings (a proxy for test labels). "
                                "Excluded from the labelled mix; re-run dedupe_sentiment.py --against the test file "
                                "before any use",
                 "stratify": ["variety_guess", "domain"]}


SOURCES = {
    "go_emotions": (go_emotions, 300_000),
    "tweet_eval_sentiment": (tweet_eval, 300_000),
    "google_local_2018_au_uk": (google_local_2018, 300_000),
    "dynasent": (dynasent, 300_000),
    "sst2": (sst2, 300_000),
    "rotten_tomatoes": (rotten_tomatoes, 300_000),
    "qut_ausvotes2025": (qut_ausvotes2025, 300_000),
    "imdb": (imdb, 300_000),
    "yelp_polarity": (yelp_polarity, 150_000),
    "amazon_polarity": (amazon_polarity, 150_000),
    "sentiment140": (sentiment140, 100_000),
}


SKIPPED = {
    "BESSTIE and anything by its authors (e.g. surrey-nlp/alignment-australian-final, -british-final)":
        "competition rule: may contain the test labels",
    "argilla/tripadvisor-hotel-reviews (CC-BY-NC-4.0)": "text is lowercased and stopword-stripped, unlike natural text",
    "UCSD Google Local 2021": "US-only; the 2018 worldwide release is used instead",
    "Amazon Reviews 2023 (McAuley-Lab)": "US storefront only, redundant with amazon_polarity",
    "SemEval-2017 Task 4 / cardiffnlp tweet_sentiment_multilingual (english)": "same data as tweet_eval sentiment",
    "silvia-casola/BREXIT": "hate-speech/offensiveness labels, not sentiment; 1.1k tweets",
    "HF 'reddit sentiment' sets (Nech-C/*, hnam25/*, arrmlet/political-social-reddit-us-sentiment-v1, ...)":
        "tiny, model-labelled or US/off-domain",
    "Trustpilot corpus (Hovy et al. 2015, UK reviews)": "no public download found in time",
    "Kaggle datasets (e.g. tweet_sentiment_extraction, Reddit_Data)": "Kaggle competition/ToS terms or TextBlob-derived labels",
}


# --------------------------------------------------------------------------- pipeline
def cap_balanced(df: pl.DataFrame, cap: int, strata: list[str]) -> pl.DataFrame:
    """Equal rows per label (min(cap/2, minority count)); within a label, sample strata proportionally."""
    quota = min(cap // 2, *(df.filter(pl.col("sentiment") == k).height for k in (0, 1)))
    parts = []
    for k in (0, 1):
        sub = df.filter(pl.col("sentiment") == k)
        if sub.height > quota:
            frac = quota / sub.height
            sub = (sub.with_columns(pl.int_range(pl.len()).shuffle(seed=SEED + k).over(strata).alias("_r"),
                                    pl.len().over(strata).alias("_n"))
                   .filter(pl.col("_r") < (pl.col("_n") * frac).round().cast(pl.Int64)).drop("_r", "_n"))
        parts.append(sub)
    return pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED)


def summarise(df: pl.DataFrame) -> dict:
    return {
        "rows": df.height,
        "pos_rate": round(float(df["sentiment"].mean()), 4) if df.height else None,
        "variety": dict(df.group_by("variety_guess").len().sort("len", descending=True).iter_rows()),
        "median_chars": int(df["text"].str.len_chars().median()) if df.height else None,
        "top_domains": dict(df.group_by("domain").len().sort("len", descending=True).head(10).iter_rows()),
    }


def process(name: str, idx, cap: int, manifest: dict) -> None:
    t0 = time.time()
    fn, default_cap = SOURCES[name]
    df, meta = fn()
    n_raw = df.height
    df = df.with_columns(pl.col("text").map_elements(clean, return_dtype=pl.String))
    df = df.filter(pl.col("text").str.len_chars() >= 2)
    if name == "google_local_2018_au_uk":
        df = df.filter(pl.col("text").map_elements(looks_english, return_dtype=pl.Boolean))
    n_clean = df.height
    df = df.with_columns(pl.col("text").map_elements(norm, return_dtype=pl.String).alias("_n"))
    df = df.filter(pl.col("_n") != "")
    # internal dedupe: drop every copy of a text whose copies disagree on the label, else keep one
    conflict = df.group_by("_n").agg(pl.col("sentiment").n_unique().alias("_k"))
    df = df.join(conflict, on="_n").filter(pl.col("_k") == 1).unique("_n", keep="first", maintain_order=True)
    n_internal = df.height
    hit = idx.flag(df["text"].to_list())
    df = df.filter(pl.Series(~hit)).drop("_n", "_k")
    n_pool = int(hit.sum())
    binary, neutral = df.filter(pl.col("sentiment").is_not_null()), df.filter(pl.col("sentiment").is_null())
    strata = meta.pop("stratify", ["domain"])
    capped = cap_balanced(binary, min(cap, default_cap), strata)
    binary.write_parquet(OUT / f"{name}__full.parquet")
    capped.write_parquet(OUT / f"{name}.parquet")
    if neutral.height:
        neutral.write_parquet(OUT / f"{name}__neutral.parquet")
    lic = df["license"][0]
    manifest[name] = {
        **meta, "license": lic, "label_quality": df["label_quality"][0], "source": df["source"][0],
        "counts": {"raw": n_raw, "after_clean": n_clean, "after_internal_dedupe": n_internal,
                   "dropped_pool_dedupe": n_pool, "binary_full": binary.height, "neutral": neutral.height,
                   "capped": capped.height},
        "full_pos_rate": round(float(binary["sentiment"].mean()), 4),
        "capped_summary": summarise(capped),
        "neutral_orig_labels": dict(neutral.group_by("orig_label").len().sort("len", descending=True).head(8).iter_rows())
        if neutral.height else {},
        "files": [f"data/external/sentiment/{name}.parquet", f"data/external/sentiment/{name}__full.parquet"]
        + ([f"data/external/sentiment/{name}__neutral.parquet"] if neutral.height else []),
    }
    print(f"  {name}: raw {n_raw} clean {n_clean} dedup {n_internal} pool-drop {n_pool} binary {binary.height} "
          f"neutral {neutral.height} capped {capped.height} ({time.time() - t0:.0f}s)", flush=True)


def update_manifest(name: str, entry: dict | None, error: str | None = None) -> None:
    """Re-read and merge per dataset, so parallel runs (--only) don't clobber each other."""
    mpath = OUT / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    failed = {k: v for k, v in manifest.setdefault("_failed", {}).items() if k in SOURCES}
    manifest["_failed"] = failed
    manifest["_skipped"] = SKIPPED
    if entry is not None:
        manifest[name] = entry
        failed.pop(name, None)
    else:
        failed[name] = error
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--cap", type=int, default=300_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    idx = build_index()
    for name in a.only.split(",") if a.only else list(SOURCES):
        print(f"[{name}]", flush=True)
        entry: dict = {}
        try:
            process(name, idx, a.cap, entry)
        except Exception as e:  # keep going; report at the end
            print(f"  FAILED {name}: {type(e).__name__}: {e}", flush=True)
            update_manifest(name, None, f"{type(e).__name__}: {e}")
            continue
        update_manifest(name, entry[name])

if __name__ == "__main__":
    main()
