"""Download ConvoKit subreddit corpora (Cornell; full subreddit histories up to Oct 2018) for UK/AU subs.

  uv run python experiments/data/convokit_fetch.py [sub ...]

Zips land in data/external/raw/convokit/<sub>.corpus.zip; reddit_s_build.py reads them directly.
Source: https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/ (ConvoKit, Chang et al. 2020).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "raw" / "convokit"
BASE = "https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/"

# Not in the HF UK-city dump (CasualUK, london, ... are taken from there).
SUBS = ["australia", "unitedkingdom", "ukpolitics", "AustralianPolitics", "melbourne", "sydney", "brisbane",
        "perth", "Adelaide", "canberra", "AusFinance", "straya", "circlejerkaustralia", "aussie", "queensland",
        "tasmania", "AskUK", "britishproblems", "Scotland", "LabourUK", "GreatBritishMemes", "UKPersonalFinance",
        "britishpolitics", "Wales", "northernireland", "bristol", "edinburgh", "tories", "england"]


def _get(url: str) -> str:
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except OSError as e:  # the Cornell server intermittently times out
            print(f"retry {attempt} {url}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subs = sys.argv[1:] or SUBS
    dirs = re.findall(r'href="([^"]+~-~[^"]+/)"', _get(BASE))
    for sub in subs:
        dest = OUT / f"{sub}.corpus.zip"
        if dest.exists():
            continue
        cands = [d for d in dirs if (lambda a, b: a <= sub <= b)(*urllib.parse.unquote(d.rstrip("/")).split("~-~"))]
        url = None
        for d in cands:
            if f'href="{sub}.corpus.zip"' in _get(BASE + d):
                url = BASE + d + f"{sub}.corpus.zip"
                break
        if not url:
            print(f"{sub}: not found in {cands}", flush=True)
            continue
        tmp = dest.with_suffix(".part")
        subprocess.run(["wget", "-q", "-c", "--tries=10", "--timeout=60", "-O", str(tmp), url], check=True)
        tmp.rename(dest)
        print(f"{sub}: {dest.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
