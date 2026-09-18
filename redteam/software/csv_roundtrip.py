"""Attack 1a: does pandas (scorer) read our polars-written answer.csv row-for-row identically?

Writes answer-like frames with nasty texts via the same code path as submission._write
(polars write_csv defaults), then reads with polars and with pandas (in a subprocess, pandas
is not a project dependency) and compares row counts and variety sequences.
"""

import json
import subprocess
import sys
from pathlib import Path

import polars as pl

OUT = Path(__file__).parent / "out_roundtrip"
OUT.mkdir(exist_ok=True)

NASTY = {
    "plain": "hello world",
    "comma": "a, b, c",
    "quote": 'he said "hi"',
    "lead_quote": '"quoted start',
    "newline": "line1\nline2",
    "crlf": "line1\r\nline2",
    "bare_cr": "line1\rline2",
    "tab": "a\tb",
    "nul": "a\x00b",
    "unicode": "café — naïve 😀   sep",
    "NA": "NA",
    "null": "null",
    "None": "None",
    "nan": "nan",
    "empty": "",
    "space": " ",
    "hash": "# comment?",
    "backslash": 'a\\"b',
    "only_quote": '"',
    "trailing_bs": "end\\",
    "numeric": "12345",
}


def main():
    results = {}
    for name, txt in NASTY.items():
        rows = [("google", "en-AU", "before", 1, 0), ("reddit", "en-UK", txt, 0, 1),
                ("google", "en-UK", "after", 1, 1)]
        df = pl.DataFrame(rows, schema=["source", "variety", "text", "sentiment", "sarcasm"], orient="row")
        p = OUT / f"{name}.csv"
        df.write_csv(p)  # same as submission._write
        back = pl.read_csv(p)
        results[name] = {"polars_rows": back.height,
                         "polars_text_ok": back["text"].to_list() == df["text"].to_list()}
    code = r"""
import json, sys, pandas as pd
out = {}
for name in json.loads(sys.argv[1]):
    try:
        d = pd.read_csv(sys.argv[2] + '/' + name + '.csv')
        out[name] = {'pandas_rows': len(d), 'varieties': d['variety'].tolist(),
                     'sent': d['sentiment'].tolist(), 'dtype': str(d['sentiment'].dtype)}
    except Exception as e:
        out[name] = {'pandas_error': repr(e)[:200]}
print(json.dumps(out))
"""
    py = sys.argv[1] if len(sys.argv) > 1 else None
    cmd = ["uv", "run", "--no-project", "--with", "pandas", "python", "-c", code,
           json.dumps(list(NASTY)), str(OUT)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    pdres = json.loads(r.stdout) if r.returncode == 0 else {"_err": r.stderr[-800:]}
    for name in NASTY:
        pr = pdres.get(name, {})
        ok = pr.get("pandas_rows") == 3 and pr.get("varieties") == ["en-AU", "en-UK", "en-UK"]
        print(f"{name:12} polars={results[name]} pandas={'OK' if ok else pr}")


if __name__ == "__main__":
    main()
