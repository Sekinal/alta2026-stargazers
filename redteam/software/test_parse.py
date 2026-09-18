"""Attack 1b: can polars (load_test / check_submission) and pandas (scorer, reading truth.csv)
disagree on the rows of a plausible test.csv?

Each variant starts from a few valid.csv rows with blank label columns and adds one quirk.
We report polars' row count / variety sequence via alta.common.load_test and pandas'.
"""

import json
import subprocess
from pathlib import Path

import polars as pl

from alta.common import DATA, load_test

OUT = Path(__file__).parent / "out_parse"
OUT.mkdir(exist_ok=True)

HDR = "source,variety,text,sentiment,sarcasm"
ROWS = ["google,en-AU,Nice food,,", "reddit,en-UK,\"Meh, fine\",,", "google,en-UK,Lovely,,",
        "reddit,en-AU,\"multi\nline\",,"]


def variants() -> dict[str, bytes]:
    base = "\n".join([HDR] + ROWS) + "\n"
    v = {
        "baseline": base,
        "crlf": base.replace("\n", "\r\n").replace("multi\r\nline", "multi\nline"),
        "crlf_everywhere": base.replace("\n", "\r\n"),
        "no_final_newline": base.rstrip("\n"),
        "blank_line_middle": "\n".join([HDR, ROWS[0], "", ROWS[1], ROWS[2], ROWS[3]]) + "\n",
        "blank_lines_end": base + "\n\n",
        "whitespace_line": "\n".join([HDR, ROWS[0], "   ", ROWS[1], ROWS[2], ROWS[3]]) + "\n",
        "commas_only_line": "\n".join([HDR, ROWS[0], ",,,,", ROWS[1], ROWS[2], ROWS[3]]) + "\n",
        "bare_quote_in_unquoted": "\n".join([HDR, 'google,en-AU,He said "wow" ok,,'] + ROWS[1:]) + "\n",
        "leading_quote_unclosed": "\n".join([HDR, 'google,en-AU,"wow it was good,,'] + ROWS[1:]) + "\n",
        "labels_NA_text": "\n".join([HDR, "google,en-AU,NA,,"] + ROWS[1:]) + "\n",
        "variety_padded": "\n".join([HDR, "google, en-AU,Nice,,"] + ROWS[1:]) + "\n",
        "hash_line": "\n".join([HDR, "google,en-AU,#1 pizza,,"] + ROWS[1:]) + "\n",
        "bare_cr_in_quoted": "\n".join([HDR, 'google,en-AU,"a\rb",,'] + ROWS[1:]) + "\n",
        "old_mac_cr_only": base.replace("\n", "\r").replace("multi\rline", "multi line"),
        "extra_trailing_comma": "\n".join([HDR, "google,en-AU,Nice,,,"] + ROWS[1:]) + "\n",
        "label_cols_absent": "\n".join(["source,variety,text", "google,en-AU,Nice",
                                        'reddit,en-UK,"Meh, fine"', "google,en-UK,Lovely"]) + "\n",
        "id_col_first": "\n".join(["id," + HDR] + [f"{i}," + r for i, r in enumerate(ROWS)]) + "\n",
    }
    out = {k: s.encode() for k, s in v.items()}
    out["bom"] = b"\xef\xbb\xbf" + base.encode()
    out["utf16"] = base.encode("utf-16")
    out["latin1"] = base.replace("Nice food", "Café food").encode("latin-1")
    return out


PANDAS = r"""
import json, sys, pandas as pd
out = {}
for p in json.loads(sys.argv[1]):
    try:
        d = pd.read_csv(p)
        out[p] = {'rows': len(d), 'cols': list(d.columns), 'variety': d['variety'].tolist() if 'variety' in d else None}
    except Exception as e:
        out[p] = {'error': type(e).__name__ + ': ' + str(e)[:120]}
print(json.dumps(out))
"""


def pandas_read(paths, extra=()):
    cmd = ["uv", "run", "--no-project", *extra, "--with", "pandas", "python", "-c", PANDAS,
           json.dumps([str(p) for p in paths])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    return json.loads(r.stdout)


def main():
    paths = {}
    for k, b in variants().items():
        p = OUT / f"{k}.csv"
        p.write_bytes(b)
        paths[k] = p
    pd_res = pandas_read(paths.values())
    for k, p in paths.items():
        try:
            t = load_test(p)
            pol = {"rows": t.height, "cols": t.columns[:5],
                   "variety": t["variety"].to_list() if "variety" in t.columns else None}
        except Exception as e:
            pol = {"error": type(e).__name__ + ": " + str(e).splitlines()[0][:120]}
        pdr = pd_res[str(p)]
        agree = pol.get("rows") == pdr.get("rows") and pol.get("variety") == pdr.get("variety")
        flag = "AGREE" if agree else "DISAGREE"
        print(f"{k:24} {flag:8} polars={pol}\n{'':33} pandas={pdr}")


if __name__ == "__main__":
    main()
