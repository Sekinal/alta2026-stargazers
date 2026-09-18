"""Build a realistic fake test phase:

  redteam/software/fake/test.csv   valid.csv rows, shuffled, labels blank, nasty texts injected,
                                   CRLF line endings + UTF-8 BOM (csv module, as a Windows/Excel export)
  redteam/software/fake/truth.csv  the same rows in the same order with the gold labels
  preds/rt_sw_e2e.{parquet,json}   OOF copied from emb_ens3_lr; test split = the OOF probability
                                   of the valid row each test row came from (so the expected answer
                                   of every test row is known exactly)

Usage: uv run python redteam/software/make_fake.py [--order-seed N] [--run NAME] [--out DIR]
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import polars as pl

from alta.common import DATA, load_pool, load_test, save_preds

HERE = Path(__file__).parent
SUFFIXES = ['\nsecond line, with comma', ' "quoted" bit', ',,,', ' café \U0001F600  ', '\r\nCRLF inside',
            ' NA', "", ' trailing backslash\\', ' ""', '\ttab']


def build(order_seed: int, run: str, out: Path, base_run: str = "emb_ens3_lr") -> None:
    out.mkdir(parents=True, exist_ok=True)
    valid = pl.read_csv(DATA / "valid.csv").with_row_index("vrow")
    rng = np.random.default_rng(order_seed)
    perm = rng.permutation(valid.height)
    test = valid[perm]
    texts = test["text"].to_list()
    for i, s in enumerate(SUFFIXES):
        texts[i * 7] = texts[i * 7] + s
    texts[3] = "NA"  # a text pandas reads as NaN
    test = test.with_columns(pl.Series("text", texts))

    def dump(frame, labels: bool, path: Path):
        buf = io.StringIO()
        w = csv.writer(buf)  # lineterminator \r\n, QUOTE_MINIMAL, like Excel/pandas-on-Windows
        w.writerow(["source", "variety", "text", "sentiment", "sarcasm"])
        for r in frame.iter_rows(named=True):
            w.writerow([r["source"], r["variety"], r["text"],
                        r["sentiment"] if labels else "", r["sarcasm"] if labels else ""])
        path.write_bytes(b"\xef\xbb\xbf" + buf.getvalue().encode())

    dump(test, False, out / "test.csv")
    dump(test, True, out / "truth.csv")
    test.select("vrow").write_parquet(out / "perm.parquet")

    # fake run: OOF from base_run, test probs = OOF probs of the originating valid row
    base = pl.read_parquet(DATA.parent / "preds" / f"{base_run}.parquet").filter(pl.col("split") == "oof")
    pool = load_pool()
    vuid = pool.filter(pl.col("origin") == "valid").sort("uid")["uid"]
    tl = load_test(out / "test.csv")  # uids as load_test assigns them
    tp = pl.DataFrame({"uid": tl["uid"], "vuid": vuid.gather(test["vrow"].to_list())}).join(
        base.select("uid", "p_sent", "p_sarc").rename({"uid": "vuid"}), on="vuid", how="left",
        maintain_order="left").drop("vuid")
    save_preds(run, base.select("uid", "p_sent", "p_sarc"), tp, meta={"redteam": True})
    print(f"wrote {out}/test.csv ({test.height} rows), run {run}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--run", default="rt_sw_e2e")
    ap.add_argument("--out", type=Path, default=HERE / "fake")
    a = ap.parse_args()
    build(a.order_seed, a.run, a.out)
