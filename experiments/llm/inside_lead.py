"""Fold-k check of candidate runs inside the lead: logit mean of the lead's members + the run on the
holdout rows (saved as partial runs llm_tmp_in_<run>), screened against the lead itself.

  uv run python experiments/llm/inside_lead.py RUN [RUN ...] [--lead stk_v4] [--fold 0]
"""
import argparse
import subprocess
import sys

import numpy as np
import polars as pl

sys.path.insert(0, __import__("os").getcwd())
from alta.common import load_pool, save_preds  # noqa: E402
from alta.validation.data import read_run  # noqa: E402

MEMBERS = {"stk_v4": ["enc_deb_l_x3", "cls_e5me5_lr", "llm_q35_9b_probe"],
           "stk_v5": ["enc_deb_l_x5", "cls_e5me5_lr", "llm_q35_9b_probe"]}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--lead", default="stk_v4")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--half", default="both", choices=["both", "sarc"],
                help="sarc: sentiment stays the lead's own; only the sarcasm logit mean gets the run")
a = ap.parse_args()
uids = load_pool().filter(pl.col("fold") == a.fold).select("uid")
names = []
for r in a.runs:
    frames = []
    for m in MEMBERS[a.lead] + [r]:
        p = read_run(m)[0].filter(pl.col("split") == "oof")
        frames.append(uids.join(p, on="uid", how="left"))
    for f in frames:
        assert f["p_sarc"].null_count() == 0
    out = uids.with_columns(*[pl.Series(c, 1 / (1 + np.exp(-np.mean([logit(f[c].to_numpy()) for f in frames], 0))))
                              for c in ("p_sent", "p_sarc")])
    if a.half == "sarc":
        lead = uids.join(read_run(a.lead)[0].filter(pl.col("split") == "oof"), on="uid", how="left")
        out = out.with_columns(lead["p_sent"])
    name = f"llm_tmp_in{'S' if a.half == 'sarc' else ''}_{r}"
    save_preds(name, out, meta={"kind": "blend-holdout", "members": MEMBERS[a.lead] + [r], "fold": a.fold})
    names.append(name)
subprocess.run([sys.executable, "-m", "alta.validate", "screen", a.lead, *names, *a.runs, "--fold", str(a.fold)],
               check=True)
