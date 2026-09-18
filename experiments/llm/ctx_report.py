"""UK/AU-reddit sarcasm by context slice for runs (context matched vs not; thread seen in the
row's training folds vs unseen). Hard labels from segment+ukg cross-fitted on the full OOF.

  uv run python experiments/llm/ctx_report.py llm_q35_9b_probe llm_q35_9b_ctx_probe
"""

import sys

import polars as pl
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, __import__("os").getcwd())
from alta.common import load_pool  # noqa: E402
from alta.validation import policies  # noqa: E402
from alta.validation.data import load_oof  # noqa: E402
sys.path.insert(0, "experiments/llm")
from llm_common import CONTEXT_POOL  # noqa: E402

pool = load_pool()
ctx = pl.read_parquet(CONTEXT_POOL).select("uid", "matched", "link_id")
d = pool.select("uid", "fold", "variety", "source", "sarcasm").join(ctx, on="uid", how="left")
d = d.with_columns(pl.col("matched").fill_null(False),
                   pl.col("link_id").alias("thread"))  # submission id
# thread seen in training folds: any row of the same thread in a different fold
folds_per_thread = d.filter(pl.col("thread").is_not_null()).group_by("thread").agg(pl.col("fold").unique())
d = d.join(folds_per_thread, on="thread", how="left", suffix="_t")
seen = [bool(ft is not None and any(f != k for f in ft)) for k, ft in zip(d["fold"], d["fold_t"])]
d = d.with_columns(pl.Series("thread_seen", seen))

rows = []
for run in sys.argv[1:]:
    ev = load_oof(run, pool)
    pred, _ = policies.crossfit(policies.DEFAULT_POLICY, ev)
    x = d.with_columns(pl.Series("p", ev.p["sarc"]), pl.Series("yhat", pred["sarc"].astype(int)))
    for var in ("en-UK", "en-AU"):
        r = x.filter((pl.col("variety") == var) & (pl.col("source") == "reddit"))
        for name, m in (("all", pl.lit(True)), ("matched", pl.col("matched")), ("unmatched", ~pl.col("matched")),
                        ("thread_seen", pl.col("thread_seen")),
                        ("thread_unseen", pl.col("matched") & ~pl.col("thread_seen"))):
            s = r.filter(m)
            if s.height < 20 or s["sarcasm"].n_unique() < 2:
                continue
            rows.append({"run": run, "variety": var, "slice": name, "n": s.height, "n_pos": int(s["sarcasm"].sum()),
                         "auc": roc_auc_score(s["sarcasm"], s["p"]),
                         "macro_f1": f1_score(s["sarcasm"], s["yhat"], average="macro")})
with pl.Config(tbl_rows=100, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
    print(pl.DataFrame(rows).with_columns(pl.col("auc", "macro_f1").round(3)))
