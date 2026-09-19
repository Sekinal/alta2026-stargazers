"""Paper analyses on cached Qwen3.5 features (CPU only).

  uv run python experiments/llm/paper_runs.py thread     # llm_q35_9b_probe recipe under thread folds
  uv run python experiments/llm/paper_runs.py transfer   # AU->UK / UK->AU probe transfer
  uv run python experiments/llm/paper_runs.py zs         # zero-shot vs gold disagreement (reddit sarcasm)
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_common import CONTEXT_POOL, ROOT  # noqa: E402
from make_run import fit_predict, load_multi, probe, segment_dummies  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import N_FOLDS, load_pool, save_preds  # noqa: E402
from alta.validation import policies  # noqa: E402
from alta.validation.data import load_oof  # noqa: E402
from alta.validation.metrics import components, per_fold  # noqa: E402

OUT = ROOT / "results" / "paper"
MODEL, SARC, SENT, LAYER, C = "qwen3.5-9b-nf4", "sarc_v2", "sent_v1", 1, 0.002
THREAD_FOLDS = ROOT / "data" / "external" / "context" / "thread_folds.parquet"


def features(pool: pl.DataFrame, tag: str) -> tuple[np.ndarray, np.ndarray]:
    prob, margin, hid = load_multi(MODEL, tag, "pool", pool["uid"].to_list(), LAYER)
    return np.concatenate([hid, margin, segment_dummies(pool)], 1), prob


def probe_oof(pool: pl.DataFrame) -> pl.DataFrame:
    """The llm_q35_9b_probe recipe on whatever `fold` column `pool` carries."""
    Xs, ps = features(pool, SARC)
    Xn, _ = features(pool, SENT)
    ukg = ((pool["variety"] == "en-UK") & (pool["source"] == "google")).to_numpy()
    grp = pool["variety"].cast(pl.String).to_numpy()
    o, _ = probe(pool.filter(pl.Series(~ukg)), Xs[~ukg], pool["sarcasm"].to_numpy()[~ukg], grp[~ukg],
                 C=C)
    p_sarc = ps * 1e-3
    p_sarc[~ukg] = o
    p_sent, _ = probe(pool, Xn, pool["sentiment"].to_numpy(), np.zeros(pool.height), C=C)
    return pool.select("uid").with_columns(pl.Series("p_sent", p_sent), pl.Series("p_sarc", p_sarc))


def eval_on_folds(run: str, folds: np.ndarray) -> tuple[dict, list[dict]]:
    """segment+ukg cross-fitted over the given folds; overall and per-fold components."""
    d = load_oof(run)
    d = dataclasses.replace(d, fold=folds.astype(np.int8))
    pred, _ = policies.crossfit(policies.DEFAULT_POLICY, d)
    return components(d, pred), per_fold(d, pred)


def thread():
    pool = load_pool()
    tf = pool.select("uid").join(pl.read_parquet(THREAD_FOLDS), on="uid", how="left")
    assert tf["thread_fold"].null_count() == 0
    tpool = pool.with_columns(tf["thread_fold"].cast(pl.Int8).alias("fold"))
    oof = probe_oof(tpool)
    save_preds("llm_q35_9b_probe_thr", oof, meta={
        "model": MODEL, "kind": "probe", "prompt_sarc": SARC, "prompt_sent": SENT, "layer": LAYER, "C": C,
        "shots": 0, "seed": 0, "folds": "THREAD folds (data/external/context/thread_folds.parquet: "
                                          "reddit threads by link_id never split across folds)",
        "description": "llm_q35_9b_probe recipe refit under thread folds (paper robustness run); "
                       "OOF only; cross-fitted over thread_fold, not the project folds"})
    rows = []
    for run, folds, name in (("llm_q35_9b_probe", pool["fold"].to_numpy(), "standard"),
                             ("llm_q35_9b_probe_thr", tpool["fold"].to_numpy(), "thread")):
        c, pf = eval_on_folds(run, folds)
        rows.append({"folds": name, "fold": "all", **c})
        rows += [{"folds": name, "fold": str(i), **f} for i, f in enumerate(pf)]
        ev = load_oof(run)
        ukr = (ev.variety == 1) & (ev.source == 1)
        rows[-N_FOLDS - 1]["auc_sarc_uk_reddit"] = roc_auc_score(ev.y["sarc"][ukr], ev.p["sarc"][ukr])
    df = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(4))
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / "thread_folds_llm_probe.csv")
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=200):
        print(df)


def transfer():
    pool = load_pool()
    fold = pool["fold"].to_numpy()
    var = pool["variety"].cast(pl.String).to_numpy()
    red = (pool["source"] == "reddit").to_numpy()
    ukg = (var == "en-UK") & ~red
    rows = []
    for task, tag, gold in (("sarc", SARC, "sarcasm"), ("sent", SENT, "sentiment")):
        X, _ = features(pool, tag)
        y = pool[gold].to_numpy()
        for train_v in ("en-AU", "en-UK", "both"):
            tr_mask = (var == train_v) if train_v != "both" else np.ones(len(var), bool)
            if task == "sarc":
                tr_mask &= ~ukg
            p = np.zeros(pool.height)
            if train_v == "both" and task == "sarc":  # the existing recipe: one model per variety
                for v in ("en-AU", "en-UK"):
                    m = (var == v) & ~ukg
                    for k in range(N_FOLDS):
                        tr, te = m & (fold != k), m & (fold == k)
                        p[te] = fit_predict(X[tr], y[tr], [X[te]], C)[0]
            else:
                for k in range(N_FOLDS):
                    tr = tr_mask & (fold != k)
                    te = fold == k
                    p[te] = fit_predict(X[tr], y[tr], [X[te]], C)[0]
            for eval_v in ("en-AU", "en-UK"):
                m = (var == eval_v) & (red if task == "sarc" else True)
                # threshold chosen on the other folds of the evaluated slice (honest), and at 0.5
                pred = np.zeros(m.sum(), bool)
                fm = fold[m]
                for k in range(N_FOLDS):
                    thr = policies.best_threshold(y[m][fm != k], p[m][fm != k])
                    pred[fm == k] = p[m][fm == k] >= thr
                rows.append({"task": task, "eval_rows": "reddit" if task == "sarc" else "all",
                             "train": train_v, "eval": eval_v, "n": int(m.sum()), "n_pos": int(y[m].sum()),
                             "auc": roc_auc_score(y[m], p[m]),
                             "macro_f1_crossfit_thr": f1_score(y[m], pred, average="macro"),
                             "macro_f1_at_0.5": f1_score(y[m], p[m] >= 0.5, average="macro")})
    df = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(4))
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / "transfer_llm_probe.csv")
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print(df)


def zs():
    pool = load_pool()
    ctx = pl.read_parquet(CONTEXT_POOL).select("uid", "matched", "subreddit", "is_top_level")
    p = pl.read_parquet(ROOT / "preds" / "llm_q35_4b_zs.parquet").filter(pl.col("split") == "oof")
    d = (pool.filter(pl.col("source") == "reddit").select("uid", "variety", "sarcasm")
         .join(p.select("uid", "p_sarc"), on="uid").join(ctx, on="uid", how="left")
         .with_columns((pl.col("p_sarc") >= 0.5).cast(pl.Int8).alias("zs"),
                       pl.col("variety").cast(pl.String),
                       pl.when(pl.col("matched")).then(pl.col("is_top_level").cast(pl.String))
                       .otherwise(pl.lit("unmatched")).alias("top_level"),
                       pl.when(pl.col("matched")).then(pl.col("subreddit")).otherwise(pl.lit("unmatched"))
                       .alias("subreddit")))

    def summary(g: pl.DataFrame) -> dict:
        y, z = g["sarcasm"].to_numpy(), g["zs"].to_numpy()
        return {"n": g.height, "gold_rate": y.mean(), "zs_rate": z.mean(), "agree": (y == z).mean(),
                "zs_yes_gold_no": ((z == 1) & (y == 0)).mean(), "zs_no_gold_yes": ((z == 0) & (y == 1)).mean(),
                "kappa": cohen_kappa_score(y, z) if len(set(y)) > 1 or len(set(z)) > 1 else float("nan"),
                "auc": roc_auc_score(y, g["p_sarc"]) if len(set(y)) > 1 else float("nan")}

    rows = []
    for v in ("en-AU", "en-UK"):
        dv = d.filter(pl.col("variety") == v)
        rows.append({"variety": v, "group_type": "all", "group": "all", **summary(dv)})
        for gt in ("top_level", "subreddit"):
            for (g,), part in dv.group_by(gt):
                if part.height >= 25:
                    rows.append({"variety": v, "group_type": gt, "group": str(g), **summary(part)})
    df = (pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3))
          .sort("variety", "group_type", "n", descending=[False, False, True]))
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / "zs_disagreement.csv")
    with pl.Config(tbl_rows=80, tbl_width_chars=220):
        print(df)


if __name__ == "__main__":
    {"thread": thread, "transfer": transfer, "zs": zs}[sys.argv[1]]()
