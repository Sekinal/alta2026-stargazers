"""Turn cached LLM scores into saved runs (preds/<run>.parquet via alta.common.save_preds).

Kinds
  zs     p = verbalizer probability of the sarc / sent prompt (no training at all)
  probe  logistic regression on the prompt's last-position hidden state (+ the verbalizer
         margin), cross-fitted on the shared folds: fold k is predicted by a model fitted on
         folds != k. Sarcasm: one model per variety (cues don't transfer AU<->UK); en-UK google
         sarcasm (1 positive in 2065) gets the zero-shot probability scaled to ~0. Sentiment:
         one model on all rows with segment indicators. Test rows get the mean of the 5 fold
         models.

  uv run python experiments/llm/make_run.py --run llm_q8_zs_v1 --kind zs --sarc sarc_v1 --sent sent_v1
  uv run python experiments/llm/make_run.py --run llm_q8_probe_v1 --kind probe --sarc sarc_v1 \
      --sent sent_v1 --layer 1
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_common import CACHE, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import N_FOLDS, load_pool, load_test, save_preds  # noqa: E402


def load_cache(model: str, tag: str, split: str, uids: list[str]):
    d = CACHE / model / tag / split
    parts = sorted(d.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(d)
    df = pl.concat([pl.read_parquet(p) for p in parts])
    hid = np.concatenate([np.load(p.with_suffix(".npy")) for p in parts])
    pos = {u: i for i, u in enumerate(df["uid"].to_list())}
    missing = [u for u in uids if u not in pos]
    if missing:
        raise RuntimeError(f"{tag}/{split}: {len(missing)} uids not scored, e.g. {missing[:3]}")
    idx = np.array([pos[u] for u in uids])
    return df["prob"].to_numpy()[idx], df["margin"].to_numpy()[idx], hid[idx].astype(np.float32)


def segment_dummies(df: pl.DataFrame) -> np.ndarray:
    return np.stack([(df["variety"] == "en-UK").to_numpy(), (df["source"] == "reddit").to_numpy(),
                     ((df["variety"] == "en-UK") & (df["source"] == "reddit")).to_numpy()],
                    1).astype(np.float32)


def fit_predict(X, y, Xs: list[np.ndarray], C: float) -> list[np.ndarray]:
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y)
    return [lr.predict_proba(sc.transform(Z))[:, 1] for Z in Xs]


def probe(pool, feat_pool, y, groups, feat_test=None, groups_test=None, C=0.05):
    """Cross-fitted probe; one model per group label (e.g. variety)."""
    oof = np.full(pool.height, np.nan)
    test = None if feat_test is None else np.zeros(feat_test.shape[0])
    fold = pool["fold"].to_numpy()
    for g in np.unique(groups):
        gm = groups == g
        for k in range(N_FOLDS):
            tr, te = gm & (fold != k), gm & (fold == k)
            outs = [feat_pool[te]]
            if test is not None:
                outs.append(feat_test[groups_test == g])
            res = fit_predict(feat_pool[tr], y[tr], outs, C)
            oof[te] = res[0]
            if test is not None:
                test[groups_test == g] += res[1] / N_FOLDS
    return oof, test


def build(run: str, kind: str, model: str, sarc: str, sent: str, layer: int, C: float,
          test_path: str = "", extra_meta: dict | None = None, save: bool = True):
    pool = load_pool()
    test = load_test(test_path) if test_path else None
    out = {}
    for task, tag in (("sarc", sarc), ("sent", sent)):
        prob, margin, hid = load_cache(model, tag, "pool", pool["uid"].to_list())
        if test is not None:
            prob_t, margin_t, hid_t = load_cache(model, tag, "test", test["uid"].to_list())
        if kind == "zs":
            out[task] = (prob, prob_t if test is not None else None)
            continue
        X = np.concatenate([hid[:, layer], margin[:, None], segment_dummies(pool)], 1)
        Xt = (np.concatenate([hid_t[:, layer], margin_t[:, None], segment_dummies(test)], 1)
              if test is not None else None)
        if task == "sarc":
            y = pool["sarcasm"].to_numpy()
            # UK google: 1 positive in 2065 -> keep it out of the UK model, give it a tiny prob
            grp = np.where((pool["variety"] == "en-UK").to_numpy()
                           & (pool["source"] == "google").to_numpy(), "ukg",
                           pool["variety"].cast(pl.String).to_numpy())
            grp_t = (np.where((test["variety"] == "en-UK").to_numpy()
                              & (test["source"] == "google").to_numpy(), "ukg",
                              test["variety"].cast(pl.String).to_numpy())
                     if test is not None else None)
            fit = grp != "ukg"
            o, t = probe(pool.filter(pl.Series(fit)), X[fit], y[fit], grp[fit],
                         None if test is None else Xt[grp_t != "ukg"],
                         None if test is None else grp_t[grp_t != "ukg"], C)
            oof = np.where(fit, 0.0, prob * 1e-3)
            oof[fit] = o
            tt = None
            if test is not None:
                tt = prob_t * 1e-3
                tt[grp_t != "ukg"] = t
        else:
            y = pool["sentiment"].to_numpy()
            oof, tt = probe(pool, X, y, np.zeros(pool.height), Xt,
                            None if test is None else np.zeros(test.height), C)
        out[task] = (oof, tt)
    oof_df = pool.select("uid").with_columns(pl.Series("p_sent", out["sent"][0]),
                                              pl.Series("p_sarc", out["sarc"][0]))
    test_df = None
    if test is not None:
        test_df = test.select("uid").with_columns(pl.Series("p_sent", out["sent"][1]),
                                                   pl.Series("p_sarc", out["sarc"][1]))
    meta = {"model": model, "kind": kind, "prompt_sarc": sarc, "prompt_sent": sent,
            "layer": layer, "C": C, "seed": 0,
            "recipe": {"kind": kind, "model": model, "sarc": sarc, "sent": sent,
                       "layer": layer, "C": C},
            **(extra_meta or {})}
    if save:
        save_preds(run, oof_df, test_df, meta, test_csv=test_path or None)
    return oof_df, test_df, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", choices=["zs", "probe"], required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--sarc", required=True)
    ap.add_argument("--sent", required=True)
    ap.add_argument("--layer", type=int, default=1, help="index into the cached --layers")
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--desc", default="")
    ap.add_argument("--test", default="")
    a = ap.parse_args()
    build(a.run, a.kind, a.model, a.sarc, a.sent, a.layer, a.C, a.test,
          {"description": a.desc, "shots": a.sarc.split("_")[-1] if "knn" in a.sarc else 0})
    print(json.dumps({"saved": a.run}))


if __name__ == "__main__":
    main()
