"""Paper figure: learning curves (official score vs training rows) for the three lead member families.

DeBERTa points: single-seed runs enc_deb_l_lc{25,50,75}_s{13,14} and, at 100%, the 10 single seeds
(enc_deb_l_s13..17, enc_deb_l_r13..17); probes: results/paper/learning_curve_probes.csv.
Palette: validated categorical slots 1-3 (dataviz reference palette); every line is direct-labelled
(aqua is < 3:1 against the surface) and has its own marker shape for grayscale print.

  uv run python experiments/classical/plot_paper.py   -> paper/fig_learning_curve.pdf, results/paper/learning_curve_deberta.csv
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from common import ROOT, pool, quick  # noqa: E402
from alta.common import PREDS  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N_FULL = 5030  # mean training rows per outer fold (4/5 of 6,288)


def run_score(run: str, df) -> float | None:
    p = PREDS / f"{run}.parquet"
    if not p.exists():
        return None
    o = df.select("uid").join(pl.read_parquet(p).filter(pl.col("split") == "oof"), on="uid", how="left",
                              maintain_order="left")
    if o["p_sent"].null_count():
        return None
    return quick(o["p_sent"].to_numpy(), o["p_sarc"].to_numpy(), specs=("segment+ukg",))["segment+ukg"]["score"]


def deberta_curve(df) -> pl.DataFrame:
    rows = []
    for f in (25, 50, 75):
        for s in (13, 14):
            sc = run_score(f"enc_deb_l_lc{f}_s{s}", df)
            if sc is not None:
                rows.append({"frac": f / 100, "seed": s, "score": sc})
    for s in range(13, 18):
        for pre in ("s", "r"):
            sc = run_score(f"enc_deb_l_{pre}{s}", df)
            if sc is not None:
                rows.append({"frac": 1.0, "seed": s if pre == "s" else 100 + s, "score": sc})
    return pl.DataFrame(rows)


def main() -> None:
    df = pool()
    deb = deberta_curve(df)
    deb.write_csv(ROOT / "results" / "paper" / "learning_curve_deberta.csv")
    probes = pl.read_csv(ROOT / "results" / "paper" / "learning_curve_probes.csv")
    series = [("DeBERTa (1 seed)", BLUE, "o", deb),
              ("9B LLM probe", ORANGE, "s", probes.filter(pl.col("member") == "llm")),
              ("e5+me5 probe", AQUA, "^", probes.filter(pl.col("member") == "e5"))]
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for name, col, mk, d in series:
        if d.height == 0:
            continue
        g = d.group_by("frac").agg(pl.col("score").mean().alias("m"), pl.col("score").std().fill_null(0).alias("sd"),
                                   pl.len().alias("n")).sort("frac")
        x = g["frac"].to_numpy() * N_FULL
        ax.errorbar(x, g["m"], yerr=g["sd"], color=col, marker=mk, ms=4, lw=1.5, capsize=2, elinewidth=0.8)
        ax.annotate(name, (x[-1], g["m"][-1]), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2500, 5000])
    ax.set_xticklabels(["500", "1k", "2.5k", "5k"])
    ax.set_xlim(400, 5200 * 2.3)
    ax.set_xlabel("training rows (log scale)", color=INK2)
    ax.set_ylabel("official score (OOF)", color=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    out = ROOT / "paper" / "fig_learning_curve.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(deb.group_by("frac").agg(pl.col("score").mean().round(4), pl.len()).sort("frac"))
    print("saved", out)


if __name__ == "__main__":
    main()
