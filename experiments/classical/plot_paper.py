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




def cue_flip_figure() -> None:
    """Dumbbell chart of results/paper/cue_flip.csv (AUC of zero-shot cues for gold sarcasm among negative Reddit comments,
    per variety, 95% bootstrap CI). Blue/orange = validated categorical slots 1-2; markers differ for grayscale print."""
    d = pl.read_csv(ROOT / "results" / "paper" / "cue_flip.csv")
    cues = d.filter(pl.col("variety") == "en-AU")["cue"].to_list()
    groups = d.filter(pl.col("variety") == "en-AU")["group"].to_list()
    # layout: derision block on top, a gap, irony block below
    ypos, y = {}, 0.0
    for c, g in zip(cues, groups):
        if ypos and g != groups[cues.index(list(ypos)[-1])]:
            y += 0.9
        ypos[c] = y
        y += 1.0
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    lo_x, hi_x = 0.28, 0.86
    ax.axvspan(lo_x, 0.5, color="#f0efec", zorder=0, lw=0)
    ax.axvline(0.5, color=INK2, lw=0.8, zorder=1)
    get = {(r["variety"], r["cue"]): r for r in d.iter_rows(named=True)}
    for c in cues:
        a, u = get[("en-AU", c)], get[("en-UK", c)]
        yy = ypos[c]
        ax.plot([a["auc"], u["auc"]], [yy, yy], color="#b8b7b1", lw=1.6, zorder=2, solid_capstyle="round")
        for r, col, mk, dy in ((a, BLUE, "o", -0.13), (u, ORANGE, "s", 0.13)):
            ax.plot([r["lo"], r["hi"]], [yy + dy, yy + dy], color=col, lw=0.9, zorder=3, alpha=0.9)
            ax.plot(r["auc"], yy + dy, marker=mk, ms=5.2, color=col, mec="white", mew=0.8, zorder=4)
    ax.set_yticks([ypos[c] for c in cues])
    ax.set_yticklabels(cues, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(y + 0.3, -0.95)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("AUC for gold sarcasm, negative Reddit comments", color=INK2)
    # region captions and group headers (text in ink tokens, never series colour)
    ax.text(0.49, y - 0.05, "\u2190 predicts not sarcastic", ha="right", va="center", fontsize=6.3, color=INK2, style="italic")
    ax.text(0.51, y - 0.05, "predicts sarcastic \u2192", ha="left", va="center", fontsize=6.3, color=INK2, style="italic")
    first = {g: min(ypos[c] for c, gg in zip(cues, groups) if gg == g) for g in set(groups)}
    for g, label in (("derision", "DERISION"), ("irony", "IRONY")):
        ax.text(-0.02, first[g] - 0.62, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.3, color=INK2, weight="bold")
    # direct labels above the top row's markers + a legend above the plot (identity never by colour alone)
    top = cues[0]
    for var, dy in (("en-AU", -0.13), ("en-UK", 0.13)):
        ax.annotate(var.split("-")[1], (get[(var, top)]["auc"], ypos[top] + dy), xytext=(0, 5.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.3, color=INK, weight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, mec="white", lw=0, ms=5, label="en-AU annotator"),
                       Line2D([], [], marker="s", color=ORANGE, mec="white", lw=0, ms=5, label="en-UK annotator")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.5, handletextpad=0.2,
              columnspacing=1.2, borderaxespad=0.0)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    fig.tight_layout()
    out = ROOT / "paper" / "fig_cue_flip.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("saved", out)


if __name__ == "__main__":
    import sys
    cue_flip_figure() if "cues" in sys.argv[1:] else main()
