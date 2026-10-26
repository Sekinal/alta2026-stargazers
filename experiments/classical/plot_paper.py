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
    plt.rcParams.update({"font.family": "serif", "font.size": 8.3})
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
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
    ax.set_xlabel("AUC for the gold sarcasm label (model-negative comments)", color=INK2)
    # region captions and group headers (text in ink tokens, never series colour)
    ax.text(0.49, y - 0.05, "\u2190 predicts not sarcastic", ha="right", va="center", fontsize=7, color=INK2, style="italic")
    ax.text(0.51, y - 0.05, "predicts sarcastic \u2192", ha="left", va="center", fontsize=7, color=INK2, style="italic")
    first = {g: min(ypos[c] for c, gg in zip(cues, groups) if gg == g) for g in set(groups)}
    for g, label in (("derision", "DERISION"), ("irony", "PLAYFUL")):
        ax.text(-0.02, first[g] - 0.62, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7, color=INK2, weight="bold")
    # identity carried by the legend above the plot
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, mec="white", lw=0, ms=5, label="en-AU annotator"),
                       Line2D([], [], marker="s", color=ORANGE, mec="white", lw=0, ms=5, label="en-UK annotator")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.3, handletextpad=0.2,
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




def _style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, length=2)


def headline_figure() -> None:
    """Figure 1. (a) Share of Reddit comments labelled sarcastic by the gold annotator vs two blind LLM readers (R6 uniform
    sample, 150 per variety, no conditioning; Wilson 95% CIs). (b) Mean AUC of four derision cues vs four playful cues for
    the gold sarcasm label, among comments a label-free sentiment model calls negative (results/paper/cue_flip.csv)."""
    import json
    import glob
    s = pl.read_csv(ROOT / "experiments" / "reannot" / "sample.csv")
    for w in ("a1", "a2"):
        rows = [json.loads(x) for f in sorted(glob.glob(str(ROOT / f"experiments/reannot/labels/{w}_batch_*.jsonl")))
                for x in open(f)]
        s = s.join(pl.DataFrame(rows).rename({"sarcasm": f"{w}_sarc", "sentiment": f"{w}_sent"}), on="item")

    def wilson(k, n, z=1.96):
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return p, (c - h) / d, (c + h) / d

    plt.rcParams.update({"font.family": "serif", "font.size": 8.3})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.35, 2.05))
    u = s.filter(pl.col("stratum") == "U")
    for x0, v in enumerate(("en-AU", "en-UK")):
        d = u.filter(pl.col("variety") == v)
        n = d.height
        g = wilson(int(d["sarcasm"].sum()), n)
        r = wilson(int(round((d["a1_sarc"].sum() + d["a2_sarc"].sum()) / 2)), n)
        for val, col, dx, lab in ((g, BLUE, -0.17, "gold label"), (r, ORANGE, 0.17, "LLM readers")):
            a1.bar(x0 + dx, val[0], width=0.3, color=col, zorder=3, label=lab if v == "en-AU" else None)
            a1.errorbar(x0 + dx, val[0], yerr=[[val[0] - val[1]], [val[2] - val[0]]], color=INK2, lw=0.7, capsize=1.5,
                        zorder=4)
            a1.text(x0 + dx, val[2] + 0.025, f"{val[0]:.0%}", ha="center", va="bottom", fontsize=6.8, color=INK)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["en-AU", "en-UK"], color=INK)
    a1.set_ylim(0, 0.78)
    a1.set_yticks([0, 0.2, 0.4, 0.6])
    a1.set_yticklabels(["0", "20", "40", "60%"])
    a1.set_title("(a) comments labelled\nsarcastic", fontsize=8, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1, borderaxespad=0)
    _style(a1, "y")
    c = pl.read_csv(ROOT / "results" / "paper" / "cue_flip.csv")
    for x0, v in enumerate(("en-AU", "en-UK")):
        for grp, col, dx, lab in (("derision", "#4a3aa7", -0.17, "derision cues"), ("irony", AQUA, 0.17, "playful cues")):
            m = c.filter((pl.col("variety") == v) & (pl.col("group") == grp))["auc"].mean()
            a2.bar(x0 + dx, m - 0.5, bottom=0.5, width=0.3, color=col, zorder=3, label=lab if v == "en-AU" else None)
            a2.text(x0 + dx, m + (0.012 if m > 0.5 else -0.012), f"{m:.2f}", ha="center",
                    va="bottom" if m > 0.5 else "top", fontsize=7.3, color=INK)
    a2.axhline(0.5, color=INK2, lw=0.7, zorder=4)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["en-AU", "en-UK"], color=INK)
    a2.set_ylim(0.36, 0.92)
    a2.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    a2.set_title("(b) cue AUC for the\nsarcasm label", fontsize=8, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=7, loc="upper center", handlelength=1, borderaxespad=0, ncol=1)
    _style(a2, "y")
    fig.tight_layout(w_pad=1.2)
    out = ROOT / "paper" / "fig_headline.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("saved", out)


def system_figure() -> None:
    """System overview: input -> three members -> logit mean -> cross-fitted per-segment thresholds."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({"font.family": "serif", "font.size": 7.2})
    fig, ax = plt.subplots(figsize=(3.35, 1.5))
    ax.set_xlim(-0.15, 11.0)
    ax.set_ylim(0.05, 4.45)
    ax.axis("off")

    def box(x, y, w, h, text, fc, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=INK2, lw=0.7, clip_on=False))
        if sub:
            ax.text(x + w / 2, y + h * 0.63, text, ha="center", va="center", fontsize=7.2, color=INK, weight="bold")
            ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=6.4, color=INK2)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color=INK, linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=INK2,
                                     shrinkA=0, shrinkB=0))

    box(0.0, 1.55, 1.75, 1.4, "text\n[variety]\n[source]", "#f4f3ef")
    members = [("DeBERTa-v3-large", "fine-tuned, 5 seeds", "#dbe8f9"),
               ("e5 + mE5", "frozen; linear probe", "#d6f2e7"),
               ("Qwen3.5-9B", "frozen; linear probe", "#fbe1d6")]
    ys = [3.15, 1.7, 0.25]
    for (t, sub, c), y in zip(members, ys):
        box(2.3, y, 3.6, 1.1, t, c, sub)
        arrow(1.75, 2.25, 2.3, y + 0.55)
        arrow(5.9, y + 0.55, 6.45, 2.25)
    box(6.45, 1.7, 1.55, 1.1, "average\nlogits", "#f4f3ef")
    arrow(8.0, 2.25, 8.45, 2.25)
    box(8.45, 1.4, 2.4, 1.7, "per-segment\nthresholds\n(cross-fitted)", "#f4f3ef")
    fig.tight_layout(pad=0.1)
    out = ROOT / "paper" / "fig_system.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.01)
    print("saved", out)


def gains_figure() -> None:
    """Two panels: (a) learning curves, (b) ensemble size (DeBERTa seeds alone vs inside the three-family ensemble)."""
    df = pool()
    deb = pl.read_csv(ROOT / "results" / "paper" / "learning_curve_deberta.csv").filter(pl.col("seed").is_in([13, 14]))
    probes = pl.read_csv(ROOT / "results" / "paper" / "learning_curve_probes.csv")
    ens = pl.read_csv(ROOT / "results" / "paper" / "ensemble_curve.csv")
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.15))
    for name, col, mk, d in (("DeBERTa (seeds 13, 14)", BLUE, "o", deb),
                             ("9B LLM probe", ORANGE, "s", probes.filter(pl.col("member") == "llm")),
                             ("e5 probe", AQUA, "^", probes.filter(pl.col("member") == "e5"))):
        g = d.group_by("frac").agg(pl.col("score").mean().alias("m"), pl.col("score").std().fill_null(0).alias("sd")).sort("frac")
        x = g["frac"].to_numpy() * 100
        a.errorbar(x, g["m"], yerr=g["sd"], color=col, marker=mk, ms=4, lw=1.5, capsize=2, elinewidth=0.8, zorder=3)
        a.annotate(name, (x[-1], g["m"][-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, color=INK)
    a.axvspan(50, 100, color="#f4f3ef", zorder=0, lw=0)
    a.text(75, 0.792, "plateau", ha="center", fontsize=6.5, color=INK2, style="italic")
    a.set_xlim(5, 138)
    a.set_xticks([10, 25, 50, 75, 100])
    a.set_xticklabels(["10%", "25%", "50%", "75%", "100%"])
    a.set_xlabel("share of training data", color=INK2)
    a.set_ylabel("official score (OOF)", color=INK2)
    a.set_title("(a) more data helps little beyond half", fontsize=7.5, color=INK, loc="left")
    _style(a, "y")
    g = ens.group_by("n").agg(pl.col("deb_score").mean().alias("deb"), pl.col("lead_score").mean().alias("lead")).sort("n")
    n = g["n"].to_numpy()
    b.plot(n, g["deb"], color=BLUE, marker="o", ms=3.5, lw=1.5, zorder=3)
    b.plot(n, g["lead"], color=ORANGE, marker="s", ms=3.5, lw=1.5, zorder=3)
    b.annotate("DeBERTa seeds only", (n[-1], g["deb"][-1]), xytext=(4, -1), textcoords="offset points", va="center",
               fontsize=6.5, color=INK)
    b.annotate("+ e5 probe + 9B probe", (n[-1], g["lead"][-1]), xytext=(4, 1), textcoords="offset points", va="center",
               fontsize=6.5, color=INK)
    b.annotate("", xy=(5.5, g["lead"][4]), xytext=(5.5, g["deb"][4]),
               arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7))
    b.text(5.75, (g["lead"][4] + g["deb"][4]) / 2, "+.007 from\nmodel diversity", fontsize=6.3, color=INK2, va="center")
    b.set_xlim(0.5, 14.8)
    b.set_xticks([1, 2, 3, 5, 10])
    b.set_xlabel("number of DeBERTa seeds averaged", color=INK2)
    b.set_title("(b) more seeds help little beyond three", fontsize=7.5, color=INK, loc="left")
    _style(b, "y")
    fig.tight_layout(w_pad=1.5)
    out = ROOT / "paper" / "fig_gains.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("saved", out)


if __name__ == "__main__":
    import sys
    figs = {"cues": cue_flip_figure, "headline": headline_figure, "system": system_figure, "gains": gains_figure}
    args = sys.argv[1:] or ["learning"]
    for a_ in (figs if args == ["all"] else args):
        figs.get(a_, main)()
