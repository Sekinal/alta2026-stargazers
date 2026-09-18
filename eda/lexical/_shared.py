"""Shared helpers for the lexical EDA: tokenisation, segments, colours, plot theme, export."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lets_plot import (
    LetsPlot, element_blank, element_line, element_rect, element_text, ggsave, theme,
    theme_minimal2,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import load_pool  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
TABLES = HERE / "tables"
PLOTS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

LetsPlot.setup_html()

# Colours (validated reference palette: blue/orange/aqua pass all-pairs CVD checks).
VAR_COL = {"en-AU": "#2a78d6", "en-UK": "#eb6834"}
SENT_COL = {"negative": "#e34948", "positive": "#1baf7a"}
SARC_COL = {"not sarcastic": "#8a8984", "sarcastic": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SURFACE = "#fcfcfb"


def th(legend: str = "top"):
    return theme_minimal2() + theme(
        plot_title=element_text(size=15, face="bold", color=INK),
        plot_subtitle=element_text(size=11, color=INK2),
        plot_caption=element_text(size=9, color=INK2),
        axis_title=element_text(size=11, color=INK2),
        axis_text=element_text(size=10, color=INK2),
        strip_text=element_text(size=11, face="bold", color=INK),
        panel_grid_major=element_line(color=GRID, size=0.4),
        panel_grid_minor=element_blank(),
        plot_background=element_rect(fill=SURFACE, color=SURFACE),
        legend_position=legend,
        legend_title=element_text(size=10, color=INK2),
    )


def save(p, name: str, w: int = 900, h: int = 600, scale: float = 2.0) -> None:
    """Write PNG + interactive HTML of a lets-plot figure into plots/."""
    p = p + theme(plot_background=element_rect(fill=SURFACE, color=SURFACE))
    ggsave(p, f"{name}.png", path=str(PLOTS), w=w, h=h, unit="px", scale=scale)
    ggsave(p, f"{name}.html", path=str(PLOTS), w=w, h=h, unit="px")


# ---------------------------------------------------------------- text normalisation
_URL = re.compile(r"https?://\S+|www\.\S+")


def normalise(s: pl.Expr) -> pl.Expr:
    """Lower-case text with URLs / user mentions / subreddits collapsed to placeholder tokens."""
    return (
        s.str.replace_all(r"[‘’ʼ`]", "'")
        .str.replace_all(r"[“”]", '"')
        .str.replace_all(_URL.pattern, " xxurl ")
        .str.replace_all(r"(?:^|\s)/?u/[A-Za-z0-9_-]+", " xxuser ")
        .str.replace_all(r"(?:^|\s)/?r/[A-Za-z0-9_]+", " xxsubreddit ")
        .str.to_lowercase()
    )


# Words keep internal apostrophes (don't, it's); "/s" survives as a token.
TOKEN_PATTERN = r"(?u)(?:/s\b|[a-z][a-z0-9]*(?:'[a-z]+)?|\d+)"


def pool_with_norm() -> pl.DataFrame:
    return load_pool().with_columns(
        normalise(pl.col("text")).alias("norm"),
        pl.col("source").cast(pl.String),
        pl.col("variety").cast(pl.String),
        (pl.col("source").cast(pl.String) + " · " + pl.col("variety").cast(pl.String)).alias("segment"),
    )


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), 0)
    den = 1 + z**2 / np.maximum(n, 1)
    centre = (p + z**2 / (2 * np.maximum(n, 1))) / den
    half = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z**2 / (4 * np.maximum(n, 1) ** 2)) / den
    return centre - half, centre + half
