"""Shared helpers for the semantic EDA scripts (paths, models, palette, plot theme)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
from lets_plot import (
    LetsPlot, element_blank, element_line, element_rect, element_text, ggsave,
    theme, theme_minimal2,
)

from alta.common import load_pool

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
PLOTS = HERE / "plots"
CACHE.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# short name -> (HF id, max_seq_length, query prefix)
MODELS = {
    "bge_large": ("BAAI/bge-large-en-v1.5", 512),
    "bge_base": ("BAAI/bge-base-en-v1.5", 512),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", 384),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 256),
}

# consistent colours across every plot
VARIETY_COLORS = {"en-AU": "#1b9e77", "en-UK": "#7570b3"}
SOURCE_COLORS = {"google": "#e6ab02", "reddit": "#d95f02"}
LABEL_COLORS = {"0": "#9aa5b1", "1": "#d7263d"}
SENT_COLORS = {"neg": "#c0392b", "pos": "#2e86de"}
JOINT_COLORS = {
    "neg / not sarc": "#c0392b",
    "pos / not sarc": "#2e86de",
    "neg / sarc": "#f39c12",
    "pos / sarc": "#8e44ad",
}

LetsPlot.setup_html()


def pool() -> pl.DataFrame:
    return load_pool().with_columns(pl.col("source", "variety").cast(pl.String))


def emb(model: str) -> np.ndarray:
    """L2-normalised embeddings aligned with pool() row order."""
    return np.load(CACHE / f"emb_{model}.npy")


def segment(df: pl.DataFrame) -> pl.Expr:
    return pl.concat_str(pl.col("variety").str.slice(3), pl.col("source"), separator="-")


def my_theme():
    return theme_minimal2() + theme(
        plot_title=element_text(size=16, face="bold"),
        plot_subtitle=element_text(size=12, color="#555555"),
        plot_caption=element_text(size=10, color="#777777"),
        strip_text=element_text(size=12, face="bold"),
        strip_background=element_rect(fill="#eef1f5", color="#eef1f5"),
        legend_position="bottom",
        panel_grid_minor=element_blank(),
        panel_grid_major=element_line(color="#e5e5e5", size=0.4),
        axis_title=element_text(size=12),
    )


def save(p, name: str, w: int = 1000, h: int = 700, scale: float = 2.0) -> None:
    from lets_plot import ggsize
    p = p + ggsize(w, h)
    ggsave(p, f"{name}.png", path=str(PLOTS), scale=scale)
    ggsave(p, f"{name}.html", path=str(PLOTS))
