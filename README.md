# ALTA 2026 Shared Task — sentiment & sarcasm (en-AU / en-UK, BESSTIE)

Score = (min(sent-AU, sent-UK) + min(sarc-AU, sarc-UK)) / 2, macro-F1 per variety.

## Layout
- `alta/common.py` — data loading (polars, parquet-cached), 5-fold CV over train+valid,
  official metric (verified against `scoring/evaluate.py`), prediction I/O.
- `eda/` — exploratory analysis (lets-plot).
- `data/`, `scoring/`, `baseline/` — competition files (git-ignored).

## Setup
Unzip `alta2026_public_data.zip` into `data/`, then `uv run python -m alta.common` to build the cache.
