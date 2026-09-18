# ALTA 2026 Shared Task — sentiment & sarcasm (en-AU / en-UK, BESSTIE)

Score = (min(sent-AU, sent-UK) + min(sarc-AU, sarc-UK)) / 2, macro-F1 per variety.

## Layout
- `alta/common.py` — data loading (polars, parquet-cached), 5-fold CV over train+valid,
  official metric (verified against `scoring/evaluate.py`), prediction I/O.
- `eda/` — exploratory analysis (lets-plot).
- `data/`, `scoring/`, `baseline/` — competition files (git-ignored).

## Setup
Unzip `alta2026_public_data.zip` into `data/`, then `uv run python -m alta.common` to build the cache.

## Selection charter: generalization first
Holdout score is necessary, not sufficient. A candidate is preferred only if it also:
1. **Confirms on an untouched holdout.** Neural models are screened on fold 0 (train 1-4) and every
   finalist is re-run on fold 4 (train 0-3) with no changes after seeing the result. Non-neural models
   use full 5-fold OOF.
2. **Is stable.** Mean ± SD over >= 3 seeds; within noise (p < 0.8), the lower-variance / simpler
   option wins. Final predictions come from seed/fold ensembles.
3. **Holds its worst slices.** UK-reddit sarcasm, AU-reddit sentiment, short (<= 10 words) and long
   (> 80 words) texts.
4. **Survives perturbation.** Lowercase, punctuation stripping, ~3% typos, AU<->UK spelling/slang swaps,
   tokeniser artefacts, dropped segment prefix, and (for LLMs) prompt paraphrases.
5. **Is not over-tuned.** Fixed epoch counts, regularised stackers (few members, each passing its
   permutation canary, with per-fold gains), no selection on valid.csv or the dev leaderboard.
