# EDA: lexical, stylistic and dialect signal

Numbers are on the pool of 6,288 rows (train + valid); models are 5-fold out-of-fold on the shared folds.
Re-run with `uv run python eda/lexical/eda_lexical.py [01 02 03 04]`.
Plots are in `plots/` (PNG + HTML), tables in `tables/`.

**Main finding:** sarcasm is the weak side of the score. Lexical sarcasm cues don't carry over between AU and UK at all, and the UK sarcasm signal is close to noise at this sample size. Sentiment cues transfer almost fully.

| cell | n | sarcastic | positive |
|---|---:|---:|---:|
| reddit · AU | 2,004 | 42% | 32% |
| reddit · UK | 1,143 | 22% | 11% |
| google · AU | 1,076 | 73 texts | 74% |
| google · UK | 2,065 | 1 text | 75% |

## Findings

### 1. Sarcasm cues don't transfer between AU and UK (reddit)
Plots: `logodds_sarcasm_reddit.png`, `transfer_sarc_reddit.png`. Table: `transfer_summary.csv`.

- Across 890 shared terms, the correlation of per-term sarcasm log-odds (Monroe et al. 2008, informative Dirichlet prior) between AU and UK is r = 0.00.
- Only 44% of the top-50 AU cues point the same way in UK, which is chance.
- Split-half reliability within a variety is 0.20 for AU and only 0.05 for UK (248 positives). So even a fully shared signal would show at most r ≈ 0.18 across varieties.

### 2. AU sarcasm is mostly topic; UK sarcasm is mostly set phrases
Tables: `logodds_sarc_reddit_AU.csv`, `logodds_sarc_reddit_UK.csv`.

- AU sarcastic: `lnp` (z 3.9), `shit`, `dutton`, `labor`, `liberals`, `politicians`, `rich`, `why`.
- AU non-sarcastic: advice and travel language (`i`, `nice`, `city`, `if you`, `recommend`, `beach`).
- UK sarcastic: `oh` (z 2.7), `boy`, `/s`, `can't wait`, `be fair`, `i'm sure`, `yeah`. No UK term reaches |z| > 3.
- AU texts mentioning a party or politician are 65% sarcastic and 3% positive.

### 3. Sentiment cues do transfer
Plots: `logodds_sentiment_*.png`, `transfer_sent_reddit.png`, `transfer_sent_google.png`.

| source | AU vs UK r | reliability ceiling | top-50 sign agreement |
|---|---:|---:|---:|
| reddit | 0.38 | ~0.44 | 90% |
| google | 0.70 | ~0.67 | 100% |

### 4. Few style markers work in both varieties, and some flip sign
Plots: `style_lift_sarcasm_reddit.png`, `style_rates_sarcasm_reddit.png`. Table: `style_rates.csv`.

| marker | AU sarcastic vs not | UK sarcastic vs not | direction |
|---|---|---|---|
| "oh" / "ah" | 2.3% vs 0.6% (log2 lift +1.8) | 5.6% vs 1.0% (+2.5) | same |
| question mark | 24% vs 13% | 23% vs 18% | same |
| "I / me / my" | lower | lower | same |
| profanity | 13% vs 6% (+1.2) | 13% vs 22% (−0.7) | opposite |
| thanks / cheers | −1.5 | +1.1 | opposite |

- `/s` appears in only 3 of 248 UK sarcastic texts.
- Emoji, URLs and hashtags are essentially absent from the data.

### 5. "Sarcasm inverts sentiment" holds only for UK
Plots: `surface_polarity_bars.png`, `surface_compound_density.png`. Tables: `surface_polarity.csv`, `vader_fooled.csv`.

Share of negative texts that VADER scores as positive (compound > 0.05):

| segment | plain negative | sarcastic negative |
|---|---:|---:|
| reddit · UK | 29% | 49% |
| reddit · AU | 39% | 40% |
| google · AU | 52% | 64% |

- UK sarcastic negatives don't contain more positive words (64% vs 65% have one). They contain fewer negative words: 39% have both polarities vs 54% of plain negatives.
- 92–99% of sarcastic texts are negative.

### 6. Dialect slang is rare; entities and topics separate the varieties
Plots: `dialect_markers.png`, `dialect_logodds.png`. Tables: `dialect_markers.csv`, `dialect_separability.csv`.

- Every slang marker appears in 1.5% of texts or fewer. `-ise` spelling is about 9% in both varieties.

| marker | AU reddit | UK reddit |
|---|---:|---:|
| AU places | 16.5% | 0.3% |
| Labor / LNP / Dutton / Albo | 8.6% | 0.35% |
| Tories / Labour / NHS / Reform | 1.2% | 13.8% |

- Telling AU from UK within a source (5-fold AUC) gives about 0.75 on google and 0.81 on reddit.

### 7. The AU/UK sarcasm gap is not topic mix
Plot: `topics_reddit.png`. Table: `topics_reddit.csv`.

- Across 12 reddit NMF topics, AU has the higher sarcasm rate in 10, often by 2–7×. For example, the "people / country / uk" topic is 48% vs 7%.
- The other two topics have overlapping confidence intervals.
- This looks like an annotation or community difference, not a topic effect.

### 8. Formatting quirk in google texts
- 27–32% of google texts split contractions ("did n't", "wo n't"); 0% of reddit texts do.
- That makes `n't` a top negative-sentiment cue for google. It's harmless because source is known at test time, but tokenizers will see these as odd tokens.

### 9. TF-IDF + logistic regression baseline
Plot: `tfidf_lr_scores.png`. Tables: `tfidf_lr_scores.csv`, `tfidf_oof_all_variants.parquet` (OOF probabilities for every variant, for stacking).

Features are word 1–2grams (60k) plus char_wb 2–5grams (150k); liblinear, class_weight=balanced, C=4 for sentiment and 2 for sarcasm. Thresholds:
- **0.5:** fixed.
- **cross-fit:** tuned per variety on the other 4 folds (honest).
- **tuned:** chosen on the same OOF predictions (optimistic).

| variant | threshold | sent-AU | sent-UK | sarc-AU | sarc-UK | score | reddit-only sarc AU / UK |
|---|---|---:|---:|---:|---:|---:|---|
| global | 0.5 | .841 | .918 | .687 | .614 | .728 | .620 / .480 |
| global | cross-fit | .844 | .916 | .681 | .610 | .727 | |
| global + segment one-hots | 0.5 | .845 | .915 | .667 | .582 | .714 | .533 / .510 |
| global + segment one-hots | cross-fit | .843 | .913 | .686 | .625 | .734 | |
| augment (shared + per-variety copies) | 0.5 | .839 | .919 | .690 | .601 | .720 | .593 / .549 |
| augment | cross-fit | .835 | .917 | .685 | .639 | .737 | |
| per-variety | 0.5 | .835 | .914 | .698 | .634 | .734 | .634 / .572 |
| per-variety | cross-fit | .832 | .913 | .702 | .646 | .739 | |
| per-segment (source × variety) | 0.5 | .837 | .917 | .681 | .584 | .711 | .628 / .536 |
| **per-segment** | **cross-fit** | .831 | .915 | .698 | .653 | **.742** (best honest) | |
| per-segment | tuned (optimistic) | .839 | .919 | .700 | .661 | .750 | |
| other variety only | 0.5 | .801 | .897 | .533 | .598 | .667 | .485 / .393 |
| other variety only | cross-fit | .814 | .896 | .632 | .595 | .705 | |

- The two components that cap the score are sent-AU (~0.84) and sarc-UK (0.61–0.65).
- On reddit alone, UK sarcasm macro-F1 is only 0.48–0.57. The sarc-UK column looks better because 2,065 easy google · UK negatives inflate it.
- Per-variety models help sarcasm (sarc-UK .614 → .634 at 0.5) and slightly hurt sentiment (sent-AU .841 → .835).
- Training on the other variety only wrecks sarcasm (reddit sarc-UK .393) but costs sentiment only 3–4 points.

### 10. Saved predictions
- `preds/tfidf_lr.parquet` holds the per-segment variant (OOF only); `preds/tfidf_lr.json` holds its thresholds.
- `evaluate_oof('tfidf_lr')` at fixed 0.5 gives 0.711 on the pool and 0.731 on the valid subset.
- Suggested per-variety thresholds: sarc-UK ≈ 0.33, sarc-AU ≈ 0.45, sentiment ≈ 0.5.

## Implications for modelling

1. **Sentiment: one shared model** across varieties and sources. The cues transfer (google r = 0.70; top-50 sign agreement 100%), and the global model is best on sent-AU. sent-AU is the weaker side because reddit · AU mixes political negatives with advice/travel positives, so aim effort there.
2. **Sarcasm: a shared encoder conditioned on variety**, e.g. a `[en-UK] [reddit]` prefix, feature augmentation, or per-variety heads. Tune per-variety thresholds on held-out folds. Don't rely on AU data to teach UK sarcasm: the lexical cues are uncorrelated (r = 0.00), and AU → UK scores .39 on reddit.
3. **sarc-UK is the bottleneck.** Its n-gram signal is near noise (split-half r = 0.05, 248 positives), so gains have to come from pretrained models, e.g. irony-pretrained encoders. Two extra signals to try:
   - the mismatch between surface polarity and a negative label (49% of UK sarcastic negatives read as positive vs 29% of plain negatives);
   - the stock phrases oh, boy, can't wait, I'm sure, /s, and "?".
4. **Train sentiment and sarcasm jointly:** 92–99% of sarcastic texts are negative. Predicted negativity is necessary but not sufficient for sarcasm.
5. **AU sarcasm ≈ politically charged or grievance reddit text;** lexical models already reach about .63 on reddit · AU. AU's sarcasm rate is higher in 10 of 12 topics, so set priors and thresholds per variety, not globally.
6. **Skip dialect features:** slang appears in 1.5% of texts or fewer, and variety is known at test time. Named entities already carry the variety and topic signal.
7. **Predict sarcasm = 0 for google · UK** (1 positive in 2,065). google · AU's 73 sarcastic reviews count toward sarc-AU; they look like negative reviews with "?", "...", laughter, quotes and "you".
