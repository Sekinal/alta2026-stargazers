# EDA: structure, labels, distribution shift and the metric

Numbers are on the pool of 6,288 rows (train + valid) unless marked valid.
Re-run with `uv run python eda/structure/eda_structure.py` (about 1 minute on CPU; log in `output.txt`).
Plots are in `plots/` (PNG + HTML), tables in `tables/`, example dumps in `examples/`.

## Findings

### 1. The segment decides most of the label
Plots: `01_composition_tiles.png`, `02_joint_labels.png`.

| segment | n | % positive | % sarcastic (n) |
|---|---:|---:|---:|
| en-AU google | 1,076 | 73.5 | 6.8 (73) |
| en-AU reddit | 2,004 | 31.9 | 42.0 (841) |
| en-UK google | 2,065 | 74.8 | 0.05 (1) |
| en-UK reddit | 1,143 | 11.5 | 21.7 (248) |

- Sarcasm prevalence is 29.7% in AU and 7.8% in UK.
- Only 38 of 1,163 sarcastic rows are positive.

### 2. AU and UK annotators seem to have labelled sarcasm differently
Examples: `examples/google_sarcastic.txt`, `examples/sarc1_sent1.txt`, `examples/*_reddit_sarcastic_sample.txt`.

- Most of the 73 AU google "sarcastic" rows are plain angry complaints. Only about a third are clearly ironic (e.g. t00022, t00068, t00174, t00610).
- UK google has 1 sarcastic row.
- Many positive-and-sarcastic AU rows are jokes or even sincere text (e.g. t00172, t02638, t01148).
- UK reddit sarcasm looks clean and genuinely ironic.

### 3. valid.csv is an exact stratified sample
Plot: `06_train_vs_valid.png`. Tables: `train_valid_cells.csv`, `train_vs_valid.csv`.

- All 15 (variety, source, sentiment, sarcasm) cells are within ±2.1 rows of proportional allocation. Random sampling would give deviations of about ±14.
- The train:valid ratio is 7.3 in every segment, and length KS tests give p ≥ 0.08.
- So the hidden test set is very likely stratified the same way, with the same prevalences.

### 4. A rule using only the source already scores 0.64
Plot: `08_trivial_rules.png`. Table: `trivial_rules.csv`.

| predictor | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|---|---:|---:|---:|---:|---:|
| constant | – | – | – | – | 0.37 |
| random at segment prior | – | – | – | – | 0.57 |
| source rule (google → pos, not sarc; reddit → neg, sarc) | 0.69 | 0.79 | 0.60 | 0.59 | 0.64 |
| TF-IDF + LR, out-of-fold (reference) | 0.84 | 0.91 | 0.68 | 0.59 | 0.717 |
| organisers' baseline (distilbert, valid only) | 0.911 | 0.953 | 0.714 | 0.637 | 0.774 |

### 5. UK sarcasm is barely learned by bag-of-words
- The TF-IDF model ties the source rule on sarc-UK (0.591 vs 0.589).
- Within UK reddit its sarcasm AUC is 0.545, and macro-F1 on those rows is 0.51. Its sarc-UK comes from the 2,065 google rows, which are easy true negatives.
- Even knowing the true sentiment only gets sarc-UK to 0.62.
- AU reddit sarcasm is more learnable (AUC 0.66).

### 6. Two components decide the score
- Sentiment's lower variety is AU in 100% of bootstraps: AU reddit is the hardest segment (accuracy 0.81).
- Sarcasm's lower variety is UK in 98.5% of bootstraps.
- So the score is effectively (sent-AU + sarc-UK) / 2.

### 7. Scores on a valid-sized test set are noisy, sarc-UK most of all
Plot: `10_bootstrap_noise.png`. Table: `bootstrap_noise.csv`. 2,000 bootstrap test sets with valid's exact cell counts:

| component | SD | 90% interval |
|---|---:|---|
| sent-AU | 0.018 | – |
| sent-UK | 0.012 | – |
| sarc-AU | 0.022 | – |
| sarc-UK | 0.037 | [0.53, 0.65] |
| final score | 0.021 | [0.68, 0.75] |

Paired comparisons are much sharper: a model that fixes 10% of sarcasm errors wins on 99.5% of resamples.

### 8. Thresholds matter, and the best one differs by variety
Plots: `09_threshold_curves_oof.png`, `11_threshold_sim_prevalence.png`. Table: `binormal_threshold_sim.csv`.

- In a binormal simulation, macro-F1 is best when the predicted-positive rate roughly equals the true prevalence.
- Ceiling on macro-F1 by AUC:

  | prevalence | AUC 0.80 | AUC 0.90 | AUC 0.95 |
  |---|---:|---:|---:|
  | 7.8% (UK sarcasm) | 0.64 | 0.74 | 0.82 |
  | 29.7% (AU sarcasm) | 0.71 | 0.81 | 0.87 |

- The UK sweet spot is narrow: at AUC 0.90, you stay within 1 point of the peak only for predicted rates of 5.3–11%.
- On the real TF-IDF model (out-of-fold), the best thresholds are:
  - sarcasm: 0.37 for UK vs 0.58 for AU
  - sentiment: 0.58 for UK vs 0.48 for AU
- Threshold tuning alone lifts sarc-UK from 0.591 to 0.619.
- Picking the sarc-UK threshold on a valid-sized sample is unstable (5–95% range 0.19–0.48). Because the curve is flat near the peak, this costs only about 0.006 F1 on average.

### 9. Length and formatting
Plots: `03_length_density.png`, `04_label_rate_by_length.png`, `05_formatting_markers.png`, `07_short_texts.png`.

- Google texts are about twice as long as reddit: median about 280–310 vs 140–150 characters (4 vs 2 sentences). AU reddit has a long tail (maximum 8,537 characters; 1.2% of rows over 400 words).
- Longer texts are more negative in every segment. The AUC of log(length) for positive sentiment is 0.37–0.41 (p < 1e-6). In google, the positive rate drops from 83% in the shortest fifth to 62% in the longest.
- Sarcasm vs length:
  - UK reddit: longer posts are less sarcastic (AUC 0.42; 32% sarcastic in the second-shortest fifth, 10% in the longest).
  - AU google: longer reviews are more sarcastic (AUC 0.63; the long complaints).
  - AU reddit: no relationship (0.51).
- Texts under 20 characters (124 rows, nearly all reddit) are mostly positive ("Thank you!", "Yes").
- The text is pre-cleaned: no non-ASCII characters, no emoji, URLs or HTML entities. Only 8 rows contain the "/s" marker.
- 25–30% of google reviews contain tokeniser artefacts such as "could n't" and "do n't", and 13% have spaced parentheses like "( finally)". Reddit has none.

### 10. Duplicates are negligible
Tables: `dups_text.csv`, `dups_norm.csv`, `near_duplicates.csv`.

- 13 exact-duplicate groups (27 rows); 15 groups after lowercasing and stripping punctuation (32 rows). None has conflicting labels or spans both varieties; all are reddit.
- 7 groups span train and valid. `alta.common` puts duplicates in the same fold, so cross-validation is unaffected.
- Only 11 near-duplicate pairs (char TF-IDF cosine ≥ 0.8), all short replies. Two disagree on sarcasm: "That's disgusting" (AU, sarcastic) vs "This is disgusting" / "Disgusting hag" (UK, not sarcastic). This fits the AU/UK labelling gap.

## Implications for modelling

1. **Focus on two components.** The score is effectively (sent-AU + sarc-UK) / 2, so improving UK sentiment or AU sarcasm doesn't change it. Put capacity and error analysis on AU reddit sentiment and UK reddit sarcasm. Consider up-weighting those in the loss, or a separate UK-reddit sarcasm model.
2. **Give the model source and variety,** e.g. a text prefix like `[en-UK][reddit] ...`. Both are known at test time, and a source-only rule scores 0.64.
3. **Hard-code sarcasm = 0 for en-UK google** (1 positive in 2,065). Don't do this for en-AU google: sarcasm is 6.8% there, and the reference model detects it with AUC 0.79 within that segment.
4. **Tune thresholds per variety** (optionally per variety × source) on the pooled 5-fold out-of-fold probabilities. Target predicted-positive rates near the known prevalences: UK sarcasm about 7.8% overall (about 22% of UK reddit), AU sarcasm about 30%, reddit positive sentiment about 32% in AU and 11.5% in UK. Never tune on valid's 30 UK sarcastic rows alone.
5. **Select models on the out-of-fold score** over all 6,288 rows (249 UK sarcastic), compared paired on the same rows. Treat valid-only differences under about 0.02 in score, or 0.03 in sarc-UK, as ties. Expect the leaderboard to move about ±0.02 by chance.
6. **UK sarcasm needs a model that understands context:** a strong pretrained encoder or an LLM, possibly helped by the 841 AU-reddit sarcastic rows. AU sarcasm labels are looser, so training on both varieties together may shift what "sarcastic" means for UK. Check sarc-UK specifically when pooling.
7. **Normalise google text cheaply:** rejoin tokeniser artefacts (" n't" → "n't", "( x" → "(x", "' x '" → "'x'").
8. **Truncation:** 256 tokens covers more than 98% of rows. Use head + tail truncation for the AU reddit long tail.
9. **Skip:** deduplication, train/valid shift correction, and label cleaning. The label noise is concentrated in AU google sarcasm and positive-sarcastic AU rows, which affect sarc-AU, not the score. Revisit only if pooled training visibly hurts sarc-UK.
