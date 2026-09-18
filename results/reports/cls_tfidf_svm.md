# Validation report: `cls_tfidf_svm`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.022 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.839   | 0.903   | 0.695   | 0.64    | 0.739 | 0.011   | 0.747       |
| global+ukg    | 0.842   | 0.9     | 0.696   | 0.654   | 0.748 | 0.012   | 0.756       |
| variety+ukg   | 0.842   | 0.911   | 0.69    | 0.649   | 0.745 | 0.011   | 0.751       |
| segment+ukg   | 0.849   | 0.92    | 0.689   | 0.649   | 0.749 | 0.012   | 0.758       |
| prev+ukg      | 0.843   | 0.92    | 0.678   | 0.639   | 0.741 | 0.014   | 0.745       |
| prevtuned+ukg | 0.843   | 0.92    | 0.677   | 0.65    | 0.746 | 0.013   | 0.764       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7488**

## Expected test score for `segment+ukg`
**Point estimate 0.7488 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7126, 0.7850]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.849 | 0.85  | 0.017 | 0.823 | 0.877 | null        |
| sent-UK | 0.92  | 0.92  | 0.012 | 0.899 | 0.94  | null        |
| sarc-AU | 0.689 | 0.697 | 0.021 | 0.665 | 0.732 | null        |
| sarc-UK | 0.649 | 0.647 | 0.036 | 0.589 | 0.704 | null        |
| score   | 0.749 | 0.748 | 0.018 | 0.717 | 0.777 | 0.074       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 88.3%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.826   | 0.913   | 0.686   | 0.651   | 0.739 |
| 1    | 0.852   | 0.93    | 0.691   | 0.668   | 0.76  |
| 2    | 0.857   | 0.914   | 0.7     | 0.669   | 0.763 |
| 3    | 0.841   | 0.922   | 0.672   | 0.647   | 0.744 |
| 4    | 0.868   | 0.922   | 0.695   | 0.603   | 0.736 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.404     | 0.849    | 0.891     | 0.775  | 0.918 | 0.923 | 0.115 | 0.052 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.511     | 0.92     | 0.933     | 0.913  | 0.967 | 0.975 | 0.071 | 0.042 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.724     | 0.901    | 0.954     | 0.939  | 0.972 | 0.99  | 0.072 | 0.107 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.232     | 0.767    | 0.787     | 0.572  | 0.859 | 0.778 | 0.138 | 0.057 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.75      | 0.9      | 0.949     | 0.951  | 0.975 | 0.992 | 0.07  | 0.085 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.08      | 0.749    | 0.663     | 0.466  | 0.886 | 0.647 | 0.073 | 0.049 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.381     | 0.689    | 0.524     | 0.673  | 0.796 | 0.56  | 0.171 | 0.074 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.12      | 0.649    | 0.302     | 0.466  | 0.885 | 0.318 | 0.08  | 0.087 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.001     | 0.482    | 0.0       | 0.0    | 0.755 | 0.167 | 0.061 | 0.018 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.585     | 0.609    | 0.525     | 0.731  | 0.692 | 0.575 | 0.231 | 0.11  |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.671 | 0.001 | 0.001 | 0.002 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.336     | 0.563    | 0.302     | 0.468  | 0.625 | 0.318 | 0.223 | 0.241 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.752 | 0.599 |
| 1-10  | UK      | 219 | 0.818 | 0.484 |
| 11-30 | AU      | 717 | 0.742 | 0.622 |
| 11-30 | UK      | 429 | 0.645 | 0.543 |
| 31-80 | AU      | 602 | 0.787 | 0.6   |
| 31-80 | UK      | 385 | 0.768 | 0.567 |
| >80   | AU      | 320 | 0.781 | 0.599 |
| >80   | UK      | 110 | 0.481 | 0.466 |
