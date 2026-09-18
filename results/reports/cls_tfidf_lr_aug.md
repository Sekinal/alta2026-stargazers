# Validation report: `cls_tfidf_lr_aug`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.046 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.84    | 0.921   | 0.681   | 0.624   | 0.732 | 0.017   | 0.74        |
| global+ukg    | 0.836   | 0.919   | 0.65    | 0.637   | 0.736 | 0.009   | 0.73        |
| variety+ukg   | 0.835   | 0.917   | 0.691   | 0.641   | 0.738 | 0.019   | 0.732       |
| segment+ukg   | 0.847   | 0.916   | 0.69    | 0.641   | 0.744 | 0.02    | 0.735       |
| prev+ukg      | 0.841   | 0.923   | 0.683   | 0.632   | 0.737 | 0.016   | 0.746       |
| prevtuned+ukg | 0.841   | 0.92    | 0.689   | 0.64    | 0.741 | 0.011   | 0.736       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7438**

## Expected test score for `segment+ukg`
**Point estimate 0.7438 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7103, 0.7774]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.847 | 0.848 | 0.018 | 0.819 | 0.875 | null        |
| sent-UK | 0.916 | 0.917 | 0.013 | 0.894 | 0.938 | null        |
| sarc-AU | 0.69  | 0.697 | 0.022 | 0.662 | 0.733 | null        |
| sarc-UK | 0.641 | 0.642 | 0.031 | 0.591 | 0.691 | null        |
| score   | 0.744 | 0.744 | 0.017 | 0.715 | 0.771 | 0.032       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 92.6%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.823   | 0.908   | 0.694   | 0.605   | 0.714 |
| 1    | 0.859   | 0.931   | 0.688   | 0.677   | 0.768 |
| 2    | 0.854   | 0.907   | 0.69    | 0.63    | 0.742 |
| 3    | 0.844   | 0.912   | 0.685   | 0.661   | 0.752 |
| 4    | 0.855   | 0.923   | 0.693   | 0.628   | 0.742 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.456     | 0.847    | 0.843     | 0.827  | 0.919 | 0.923 | 0.113 | 0.034 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.51      | 0.916    | 0.93      | 0.908  | 0.974 | 0.98  | 0.06  | 0.015 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.75      | 0.886    | 0.932     | 0.951  | 0.965 | 0.987 | 0.067 | 0.039 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.298     | 0.781    | 0.722     | 0.675  | 0.862 | 0.78  | 0.137 | 0.053 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.753     | 0.891    | 0.943     | 0.948  | 0.977 | 0.992 | 0.054 | 0.028 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.072     | 0.744    | 0.695     | 0.435  | 0.893 | 0.644 | 0.07  | 0.052 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.346     | 0.69     | 0.538     | 0.628  | 0.801 | 0.575 | 0.188 | 0.148 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.162     | 0.641    | 0.272     | 0.566  | 0.885 | 0.323 | 0.066 | 0.049 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.009     | 0.518    | 0.3       | 0.041  | 0.827 | 0.272 | 0.058 | 0.046 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.527     | 0.622    | 0.54      | 0.679  | 0.694 | 0.589 | 0.258 | 0.203 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.881 | 0.004 | 0.001 | 0.004 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.454     | 0.524    | 0.272     | 0.569  | 0.622 | 0.323 | 0.185 | 0.132 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.75  | 0.602 |
| 1-10  | UK      | 219 | 0.813 | 0.469 |
| 11-30 | AU      | 717 | 0.768 | 0.62  |
| 11-30 | UK      | 429 | 0.602 | 0.51  |
| 31-80 | AU      | 602 | 0.8   | 0.63  |
| 31-80 | UK      | 385 | 0.789 | 0.525 |
| >80   | AU      | 320 | 0.79  | 0.622 |
| >80   | UK      | 110 | 0.481 | 0.46  |
