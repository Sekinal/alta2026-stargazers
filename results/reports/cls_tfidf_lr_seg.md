# Validation report: `cls_tfidf_lr_seg`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.035 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.844   | 0.92    | 0.688   | 0.594   | 0.719 | 0.015   | 0.728       |
| global+ukg    | 0.839   | 0.921   | 0.675   | 0.661   | 0.75  | 0.009   | 0.747       |
| variety+ukg   | 0.838   | 0.917   | 0.7     | 0.662   | 0.75  | 0.011   | 0.748       |
| segment+ukg   | 0.839   | 0.916   | 0.699   | 0.662   | 0.75  | 0.012   | 0.749       |
| prev+ukg      | 0.841   | 0.92    | 0.686   | 0.639   | 0.74  | 0.014   | 0.738       |
| prevtuned+ukg | 0.838   | 0.92    | 0.686   | 0.661   | 0.75  | 0.01    | 0.746       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7502**

## Expected test score for `segment+ukg`
**Point estimate 0.7502 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7172, 0.7833]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.839 | 0.839 | 0.018 | 0.808 | 0.867 | null        |
| sent-UK | 0.916 | 0.916 | 0.012 | 0.896 | 0.935 | null        |
| sarc-AU | 0.699 | 0.706 | 0.021 | 0.672 | 0.741 | null        |
| sarc-UK | 0.662 | 0.663 | 0.031 | 0.613 | 0.713 | null        |
| score   | 0.75  | 0.75  | 0.017 | 0.721 | 0.776 | 0.067       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 87.3%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.81    | 0.907   | 0.694   | 0.674   | 0.742 |
| 1    | 0.852   | 0.917   | 0.684   | 0.679   | 0.765 |
| 2    | 0.852   | 0.911   | 0.712   | 0.642   | 0.747 |
| 3    | 0.837   | 0.931   | 0.704   | 0.683   | 0.76  |
| 4    | 0.842   | 0.915   | 0.703   | 0.63    | 0.736 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.455     | 0.839    | 0.834     | 0.817  | 0.919 | 0.922 | 0.12  | 0.083 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.506     | 0.916    | 0.934     | 0.904  | 0.972 | 0.978 | 0.065 | 0.052 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.772     | 0.872    | 0.915     | 0.961  | 0.961 | 0.986 | 0.073 | 0.05  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.284     | 0.768    | 0.718     | 0.639  | 0.86  | 0.766 | 0.146 | 0.098 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.754     | 0.897    | 0.945     | 0.952  | 0.973 | 0.99  | 0.056 | 0.035 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.058     | 0.697    | 0.667     | 0.336  | 0.838 | 0.553 | 0.082 | 0.084 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.356     | 0.699    | 0.546     | 0.654  | 0.806 | 0.582 | 0.17  | 0.087 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.151     | 0.662    | 0.304     | 0.59   | 0.888 | 0.329 | 0.067 | 0.057 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.01      | 0.566    | 0.636     | 0.096  | 0.853 | 0.32  | 0.075 | 0.144 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.541     | 0.628    | 0.545     | 0.703  | 0.698 | 0.595 | 0.22  | 0.058 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.423     | 0.56     | 0.304     | 0.593  | 0.639 | 0.33  | 0.188 | 0.157 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.75  | 0.592 |
| 1-10  | UK      | 219 | 0.803 | 0.464 |
| 11-30 | AU      | 717 | 0.755 | 0.644 |
| 11-30 | UK      | 429 | 0.625 | 0.554 |
| 31-80 | AU      | 602 | 0.771 | 0.631 |
| 31-80 | UK      | 385 | 0.597 | 0.562 |
| >80   | AU      | 320 | 0.78  | 0.619 |
| >80   | UK      | 110 | 0.595 | 0.453 |
