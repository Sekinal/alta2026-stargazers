# Validation report: `cls_gbdt_xgb_raw`

generated 2026-09-18 01:51, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.053 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.68 (valid 0.804, pool 0.827; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.917   | 0.968   | 0.742   | 0.617   | 0.767 | 0.021   | 0.748       |
| global+ukg    | 0.92    | 0.968   | 0.759   | 0.742   | 0.831 | 0.01    | 0.811       |
| variety+ukg   | 0.919   | 0.966   | 0.758   | 0.736   | 0.828 | 0.012   | 0.808       |
| segment+ukg   | 0.919   | 0.966   | 0.751   | 0.736   | 0.827 | 0.012   | 0.804       |
| prev+ukg      | 0.918   | 0.967   | 0.757   | 0.752   | 0.835 | 0.008   | 0.806       |
| prevtuned+ukg | 0.918   | 0.967   | 0.764   | 0.746   | 0.832 | 0.009   | 0.81        |

**Headline policy: `segment+ukg`** (default policy); **score 0.8272**

## Expected test score for `segment+ukg`
**Point estimate 0.8272 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7971, 0.8574]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.919 | 0.92  | 0.013 | 0.897 | 0.94  | null        |
| sent-UK | 0.966 | 0.966 | 0.009 | 0.951 | 0.979 | null        |
| sarc-AU | 0.751 | 0.756 | 0.02  | 0.723 | 0.789 | null        |
| sarc-UK | 0.736 | 0.739 | 0.035 | 0.68  | 0.801 | null        |
| score   | 0.827 | 0.825 | 0.015 | 0.799 | 0.848 | 0.998       |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 66.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.907   | 0.96    | 0.782   | 0.723   | 0.815 |
| 1    | 0.924   | 0.964   | 0.752   | 0.768   | 0.838 |
| 2    | 0.915   | 0.966   | 0.752   | 0.701   | 0.808 |
| 3    | 0.911   | 0.973   | 0.739   | 0.757   | 0.825 |
| 4    | 0.936   | 0.969   | 0.726   | 0.723   | 0.829 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.437     | 0.919    | 0.94      | 0.883  | 0.976 | 0.974 | 0.06  | 0.013 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.966    | 0.968     | 0.968  | 0.995 | 0.996 | 0.024 | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.959    | 0.974     | 0.984  | 0.995 | 0.998 | 0.022 | 0.016 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.272     | 0.872    | 0.89      | 0.759  | 0.952 | 0.906 | 0.081 | 0.024 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.749     | 0.967    | 0.983     | 0.984  | 0.995 | 0.998 | 0.019 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.113     | 0.878    | 0.791     | 0.779  | 0.981 | 0.895 | 0.034 | 0.017 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.349     | 0.751    | 0.613     | 0.721  | 0.87  | 0.71  | 0.134 | 0.028 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.107     | 0.736    | 0.449     | 0.618  | 0.944 | 0.545 | 0.048 | 0.019 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.04      | 0.59     | 0.302     | 0.178  | 0.89  | 0.357 | 0.053 | 0.028 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.515     | 0.709    | 0.626     | 0.768  | 0.809 | 0.729 | 0.177 | 0.034 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.913 | 0.006 | 0.001 | 0.005 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.3       | 0.677    | 0.449     | 0.621  | 0.82  | 0.546 | 0.133 | 0.048 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.889 | 0.698 |
| 1-10  | UK      | 219 | 0.891 | 0.676 |
| 11-30 | AU      | 717 | 0.862 | 0.725 |
| 11-30 | UK      | 429 | 0.833 | 0.668 |
| 31-80 | AU      | 602 | 0.866 | 0.729 |
| 31-80 | UK      | 385 | 0.886 | 0.669 |
| >80   | AU      | 320 | 0.86  | 0.64  |
| >80   | UK      | 110 | 0.924 | 0.553 |
