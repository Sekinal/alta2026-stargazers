# Validation report: `stk_sel_lr`

generated 2026-09-17 23:55, code dc7268a-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.055 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.922   | 0.973   | 0.77    | 0.746   | 0.834 | 0.02    | 0.811       |
| global+ukg    | 0.919   | 0.973   | 0.78    | 0.776   | 0.847 | 0.01    | 0.83        |
| variety+ukg   | 0.92    | 0.973   | 0.786   | 0.785   | 0.852 | 0.012   | 0.85        |
| segment+ukg   | 0.919   | 0.973   | 0.784   | 0.785   | 0.852 | 0.012   | 0.85        |
| prev+ukg      | 0.918   | 0.972   | 0.768   | 0.787   | 0.843 | 0.012   | 0.853       |
| prevtuned+ukg | 0.918   | 0.972   | 0.771   | 0.784   | 0.845 | 0.011   | 0.848       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8516**

## Expected test score for `segment+ukg`
**Point estimate 0.8516 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8219, 0.8813]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.919 | 0.92  | 0.014 | 0.897 | 0.941 | null        |
| sent-UK | 0.973 | 0.973 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.784 | 0.79  | 0.02  | 0.758 | 0.824 | null        |
| sarc-UK | 0.785 | 0.785 | 0.037 | 0.723 | 0.845 | null        |
| score   | 0.852 | 0.845 | 0.015 | 0.818 | 0.868 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 54.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.914   | 0.978   | 0.796   | 0.778   | 0.846 |
| 1    | 0.921   | 0.966   | 0.8     | 0.801   | 0.861 |
| 2    | 0.923   | 0.97    | 0.788   | 0.783   | 0.853 |
| 3    | 0.913   | 0.975   | 0.742   | 0.793   | 0.828 |
| 4    | 0.922   | 0.973   | 0.794   | 0.769   | 0.845 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.461     | 0.919    | 0.916     | 0.909  | 0.978 | 0.977 | 0.058 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.973    | 0.977     | 0.97   | 0.995 | 0.996 | 0.022 | 0.002 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.969    | 0.981     | 0.986  | 0.996 | 0.999 | 0.017 | 0.007 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.312     | 0.871    | 0.834     | 0.814  | 0.952 | 0.914 | 0.08  | 0.014 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.968    | 0.984     | 0.984  | 0.994 | 0.998 | 0.02  | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.104     | 0.915    | 0.891     | 0.809  | 0.982 | 0.919 | 0.025 | 0.011 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.321     | 0.784    | 0.675     | 0.731  | 0.884 | 0.736 | 0.126 | 0.008 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.088     | 0.785    | 0.569     | 0.647  | 0.949 | 0.619 | 0.043 | 0.004 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.024     | 0.603    | 0.462     | 0.164  | 0.903 | 0.395 | 0.051 | 0.02  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.481     | 0.752    | 0.681     | 0.78   | 0.83  | 0.755 | 0.166 | 0.019 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.195 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.248     | 0.744    | 0.569     | 0.649  | 0.841 | 0.621 | 0.119 | 0.021 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.892 | 0.719 |
| 1-10  | UK      | 219 | 0.929 | 0.724 |
| 11-30 | AU      | 717 | 0.858 | 0.774 |
| 11-30 | UK      | 429 | 0.889 | 0.752 |
| 31-80 | AU      | 602 | 0.871 | 0.764 |
| 31-80 | UK      | 385 | 0.917 | 0.715 |
| >80   | AU      | 320 | 0.852 | 0.711 |
| >80   | UK      | 110 | 0.877 | 0.647 |
