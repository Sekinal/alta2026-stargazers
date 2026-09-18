# Validation report: `cls_e5_large_svm`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.045 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.921   | 0.963   | 0.745   | 0.669   | 0.795 | 0.018   | 0.774       |
| global+ukg    | 0.921   | 0.963   | 0.764   | 0.765   | 0.842 | 0.009   | 0.824       |
| variety+ukg   | 0.921   | 0.963   | 0.769   | 0.766   | 0.844 | 0.009   | 0.826       |
| segment+ukg   | 0.921   | 0.963   | 0.768   | 0.766   | 0.844 | 0.01    | 0.828       |
| prev+ukg      | 0.918   | 0.963   | 0.75    | 0.772   | 0.834 | 0.01    | 0.826       |
| prevtuned+ukg | 0.918   | 0.963   | 0.758   | 0.768   | 0.838 | 0.013   | 0.829       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8437**

## Expected test score for `segment+ukg`
**Point estimate 0.8437 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8119, 0.8756]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.921 | 0.923 | 0.014 | 0.9   | 0.943 | null        |
| sent-UK | 0.963 | 0.963 | 0.009 | 0.948 | 0.977 | null        |
| sarc-AU | 0.768 | 0.775 | 0.02  | 0.742 | 0.806 | null        |
| sarc-UK | 0.766 | 0.767 | 0.04  | 0.697 | 0.831 | null        |
| score   | 0.844 | 0.838 | 0.016 | 0.809 | 0.861 | 0.999       |

binding varieties: sent: AU binds 99.2%, sarc: UK binds 55.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.974   | 0.769   | 0.763   | 0.836 |
| 1    | 0.933   | 0.959   | 0.779   | 0.761   | 0.847 |
| 2    | 0.926   | 0.961   | 0.761   | 0.766   | 0.843 |
| 3    | 0.916   | 0.962   | 0.742   | 0.756   | 0.829 |
| 4    | 0.921   | 0.959   | 0.791   | 0.785   | 0.853 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.451     | 0.921    | 0.928     | 0.901  | 0.973 | 0.972 | 0.062 | 0.013 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.521     | 0.963    | 0.966     | 0.964  | 0.994 | 0.995 | 0.027 | 0.007 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.732     | 0.966    | 0.984     | 0.98   | 0.993 | 0.997 | 0.021 | 0.017 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.3       | 0.876    | 0.855     | 0.805  | 0.944 | 0.903 | 0.084 | 0.033 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.963    | 0.982     | 0.981  | 0.993 | 0.998 | 0.023 | 0.01  |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.114     | 0.868    | 0.769     | 0.763  | 0.98  | 0.877 | 0.036 | 0.023 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.344     | 0.768    | 0.638     | 0.74   | 0.869 | 0.709 | 0.133 | 0.014 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.071     | 0.766    | 0.595     | 0.542  | 0.94  | 0.551 | 0.047 | 0.008 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.015     | 0.549    | 0.375     | 0.082  | 0.878 | 0.287 | 0.055 | 0.028 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.52      | 0.728    | 0.642     | 0.797  | 0.812 | 0.73  | 0.175 | 0.031 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.199     | 0.728    | 0.595     | 0.544  | 0.814 | 0.553 | 0.131 | 0.029 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.892 | 0.708 |
| 1-10  | UK      | 219 | 0.907 | 0.687 |
| 11-30 | AU      | 717 | 0.875 | 0.757 |
| 11-30 | UK      | 429 | 0.808 | 0.763 |
| 31-80 | AU      | 602 | 0.858 | 0.742 |
| 31-80 | UK      | 385 | 0.868 | 0.669 |
| >80   | AU      | 320 | 0.871 | 0.65  |
| >80   | UK      | 110 | 0.824 | 0.581 |
