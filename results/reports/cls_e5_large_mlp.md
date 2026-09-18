# Validation report: `cls_e5_large_mlp`

generated 2026-09-17 23:55, code dc7268a-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.024 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.903   | 0.962   | 0.738   | 0.733   | 0.818 | 0.009   | 0.791       |
| global+ukg    | 0.913   | 0.957   | 0.743   | 0.753   | 0.828 | 0.011   | 0.809       |
| variety+ukg   | 0.913   | 0.961   | 0.742   | 0.754   | 0.827 | 0.01    | 0.811       |
| segment+ukg   | 0.908   | 0.963   | 0.753   | 0.754   | 0.83  | 0.01    | 0.806       |
| prev+ukg      | 0.915   | 0.965   | 0.744   | 0.761   | 0.829 | 0.009   | 0.814       |
| prevtuned+ukg | 0.915   | 0.965   | 0.748   | 0.751   | 0.832 | 0.009   | 0.814       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8305**

## Expected test score for `segment+ukg`
**Point estimate 0.8305 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7999, 0.8611]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.908 | 0.909 | 0.014 | 0.886 | 0.932 | null        |
| sent-UK | 0.963 | 0.964 | 0.009 | 0.948 | 0.979 | null        |
| sarc-AU | 0.753 | 0.758 | 0.021 | 0.723 | 0.793 | null        |
| sarc-UK | 0.754 | 0.755 | 0.039 | 0.692 | 0.819 | null        |
| score   | 0.83  | 0.824 | 0.015 | 0.797 | 0.848 | 0.996       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 52.6%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.884   | 0.971   | 0.733   | 0.775   | 0.808 |
| 1    | 0.92    | 0.964   | 0.766   | 0.718   | 0.819 |
| 2    | 0.913   | 0.956   | 0.751   | 0.726   | 0.82  |
| 3    | 0.908   | 0.967   | 0.743   | 0.794   | 0.825 |
| 4    | 0.916   | 0.959   | 0.767   | 0.754   | 0.835 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.451     | 0.908    | 0.914     | 0.887  | 0.97  | 0.97  | 0.068 | 0.039 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.526     | 0.963    | 0.962     | 0.968  | 0.993 | 0.993 | 0.028 | 0.009 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.957    | 0.972     | 0.982  | 0.993 | 0.997 | 0.022 | 0.014 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.295     | 0.856    | 0.834     | 0.77   | 0.942 | 0.897 | 0.093 | 0.063 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.755     | 0.96     | 0.976     | 0.985  | 0.994 | 0.998 | 0.024 | 0.019 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.111     | 0.878    | 0.795     | 0.771  | 0.975 | 0.865 | 0.037 | 0.031 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.326     | 0.753    | 0.63      | 0.693  | 0.847 | 0.678 | 0.15  | 0.067 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.082     | 0.754    | 0.532     | 0.562  | 0.916 | 0.534 | 0.054 | 0.036 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.02      | 0.576    | 0.409     | 0.123  | 0.864 | 0.296 | 0.069 | 0.048 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.491     | 0.71     | 0.635     | 0.742  | 0.792 | 0.715 | 0.194 | 0.095 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.23      | 0.709    | 0.532     | 0.565  | 0.804 | 0.541 | 0.15  | 0.105 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.853 | 0.689 |
| 1-10  | UK      | 219 | 0.896 | 0.682 |
| 11-30 | AU      | 717 | 0.856 | 0.74  |
| 11-30 | UK      | 429 | 0.854 | 0.721 |
| 31-80 | AU      | 602 | 0.838 | 0.726 |
| 31-80 | UK      | 385 | 0.856 | 0.696 |
| >80   | AU      | 320 | 0.876 | 0.635 |
| >80   | UK      | 110 | 0.924 | 0.615 |
