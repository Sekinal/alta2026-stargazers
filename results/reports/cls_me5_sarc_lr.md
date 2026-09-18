# Validation report: `cls_me5_sarc_lr`

generated 2026-09-17 23:55, code dc7268a-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.013 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.916   | 0.965   | 0.739   | 0.758   | 0.828 | 0.006   | 0.828       |
| global+ukg    | 0.916   | 0.964   | 0.736   | 0.764   | 0.826 | 0.006   | 0.835       |
| variety+ukg   | 0.915   | 0.963   | 0.736   | 0.762   | 0.826 | 0.005   | 0.835       |
| segment+ukg   | 0.914   | 0.961   | 0.762   | 0.762   | 0.838 | 0.012   | 0.845       |
| prev+ukg      | 0.913   | 0.962   | 0.755   | 0.748   | 0.831 | 0.012   | 0.836       |
| prevtuned+ukg | 0.913   | 0.962   | 0.758   | 0.757   | 0.835 | 0.013   | 0.83        |

**Headline policy: `segment+ukg`** (default policy); **score 0.8380**

## Expected test score for `segment+ukg`
**Point estimate 0.8380 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8088, 0.8672]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.914 | 0.915 | 0.014 | 0.892 | 0.937 | null        |
| sent-UK | 0.961 | 0.961 | 0.01  | 0.945 | 0.977 | null        |
| sarc-AU | 0.762 | 0.767 | 0.02  | 0.733 | 0.799 | null        |
| sarc-UK | 0.762 | 0.762 | 0.036 | 0.702 | 0.819 | null        |
| score   | 0.838 | 0.832 | 0.015 | 0.806 | 0.854 | 1.0         |

binding varieties: sent: AU binds 99.7%, sarc: UK binds 54.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.966   | 0.782   | 0.78    | 0.845 |
| 1    | 0.915   | 0.958   | 0.78    | 0.791   | 0.847 |
| 2    | 0.911   | 0.956   | 0.749   | 0.728   | 0.82  |
| 3    | 0.921   | 0.964   | 0.739   | 0.763   | 0.83  |
| 4    | 0.915   | 0.959   | 0.757   | 0.743   | 0.829 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.444     | 0.914    | 0.928     | 0.886  | 0.974 | 0.973 | 0.062 | 0.009 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.526     | 0.961    | 0.96      | 0.965  | 0.994 | 0.994 | 0.028 | 0.007 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.737     | 0.962    | 0.979     | 0.981  | 0.996 | 0.998 | 0.02  | 0.013 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.286     | 0.864    | 0.857     | 0.769  | 0.946 | 0.9   | 0.085 | 0.018 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.75      | 0.952    | 0.975     | 0.977  | 0.992 | 0.997 | 0.026 | 0.011 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.12      | 0.89     | 0.788     | 0.824  | 0.982 | 0.898 | 0.031 | 0.014 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.352     | 0.762    | 0.625     | 0.741  | 0.851 | 0.658 | 0.15  | 0.078 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.092     | 0.762    | 0.52      | 0.618  | 0.942 | 0.578 | 0.06  | 0.068 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.036     | 0.652    | 0.487     | 0.26   | 0.893 | 0.349 | 0.1   | 0.127 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.521     | 0.715    | 0.63      | 0.782  | 0.812 | 0.741 | 0.177 | 0.052 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.259     | 0.715    | 0.52      | 0.621  | 0.819 | 0.58  | 0.167 | 0.187 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.903 | 0.689 |
| 1-10  | UK      | 219 | 0.906 | 0.688 |
| 11-30 | AU      | 717 | 0.851 | 0.725 |
| 11-30 | UK      | 429 | 0.813 | 0.698 |
| 31-80 | AU      | 602 | 0.849 | 0.732 |
| 31-80 | UK      | 385 | 0.925 | 0.718 |
| >80   | AU      | 320 | 0.846 | 0.68  |
| >80   | UK      | 110 | 0.933 | 0.665 |
