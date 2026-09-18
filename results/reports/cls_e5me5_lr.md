# Validation report: `cls_e5me5_lr`

generated 2026-09-17 23:55, code dc7268a-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.019 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.92    | 0.968   | 0.754   | 0.777   | 0.837 | 0.008   | 0.844       |
| global+ukg    | 0.916   | 0.968   | 0.754   | 0.772   | 0.835 | 0.006   | 0.837       |
| variety+ukg   | 0.916   | 0.968   | 0.753   | 0.788   | 0.834 | 0.006   | 0.844       |
| segment+ukg   | 0.917   | 0.969   | 0.777   | 0.788   | 0.847 | 0.008   | 0.852       |
| prev+ukg      | 0.925   | 0.968   | 0.763   | 0.785   | 0.844 | 0.012   | 0.849       |
| prevtuned+ukg | 0.925   | 0.968   | 0.763   | 0.787   | 0.844 | 0.008   | 0.854       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8469**

## Expected test score for `segment+ukg`
**Point estimate 0.8469 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8199, 0.8740]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.917 | 0.918 | 0.014 | 0.895 | 0.941 | null        |
| sent-UK | 0.969 | 0.969 | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.777 | 0.783 | 0.019 | 0.752 | 0.814 | null        |
| sarc-UK | 0.788 | 0.789 | 0.035 | 0.729 | 0.847 | null        |
| score   | 0.847 | 0.844 | 0.014 | 0.82  | 0.865 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 43.3%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.905   | 0.974   | 0.794   | 0.786   | 0.845 |
| 1    | 0.924   | 0.966   | 0.789   | 0.804   | 0.856 |
| 2    | 0.92    | 0.967   | 0.768   | 0.773   | 0.844 |
| 3    | 0.922   | 0.97    | 0.75    | 0.782   | 0.836 |
| 4    | 0.917   | 0.97    | 0.783   | 0.791   | 0.85  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.477     | 0.917    | 0.901     | 0.925  | 0.978 | 0.977 | 0.058 | 0.011 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.969    | 0.975     | 0.966  | 0.995 | 0.995 | 0.024 | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.971    | 0.98      | 0.99   | 0.996 | 0.999 | 0.018 | 0.015 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.334     | 0.87     | 0.806     | 0.844  | 0.953 | 0.913 | 0.079 | 0.014 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.965    | 0.981     | 0.984  | 0.993 | 0.998 | 0.022 | 0.01  |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.095     | 0.898    | 0.899     | 0.748  | 0.984 | 0.912 | 0.028 | 0.015 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.352     | 0.777    | 0.644     | 0.763  | 0.867 | 0.701 | 0.14  | 0.065 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.092     | 0.788    | 0.565     | 0.667  | 0.947 | 0.612 | 0.053 | 0.054 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.02      | 0.598    | 0.5       | 0.151  | 0.897 | 0.357 | 0.085 | 0.102 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.529     | 0.735    | 0.647     | 0.816  | 0.825 | 0.753 | 0.17  | 0.045 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.257     | 0.746    | 0.565     | 0.669  | 0.837 | 0.614 | 0.147 | 0.15  |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.898 | 0.711 |
| 1-10  | UK      | 219 | 0.921 | 0.72  |
| 11-30 | AU      | 717 | 0.859 | 0.763 |
| 11-30 | UK      | 429 | 0.85  | 0.742 |
| 31-80 | AU      | 602 | 0.865 | 0.739 |
| 31-80 | UK      | 385 | 0.912 | 0.74  |
| >80   | AU      | 320 | 0.841 | 0.683 |
| >80   | UK      | 110 | 0.824 | 0.665 |
