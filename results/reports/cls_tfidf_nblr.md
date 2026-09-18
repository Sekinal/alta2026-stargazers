# Validation report: `cls_tfidf_nblr`

generated 2026-09-17 23:48, code 6844a92, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.059 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.85    | 0.93    | 0.7     | 0.575   | 0.712 | 0.024   | 0.731       |
| global+ukg    | 0.852   | 0.929   | 0.641   | 0.627   | 0.739 | 0.021   | 0.759       |
| variety+ukg   | 0.855   | 0.928   | 0.702   | 0.618   | 0.737 | 0.018   | 0.78        |
| segment+ukg   | 0.854   | 0.93    | 0.701   | 0.618   | 0.736 | 0.018   | 0.782       |
| prev+ukg      | 0.846   | 0.93    | 0.678   | 0.617   | 0.732 | 0.013   | 0.76        |
| prevtuned+ukg | 0.846   | 0.93    | 0.687   | 0.646   | 0.746 | 0.014   | 0.773       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7362**

## Expected test score for `segment+ukg`
**Point estimate 0.7362 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7031, 0.7693]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.854 | 0.855 | 0.016 | 0.828 | 0.88  | null        |
| sent-UK | 0.93  | 0.929 | 0.012 | 0.909 | 0.948 | null        |
| sarc-AU | 0.701 | 0.709 | 0.019 | 0.677 | 0.739 | null        |
| sarc-UK | 0.618 | 0.618 | 0.03  | 0.568 | 0.667 | null        |
| score   | 0.736 | 0.736 | 0.017 | 0.709 | 0.764 | 0.01        |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 99.4%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.848   | 0.936   | 0.697   | 0.599   | 0.724 |
| 1    | 0.858   | 0.936   | 0.709   | 0.603   | 0.731 |
| 2    | 0.857   | 0.93    | 0.723   | 0.572   | 0.715 |
| 3    | 0.868   | 0.917   | 0.681   | 0.63    | 0.749 |
| 4    | 0.839   | 0.929   | 0.694   | 0.563   | 0.701 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.407     | 0.854    | 0.894     | 0.784  | 0.925 | 0.927 | 0.106 | 0.02  |
| sent-UK        | 3208 | 1676  | 0.522      | 0.512     | 0.93     | 0.942     | 0.923  | 0.976 | 0.98  | 0.054 | 0.018 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.746     | 0.915    | 0.949     | 0.963  | 0.97  | 0.988 | 0.054 | 0.034 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.226     | 0.766    | 0.796     | 0.562  | 0.863 | 0.777 | 0.135 | 0.041 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.912    | 0.954     | 0.958  | 0.975 | 0.991 | 0.05  | 0.024 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.08      | 0.779    | 0.728     | 0.511  | 0.891 | 0.642 | 0.062 | 0.016 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.408     | 0.701    | 0.529     | 0.728  | 0.802 | 0.568 | 0.17  | 0.087 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.157     | 0.618    | 0.244     | 0.494  | 0.875 | 0.319 | 0.062 | 0.032 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.007     | 0.494    | 0.143     | 0.014  | 0.806 | 0.231 | 0.06  | 0.046 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.624     | 0.619    | 0.531     | 0.79   | 0.698 | 0.582 | 0.229 | 0.109 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.994 | 0.071 | 0.001 | 0.004 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.442     | 0.498    | 0.244     | 0.496  | 0.589 | 0.32  | 0.174 | 0.092 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.772 | 0.588 |
| 1-10  | UK      | 219 | 0.835 | 0.472 |
| 11-30 | AU      | 717 | 0.748 | 0.665 |
| 11-30 | UK      | 429 | 0.713 | 0.512 |
| 31-80 | AU      | 602 | 0.758 | 0.59  |
| 31-80 | UK      | 385 | 0.796 | 0.498 |
| >80   | AU      | 320 | 0.781 | 0.591 |
| >80   | UK      | 110 | 0.481 | 0.395 |
