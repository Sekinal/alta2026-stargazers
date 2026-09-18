# Validation report: `cls_e5_large_lr_vv`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.026 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.919   | 0.966   | 0.76    | 0.719   | 0.819 | 0.011   | 0.802       |
| global+ukg    | 0.916   | 0.964   | 0.749   | 0.751   | 0.833 | 0.01    | 0.793       |
| variety+ukg   | 0.916   | 0.965   | 0.766   | 0.755   | 0.835 | 0.006   | 0.805       |
| segment+ukg   | 0.913   | 0.966   | 0.763   | 0.755   | 0.834 | 0.005   | 0.802       |
| prev+ukg      | 0.92    | 0.967   | 0.75    | 0.75    | 0.835 | 0.009   | 0.811       |
| prevtuned+ukg | 0.92    | 0.967   | 0.757   | 0.753   | 0.836 | 0.009   | 0.809       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8337**

## Expected test score for `segment+ukg`
**Point estimate 0.8337 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8020, 0.8654]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.913 | 0.914 | 0.014 | 0.891 | 0.937 | null        |
| sent-UK | 0.966 | 0.966 | 0.009 | 0.951 | 0.979 | null        |
| sarc-AU | 0.763 | 0.77  | 0.019 | 0.739 | 0.799 | null        |
| sarc-UK | 0.755 | 0.756 | 0.038 | 0.693 | 0.819 | null        |
| score   | 0.834 | 0.83  | 0.016 | 0.801 | 0.854 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 61.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.903   | 0.971   | 0.774   | 0.758   | 0.831 |
| 1    | 0.917   | 0.964   | 0.768   | 0.747   | 0.832 |
| 2    | 0.92    | 0.967   | 0.757   | 0.729   | 0.824 |
| 3    | 0.916   | 0.964   | 0.745   | 0.766   | 0.831 |
| 4    | 0.908   | 0.964   | 0.769   | 0.771   | 0.839 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.457     | 0.913    | 0.913     | 0.899  | 0.973 | 0.973 | 0.063 | 0.033 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.52      | 0.966    | 0.97      | 0.965  | 0.994 | 0.995 | 0.026 | 0.02  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.734     | 0.958    | 0.978     | 0.977  | 0.994 | 0.998 | 0.025 | 0.039 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.308     | 0.866    | 0.83      | 0.802  | 0.947 | 0.907 | 0.083 | 0.031 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.754     | 0.963    | 0.978     | 0.985  | 0.994 | 0.998 | 0.022 | 0.018 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.097     | 0.88     | 0.856     | 0.725  | 0.977 | 0.873 | 0.034 | 0.014 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.364     | 0.763    | 0.62      | 0.759  | 0.871 | 0.703 | 0.146 | 0.107 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.088     | 0.755    | 0.518     | 0.586  | 0.935 | 0.524 | 0.079 | 0.107 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.017     | 0.512    | 0.167     | 0.041  | 0.899 | 0.322 | 0.058 | 0.064 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.55      | 0.72     | 0.627     | 0.822  | 0.808 | 0.722 | 0.193 | 0.13  |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.988 | 0.038 | 0.004 | 0.023 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.247     | 0.708    | 0.518     | 0.589  | 0.795 | 0.529 | 0.216 | 0.259 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.868 | 0.673 |
| 1-10  | UK      | 219 | 0.901 | 0.65  |
| 11-30 | AU      | 717 | 0.861 | 0.757 |
| 11-30 | UK      | 429 | 0.842 | 0.735 |
| 31-80 | AU      | 602 | 0.852 | 0.736 |
| 31-80 | UK      | 385 | 0.887 | 0.684 |
| >80   | AU      | 320 | 0.877 | 0.65  |
| >80   | UK      | 110 | 0.824 | 0.596 |
