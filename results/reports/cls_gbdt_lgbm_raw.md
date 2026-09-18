# Validation report: `cls_gbdt_lgbm_raw`

generated 2026-09-18 01:51, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.040 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.39 (valid 0.813, pool 0.832; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.917   | 0.968   | 0.74    | 0.622   | 0.769 | 0.016   | 0.734       |
| global+ukg    | 0.917   | 0.967   | 0.751   | 0.74    | 0.829 | 0.011   | 0.8         |
| variety+ukg   | 0.918   | 0.965   | 0.773   | 0.748   | 0.833 | 0.014   | 0.816       |
| segment+ukg   | 0.917   | 0.963   | 0.774   | 0.748   | 0.832 | 0.015   | 0.813       |
| prev+ukg      | 0.922   | 0.97    | 0.748   | 0.733   | 0.827 | 0.01    | 0.793       |
| prevtuned+ukg | 0.922   | 0.97    | 0.768   | 0.749   | 0.836 | 0.013   | 0.812       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8325**

## Expected test score for `segment+ukg`
**Point estimate 0.8325 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8022, 0.8627]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.917 | 0.918 | 0.014 | 0.894 | 0.94  | null        |
| sent-UK | 0.963 | 0.963 | 0.009 | 0.948 | 0.977 | null        |
| sarc-AU | 0.774 | 0.782 | 0.018 | 0.753 | 0.811 | null        |
| sarc-UK | 0.748 | 0.751 | 0.032 | 0.697 | 0.801 | null        |
| score   | 0.832 | 0.833 | 0.015 | 0.806 | 0.856 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 80.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.904   | 0.96    | 0.787   | 0.714   | 0.809 |
| 1    | 0.921   | 0.958   | 0.772   | 0.78    | 0.847 |
| 2    | 0.923   | 0.961   | 0.78    | 0.732   | 0.828 |
| 3    | 0.905   | 0.97    | 0.774   | 0.755   | 0.83  |
| 4    | 0.933   | 0.965   | 0.757   | 0.755   | 0.844 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.451     | 0.917    | 0.924     | 0.897  | 0.977 | 0.975 | 0.06  | 0.016 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.963    | 0.965     | 0.964  | 0.995 | 0.996 | 0.024 | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.962    | 0.977     | 0.982  | 0.994 | 0.997 | 0.021 | 0.016 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.296     | 0.871    | 0.853     | 0.791  | 0.953 | 0.908 | 0.081 | 0.03  |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.964    | 0.979     | 0.984  | 0.995 | 0.998 | 0.019 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.107     | 0.86     | 0.779     | 0.725  | 0.981 | 0.895 | 0.034 | 0.018 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.364     | 0.774    | 0.633     | 0.777  | 0.871 | 0.713 | 0.133 | 0.03  |
| sarc-UK        | 3208 | 249   | 0.078      | 0.124     | 0.748    | 0.444     | 0.711  | 0.945 | 0.538 | 0.048 | 0.021 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.002     | 0.496    | 0.5       | 0.014  | 0.886 | 0.316 | 0.055 | 0.03  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.559     | 0.729    | 0.633     | 0.843  | 0.812 | 0.733 | 0.176 | 0.039 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.955 | 0.011 | 0.001 | 0.005 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.349     | 0.684    | 0.444     | 0.714  | 0.822 | 0.539 | 0.134 | 0.053 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.886 | 0.723 |
| 1-10  | UK      | 219 | 0.87  | 0.648 |
| 11-30 | AU      | 717 | 0.858 | 0.752 |
| 11-30 | UK      | 429 | 0.819 | 0.681 |
| 31-80 | AU      | 602 | 0.87  | 0.744 |
| 31-80 | UK      | 385 | 0.868 | 0.683 |
| >80   | AU      | 320 | 0.857 | 0.65  |
| >80   | UK      | 110 | 0.877 | 0.568 |
