# Validation report: `cls_gbdt_lgbm`

generated 2026-09-18 00:26, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [WARN] per-fold score spread 0.063 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.73 (valid 0.809, pool 0.832; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.913   | 0.963   | 0.752   | 0.663   | 0.788 | 0.027   | 0.724       |
| global+ukg    | 0.915   | 0.965   | 0.747   | 0.732   | 0.824 | 0.013   | 0.793       |
| variety+ukg   | 0.914   | 0.967   | 0.761   | 0.75    | 0.832 | 0.007   | 0.81        |
| segment+ukg   | 0.913   | 0.963   | 0.761   | 0.75    | 0.832 | 0.007   | 0.809       |
| prev+ukg      | 0.915   | 0.966   | 0.75    | 0.733   | 0.824 | 0.015   | 0.764       |
| prevtuned+ukg | 0.915   | 0.966   | 0.754   | 0.749   | 0.832 | 0.006   | 0.815       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8318**

## Expected test score for `segment+ukg`
**Point estimate 0.8318 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8046, 0.8590]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.913 | 0.914 | 0.014 | 0.891 | 0.935 | null        |
| sent-UK | 0.963 | 0.963 | 0.009 | 0.948 | 0.977 | null        |
| sarc-AU | 0.761 | 0.767 | 0.02  | 0.735 | 0.799 | null        |
| sarc-UK | 0.75  | 0.754 | 0.03  | 0.703 | 0.804 | null        |
| score   | 0.832 | 0.83  | 0.014 | 0.806 | 0.851 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 65.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.901   | 0.967   | 0.772   | 0.753   | 0.827 |
| 1    | 0.91    | 0.955   | 0.768   | 0.812   | 0.839 |
| 2    | 0.916   | 0.961   | 0.765   | 0.735   | 0.826 |
| 3    | 0.907   | 0.972   | 0.745   | 0.739   | 0.823 |
| 4    | 0.933   | 0.961   | 0.752   | 0.714   | 0.824 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.447     | 0.913    | 0.923     | 0.889  | 0.975 | 0.973 | 0.063 | 0.022 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.963    | 0.97      | 0.959  | 0.994 | 0.995 | 0.026 | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.742     | 0.958    | 0.974     | 0.982  | 0.994 | 0.998 | 0.022 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.289     | 0.864    | 0.853     | 0.773  | 0.948 | 0.904 | 0.085 | 0.029 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.958    | 0.98      | 0.978  | 0.994 | 0.998 | 0.022 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.88     | 0.836     | 0.74   | 0.977 | 0.89  | 0.034 | 0.013 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.36      | 0.761    | 0.619     | 0.751  | 0.867 | 0.697 | 0.135 | 0.021 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.13      | 0.75     | 0.441     | 0.739  | 0.939 | 0.515 | 0.049 | 0.012 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.017     | 0.581    | 0.5       | 0.123  | 0.901 | 0.36  | 0.052 | 0.022 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.544     | 0.712    | 0.621     | 0.805  | 0.8   | 0.715 | 0.179 | 0.026 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.861 | 0.003 | 0.001 | 0.007 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.365     | 0.685    | 0.441     | 0.742  | 0.803 | 0.517 | 0.136 | 0.027 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.883 | 0.701 |
| 1-10  | UK      | 219 | 0.902 | 0.636 |
| 11-30 | AU      | 717 | 0.846 | 0.722 |
| 11-30 | UK      | 429 | 0.848 | 0.688 |
| 31-80 | AU      | 602 | 0.856 | 0.719 |
| 31-80 | UK      | 385 | 0.874 | 0.676 |
| >80   | AU      | 320 | 0.878 | 0.681 |
| >80   | UK      | 110 | 0.847 | 0.535 |
