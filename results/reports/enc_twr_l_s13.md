# Validation report: `enc_twr_l_s13`

generated 2026-09-18 01:47, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.035 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.78 (valid 0.846, pool 0.856; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.916   | 0.97    | 0.791   | 0.798   | 0.854 | 0.014   | 0.819       |
| global+ukg    | 0.912   | 0.971   | 0.803   | 0.8     | 0.856 | 0.01    | 0.841       |
| variety+ukg   | 0.915   | 0.969   | 0.802   | 0.8     | 0.857 | 0.008   | 0.847       |
| segment+ukg   | 0.915   | 0.969   | 0.798   | 0.8     | 0.856 | 0.007   | 0.846       |
| prev+ukg      | 0.916   | 0.969   | 0.791   | 0.803   | 0.854 | 0.015   | 0.841       |
| prevtuned+ukg | 0.916   | 0.969   | 0.797   | 0.809   | 0.856 | 0.011   | 0.829       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8561**

## Expected test score for `segment+ukg`
**Point estimate 0.8561 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8273, 0.8850]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.915 | 0.915 | 0.014 | 0.894 | 0.938 | null        |
| sent-UK | 0.969 | 0.969 | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.798 | 0.801 | 0.021 | 0.766 | 0.835 | null        |
| sarc-UK | 0.8   | 0.802 | 0.036 | 0.742 | 0.86  | null        |
| score   | 0.856 | 0.85  | 0.015 | 0.825 | 0.873 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 48.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.91    | 0.975   | 0.832   | 0.813   | 0.861 |
| 1    | 0.923   | 0.969   | 0.791   | 0.822   | 0.857 |
| 2    | 0.915   | 0.967   | 0.797   | 0.784   | 0.85  |
| 3    | 0.905   | 0.966   | 0.782   | 0.797   | 0.843 |
| 4    | 0.92    | 0.967   | 0.785   | 0.778   | 0.849 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.447     | 0.915    | 0.925     | 0.89   | 0.974 | 0.975 | 0.073 | 0.054 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.51      | 0.969    | 0.982     | 0.958  | 0.994 | 0.995 | 0.026 | 0.009 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.747     | 0.965    | 0.974     | 0.99   | 0.996 | 0.998 | 0.022 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.285     | 0.863    | 0.857     | 0.766  | 0.946 | 0.913 | 0.1   | 0.086 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.739     | 0.968    | 0.99      | 0.977  | 0.993 | 0.997 | 0.022 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.096     | 0.886    | 0.873     | 0.733  | 0.976 | 0.912 | 0.032 | 0.022 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.329     | 0.798    | 0.686     | 0.761  | 0.901 | 0.77  | 0.14  | 0.112 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.086     | 0.8      | 0.601     | 0.667  | 0.954 | 0.658 | 0.046 | 0.031 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.046     | 0.678    | 0.49      | 0.329  | 0.913 | 0.493 | 0.053 | 0.035 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.482     | 0.767    | 0.696     | 0.799  | 0.854 | 0.786 | 0.187 | 0.164 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.995 | 0.083 | 0.001 | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.241     | 0.762    | 0.601     | 0.669  | 0.862 | 0.664 | 0.127 | 0.108 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.877 | 0.753 |
| 1-10  | UK      | 219 | 0.908 | 0.765 |
| 11-30 | AU      | 717 | 0.841 | 0.779 |
| 11-30 | UK      | 429 | 0.834 | 0.743 |
| 31-80 | AU      | 602 | 0.868 | 0.786 |
| 31-80 | UK      | 385 | 0.899 | 0.794 |
| >80   | AU      | 320 | 0.864 | 0.717 |
| >80   | UK      | 110 | 0.847 | 0.576 |
