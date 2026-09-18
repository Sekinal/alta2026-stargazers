# Validation report: `enc_deb_l_ls05_s15`

generated 2026-09-18 03:21, code ae44cce-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.023 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -3.82 (valid 0.809, pool 0.860; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.928   | 0.971   | 0.788   | 0.782   | 0.855 | 0.008   | 0.778       |
| global+ukg    | 0.928   | 0.971   | 0.793   | 0.798   | 0.861 | 0.005   | 0.809       |
| variety+ukg   | 0.928   | 0.973   | 0.79    | 0.798   | 0.859 | 0.005   | 0.809       |
| segment+ukg   | 0.928   | 0.973   | 0.792   | 0.798   | 0.86  | 0.006   | 0.809       |
| prev+ukg      | 0.927   | 0.971   | 0.8     | 0.8     | 0.863 | 0.01    | 0.814       |
| prevtuned+ukg | 0.927   | 0.971   | 0.798   | 0.801   | 0.862 | 0.008   | 0.823       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8600**

## Expected test score for `segment+ukg`
**Point estimate 0.8600 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8314, 0.8886]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.928 | 0.928 | 0.013 | 0.907 | 0.949 | null        |
| sent-UK | 0.973 | 0.973 | 0.008 | 0.961 | 0.984 | null        |
| sarc-AU | 0.792 | 0.796 | 0.021 | 0.762 | 0.83  | null        |
| sarc-UK | 0.798 | 0.799 | 0.037 | 0.736 | 0.857 | null        |
| score   | 0.86  | 0.854 | 0.014 | 0.829 | 0.877 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 45.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.988   | 0.818   | 0.85    | 0.863 |
| 1    | 0.922   | 0.975   | 0.775   | 0.79    | 0.848 |
| 2    | 0.938   | 0.974   | 0.799   | 0.76    | 0.849 |
| 3    | 0.93    | 0.961   | 0.779   | 0.819   | 0.854 |
| 4    | 0.939   | 0.967   | 0.786   | 0.766   | 0.853 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.462     | 0.928    | 0.925     | 0.92   | 0.977 | 0.977 | 0.064 | 0.044 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.513     | 0.973    | 0.983     | 0.965  | 0.993 | 0.994 | 0.026 | 0.01  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.738     | 0.98     | 0.987     | 0.991  | 0.997 | 0.999 | 0.014 | 0.003 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.314     | 0.882    | 0.846     | 0.831  | 0.95  | 0.908 | 0.091 | 0.075 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.737     | 0.97     | 0.992     | 0.977  | 0.993 | 0.997 | 0.021 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.108     | 0.911    | 0.87      | 0.817  | 0.967 | 0.895 | 0.036 | 0.024 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.317     | 0.792    | 0.69      | 0.736  | 0.886 | 0.755 | 0.135 | 0.094 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.079     | 0.798    | 0.62      | 0.635  | 0.907 | 0.649 | 0.044 | 0.019 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.046     | 0.711    | 0.56      | 0.384  | 0.864 | 0.493 | 0.049 | 0.033 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.462     | 0.759    | 0.697     | 0.767  | 0.842 | 0.773 | 0.181 | 0.132 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.848 | 0.003 | 0.002 | 0.021 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.223     | 0.762    | 0.62      | 0.637  | 0.857 | 0.681 | 0.12  | 0.082 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.887 | 0.737 |
| 1-10  | UK      | 219 | 0.924 | 0.735 |
| 11-30 | AU      | 717 | 0.882 | 0.78  |
| 11-30 | UK      | 429 | 0.889 | 0.754 |
| 31-80 | AU      | 602 | 0.879 | 0.779 |
| 31-80 | UK      | 385 | 0.917 | 0.78  |
| >80   | AU      | 320 | 0.858 | 0.691 |
| >80   | UK      | 110 | 0.847 | 0.663 |
