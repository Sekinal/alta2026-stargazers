# Validation report: `stk_sel_lr_c01`

generated 2026-09-18 00:34, code 94275f4-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.021 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.01 (valid 0.843, pool 0.856; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.927   | 0.974   | 0.786   | 0.756   | 0.842 | 0.008   | 0.801       |
| global+ukg    | 0.925   | 0.973   | 0.788   | 0.792   | 0.856 | 0.012   | 0.838       |
| variety+ukg   | 0.925   | 0.973   | 0.787   | 0.801   | 0.856 | 0.009   | 0.845       |
| segment+ukg   | 0.924   | 0.973   | 0.789   | 0.801   | 0.856 | 0.009   | 0.843       |
| prev+ukg      | 0.933   | 0.973   | 0.788   | 0.794   | 0.86  | 0.013   | 0.83        |
| prevtuned+ukg | 0.933   | 0.973   | 0.792   | 0.796   | 0.862 | 0.014   | 0.834       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8562**

## Expected test score for `segment+ukg`
**Point estimate 0.8562 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8292, 0.8832]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.924 | 0.925 | 0.013 | 0.902 | 0.946 | null        |
| sent-UK | 0.973 | 0.973 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.789 | 0.794 | 0.02  | 0.761 | 0.826 | null        |
| sarc-UK | 0.801 | 0.804 | 0.035 | 0.744 | 0.859 | null        |
| score   | 0.856 | 0.853 | 0.014 | 0.83  | 0.875 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 40.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.901   | 0.972   | 0.824   | 0.822   | 0.861 |
| 1    | 0.929   | 0.969   | 0.79    | 0.801   | 0.859 |
| 2    | 0.934   | 0.97    | 0.776   | 0.791   | 0.855 |
| 3    | 0.923   | 0.98    | 0.764   | 0.833   | 0.843 |
| 4    | 0.93    | 0.975   | 0.79    | 0.756   | 0.843 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.453     | 0.924    | 0.929     | 0.906  | 0.983 | 0.982 | 0.052 | 0.009 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.973    | 0.977     | 0.971  | 0.996 | 0.996 | 0.02  | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.734     | 0.97     | 0.985     | 0.984  | 0.998 | 0.999 | 0.014 | 0.008 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.302     | 0.878    | 0.856     | 0.809  | 0.962 | 0.929 | 0.072 | 0.012 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.974    | 0.985     | 0.989  | 0.996 | 0.998 | 0.016 | 0.002 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.1       | 0.897    | 0.877     | 0.763  | 0.982 | 0.913 | 0.028 | 0.008 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.318     | 0.789    | 0.684     | 0.733  | 0.895 | 0.761 | 0.12  | 0.009 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.088     | 0.801    | 0.598     | 0.675  | 0.954 | 0.647 | 0.041 | 0.003 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.033     | 0.63     | 0.457     | 0.219  | 0.915 | 0.406 | 0.051 | 0.022 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.472     | 0.759    | 0.692     | 0.778  | 0.848 | 0.782 | 0.157 | 0.014 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.177 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.246     | 0.763    | 0.598     | 0.677  | 0.86  | 0.649 | 0.114 | 0.021 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.914 | 0.744 |
| 1-10  | UK      | 219 | 0.893 | 0.72  |
| 11-30 | AU      | 717 | 0.869 | 0.771 |
| 11-30 | UK      | 429 | 0.906 | 0.745 |
| 31-80 | AU      | 602 | 0.867 | 0.776 |
| 31-80 | UK      | 385 | 0.899 | 0.794 |
| >80   | AU      | 320 | 0.851 | 0.706 |
| >80   | UK      | 110 | 0.821 | 0.665 |
