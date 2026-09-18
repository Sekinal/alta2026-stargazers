# Validation report: `enc_deb_l_ukc_s14`

generated 2026-09-18 04:22, code 3cf922e-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.014 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.79 (valid 0.839, pool 0.862; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.927   | 0.972   | 0.783   | 0.801   | 0.855 | 0.005   | 0.815       |
| global+ukg    | 0.925   | 0.971   | 0.801   | 0.819   | 0.863 | 0.007   | 0.828       |
| variety+ukg   | 0.925   | 0.969   | 0.801   | 0.808   | 0.863 | 0.01    | 0.841       |
| segment+ukg   | 0.924   | 0.97    | 0.8     | 0.808   | 0.862 | 0.011   | 0.839       |
| prev+ukg      | 0.926   | 0.972   | 0.793   | 0.818   | 0.86  | 0.009   | 0.834       |
| prevtuned+ukg | 0.926   | 0.972   | 0.797   | 0.808   | 0.862 | 0.01    | 0.838       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8621**

## Expected test score for `segment+ukg`
**Point estimate 0.8621 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8346, 0.8895]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.924 | 0.925 | 0.013 | 0.903 | 0.946 | null        |
| sent-UK | 0.97  | 0.97  | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.8   | 0.804 | 0.019 | 0.773 | 0.835 | null        |
| sarc-UK | 0.808 | 0.81  | 0.036 | 0.747 | 0.867 | null        |
| score   | 0.862 | 0.858 | 0.014 | 0.834 | 0.879 | 1.0         |

binding varieties: sent: AU binds 99.7%, sarc: UK binds 42.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.98    | 0.84    | 0.83    | 0.869 |
| 1    | 0.928   | 0.972   | 0.801   | 0.85    | 0.865 |
| 2    | 0.93    | 0.969   | 0.79    | 0.758   | 0.844 |
| 3    | 0.928   | 0.962   | 0.778   | 0.818   | 0.853 |
| 4    | 0.927   | 0.965   | 0.79    | 0.772   | 0.849 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.47      | 0.924    | 0.914     | 0.925  | 0.976 | 0.975 | 0.064 | 0.043 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.97     | 0.976     | 0.966  | 0.994 | 0.995 | 0.026 | 0.014 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.975    | 0.981     | 0.992  | 0.996 | 0.998 | 0.015 | 0.008 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.324     | 0.879    | 0.831     | 0.842  | 0.949 | 0.909 | 0.09  | 0.078 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.965    | 0.983     | 0.982  | 0.994 | 0.998 | 0.021 | 0.012 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.904    | 0.887     | 0.779  | 0.968 | 0.898 | 0.033 | 0.019 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.342     | 0.8      | 0.68      | 0.782  | 0.898 | 0.759 | 0.139 | 0.107 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.082     | 0.808    | 0.63      | 0.663  | 0.958 | 0.696 | 0.047 | 0.018 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.042     | 0.677    | 0.511     | 0.315  | 0.925 | 0.527 | 0.051 | 0.034 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.502     | 0.767    | 0.687     | 0.823  | 0.848 | 0.774 | 0.187 | 0.148 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.564 | 0.001 | 0.001 | 0.002 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.229     | 0.773    | 0.63      | 0.665  | 0.874 | 0.705 | 0.128 | 0.103 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.892 | 0.768 |
| 1-10  | UK      | 219 | 0.916 | 0.734 |
| 11-30 | AU      | 717 | 0.878 | 0.779 |
| 11-30 | UK      | 429 | 0.885 | 0.782 |
| 31-80 | AU      | 602 | 0.861 | 0.787 |
| 31-80 | UK      | 385 | 0.906 | 0.772 |
| >80   | AU      | 320 | 0.883 | 0.696 |
| >80   | UK      | 110 | 0.847 | 0.678 |
