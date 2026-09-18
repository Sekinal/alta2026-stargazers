# Validation report: `cls_tw_irony_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.029 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.69 (valid 0.833, pool 0.823; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.891   | 0.947   | 0.734   | 0.763   | 0.813 | 0.011   | 0.834       |
| global+ukg    | 0.891   | 0.944   | 0.723   | 0.76    | 0.807 | 0.009   | 0.83        |
| variety+ukg   | 0.891   | 0.947   | 0.724   | 0.766   | 0.808 | 0.014   | 0.83        |
| segment+ukg   | 0.889   | 0.944   | 0.758   | 0.766   | 0.823 | 0.007   | 0.833       |
| prev+ukg      | 0.886   | 0.946   | 0.741   | 0.757   | 0.814 | 0.014   | 0.817       |
| prevtuned+ukg | 0.889   | 0.946   | 0.753   | 0.759   | 0.821 | 0.009   | 0.836       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8232**

## Expected test score for `segment+ukg`
**Point estimate 0.8232 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7947, 0.8518]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.889 | 0.889 | 0.015 | 0.864 | 0.913 | null        |
| sent-UK | 0.944 | 0.944 | 0.011 | 0.925 | 0.961 | null        |
| sarc-AU | 0.758 | 0.764 | 0.02  | 0.729 | 0.797 | null        |
| sarc-UK | 0.766 | 0.765 | 0.035 | 0.709 | 0.822 | null        |
| score   | 0.823 | 0.819 | 0.014 | 0.793 | 0.841 | 0.998       |

binding varieties: sent: AU binds 99.8%, sarc: UK binds 49.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.879   | 0.953   | 0.77    | 0.779   | 0.825 |
| 1    | 0.885   | 0.945   | 0.749   | 0.795   | 0.817 |
| 2    | 0.911   | 0.942   | 0.777   | 0.742   | 0.826 |
| 3    | 0.877   | 0.934   | 0.739   | 0.746   | 0.808 |
| 4    | 0.892   | 0.944   | 0.752   | 0.766   | 0.822 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.439     | 0.889    | 0.904     | 0.854  | 0.954 | 0.953 | 0.081 | 0.008 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.944    | 0.947     | 0.946  | 0.986 | 0.988 | 0.042 | 0.02  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.732     | 0.918    | 0.958     | 0.954  | 0.985 | 0.994 | 0.044 | 0.032 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.281     | 0.84     | 0.828     | 0.73   | 0.919 | 0.859 | 0.101 | 0.02  |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.926    | 0.961     | 0.966  | 0.984 | 0.994 | 0.041 | 0.03  |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.106     | 0.853    | 0.769     | 0.71   | 0.959 | 0.788 | 0.044 | 0.013 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.34      | 0.758    | 0.627     | 0.718  | 0.84  | 0.651 | 0.159 | 0.106 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.1       | 0.766    | 0.509     | 0.655  | 0.939 | 0.527 | 0.06  | 0.064 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.012     | 0.528    | 0.308     | 0.055  | 0.9   | 0.332 | 0.114 | 0.199 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.515     | 0.714    | 0.631     | 0.775  | 0.8   | 0.729 | 0.183 | 0.056 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.28      | 0.717    | 0.509     | 0.657  | 0.808 | 0.529 | 0.167 | 0.178 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.849 | 0.703 |
| 1-10  | UK      | 219 | 0.85  | 0.693 |
| 11-30 | AU      | 717 | 0.81  | 0.722 |
| 11-30 | UK      | 429 | 0.854 | 0.735 |
| 31-80 | AU      | 602 | 0.85  | 0.732 |
| 31-80 | UK      | 385 | 0.839 | 0.696 |
| >80   | AU      | 320 | 0.857 | 0.671 |
| >80   | UK      | 110 | 0.847 | 0.51  |
