# Validation report: `cls_e5me5q3_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.024 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.60 (valid 0.838, pool 0.846; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.918   | 0.971   | 0.755   | 0.775   | 0.837 | 0.012   | 0.84        |
| global+ukg    | 0.92    | 0.97    | 0.761   | 0.78    | 0.841 | 0.014   | 0.83        |
| variety+ukg   | 0.921   | 0.97    | 0.762   | 0.771   | 0.841 | 0.015   | 0.838       |
| segment+ukg   | 0.921   | 0.97    | 0.777   | 0.771   | 0.846 | 0.012   | 0.838       |
| prev+ukg      | 0.919   | 0.971   | 0.767   | 0.776   | 0.843 | 0.013   | 0.824       |
| prevtuned+ukg | 0.919   | 0.971   | 0.765   | 0.773   | 0.842 | 0.014   | 0.834       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8460**

## Expected test score for `segment+ukg`
**Point estimate 0.8460 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8160, 0.8760]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.921 | 0.923 | 0.013 | 0.9   | 0.943 | null        |
| sent-UK | 0.97  | 0.97  | 0.008 | 0.956 | 0.984 | null        |
| sarc-AU | 0.777 | 0.783 | 0.019 | 0.752 | 0.814 | null        |
| sarc-UK | 0.771 | 0.772 | 0.036 | 0.711 | 0.832 | null        |
| score   | 0.846 | 0.841 | 0.015 | 0.814 | 0.864 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 61.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.971   | 0.785   | 0.784   | 0.851 |
| 1    | 0.926   | 0.961   | 0.792   | 0.786   | 0.856 |
| 2    | 0.916   | 0.97    | 0.769   | 0.737   | 0.827 |
| 3    | 0.918   | 0.975   | 0.75    | 0.782   | 0.834 |
| 4    | 0.928   | 0.975   | 0.787   | 0.769   | 0.849 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.448     | 0.921    | 0.932     | 0.898  | 0.977 | 0.977 | 0.059 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.525     | 0.97     | 0.969     | 0.974  | 0.995 | 0.995 | 0.023 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.976    | 0.985     | 0.99   | 0.997 | 0.999 | 0.017 | 0.013 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.291     | 0.871    | 0.86      | 0.784  | 0.951 | 0.91  | 0.081 | 0.012 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.754     | 0.966    | 0.979     | 0.986  | 0.994 | 0.998 | 0.02  | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.113     | 0.909    | 0.845     | 0.832  | 0.982 | 0.907 | 0.029 | 0.014 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.357     | 0.777    | 0.64      | 0.771  | 0.871 | 0.712 | 0.139 | 0.064 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.089     | 0.771    | 0.542     | 0.622  | 0.946 | 0.612 | 0.051 | 0.048 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.02      | 0.576    | 0.409     | 0.123  | 0.901 | 0.368 | 0.086 | 0.107 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.538     | 0.736    | 0.645     | 0.828  | 0.832 | 0.762 | 0.167 | 0.041 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.25      | 0.726    | 0.542     | 0.625  | 0.834 | 0.614 | 0.143 | 0.133 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.894 | 0.726 |
| 1-10  | UK      | 219 | 0.92  | 0.685 |
| 11-30 | AU      | 717 | 0.858 | 0.756 |
| 11-30 | UK      | 429 | 0.856 | 0.744 |
| 31-80 | AU      | 602 | 0.869 | 0.742 |
| 31-80 | UK      | 385 | 0.939 | 0.71  |
| >80   | AU      | 320 | 0.849 | 0.683 |
| >80   | UK      | 110 | 0.877 | 0.559 |
