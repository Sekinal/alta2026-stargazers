# Validation report: `cls_e5me5_svm`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.031 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.32 (valid 0.826, pool 0.843; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.917   | 0.969   | 0.757   | 0.684   | 0.8   | 0.013   | 0.762       |
| global+ukg    | 0.916   | 0.967   | 0.766   | 0.771   | 0.841 | 0.011   | 0.834       |
| variety+ukg   | 0.916   | 0.969   | 0.771   | 0.781   | 0.843 | 0.015   | 0.826       |
| segment+ukg   | 0.915   | 0.968   | 0.77    | 0.781   | 0.843 | 0.015   | 0.826       |
| prev+ukg      | 0.921   | 0.968   | 0.762   | 0.787   | 0.841 | 0.012   | 0.85        |
| prevtuned+ukg | 0.921   | 0.968   | 0.764   | 0.78    | 0.842 | 0.011   | 0.839       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8427**

## Expected test score for `segment+ukg`
**Point estimate 0.8427 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8145, 0.8708]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.915 | 0.916 | 0.014 | 0.892 | 0.938 | null        |
| sent-UK | 0.968 | 0.968 | 0.009 | 0.953 | 0.982 | null        |
| sarc-AU | 0.77  | 0.776 | 0.021 | 0.74  | 0.81  | null        |
| sarc-UK | 0.781 | 0.784 | 0.038 | 0.72  | 0.845 | null        |
| score   | 0.843 | 0.839 | 0.014 | 0.815 | 0.861 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 43.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.906   | 0.971   | 0.78    | 0.804   | 0.843 |
| 1    | 0.922   | 0.963   | 0.785   | 0.788   | 0.853 |
| 2    | 0.92    | 0.966   | 0.768   | 0.716   | 0.818 |
| 3    | 0.917   | 0.97    | 0.733   | 0.801   | 0.825 |
| 4    | 0.912   | 0.969   | 0.782   | 0.787   | 0.847 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.476     | 0.915    | 0.9       | 0.921  | 0.977 | 0.976 | 0.058 | 0.011 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.515     | 0.968    | 0.976     | 0.962  | 0.995 | 0.995 | 0.024 | 0.01  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.746     | 0.969    | 0.976     | 0.991  | 0.996 | 0.999 | 0.018 | 0.016 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.33      | 0.867    | 0.807     | 0.834  | 0.951 | 0.911 | 0.08  | 0.012 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.963    | 0.981     | 0.981  | 0.993 | 0.998 | 0.022 | 0.009 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.094     | 0.892    | 0.897     | 0.733  | 0.983 | 0.913 | 0.028 | 0.015 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.325     | 0.77     | 0.653     | 0.716  | 0.874 | 0.727 | 0.13  | 0.017 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.082     | 0.781    | 0.58      | 0.614  | 0.947 | 0.611 | 0.045 | 0.014 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.019     | 0.567    | 0.4       | 0.11   | 0.892 | 0.324 | 0.054 | 0.031 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.49      | 0.733    | 0.659     | 0.768  | 0.821 | 0.749 | 0.171 | 0.037 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.231     | 0.741    | 0.58      | 0.617  | 0.837 | 0.614 | 0.124 | 0.037 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.887 | 0.705 |
| 1-10  | UK      | 219 | 0.909 | 0.7   |
| 11-30 | AU      | 717 | 0.865 | 0.751 |
| 11-30 | UK      | 429 | 0.852 | 0.738 |
| 31-80 | AU      | 602 | 0.857 | 0.754 |
| 31-80 | UK      | 385 | 0.903 | 0.749 |
| >80   | AU      | 320 | 0.841 | 0.678 |
| >80   | UK      | 110 | 0.824 | 0.619 |
