# Validation report: `cls_gbdt_lgbm_nlog`

generated 2026-09-18 01:51, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.031 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.88 (valid 0.834, pool 0.847; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.919   | 0.972   | 0.773   | 0.772   | 0.846 | 0.014   | 0.804       |
| global+ukg    | 0.92    | 0.972   | 0.774   | 0.786   | 0.847 | 0.009   | 0.834       |
| variety+ukg   | 0.919   | 0.97    | 0.772   | 0.781   | 0.846 | 0.011   | 0.831       |
| segment+ukg   | 0.92    | 0.97    | 0.775   | 0.781   | 0.847 | 0.012   | 0.834       |
| prev+ukg      | 0.921   | 0.972   | 0.776   | 0.796   | 0.848 | 0.011   | 0.83        |
| prevtuned+ukg | 0.921   | 0.972   | 0.778   | 0.788   | 0.849 | 0.013   | 0.829       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8474**

## Expected test score for `segment+ukg`
**Point estimate 0.8474 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8182, 0.8767]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.92  | 0.921 | 0.014 | 0.897 | 0.943 | null        |
| sent-UK | 0.97  | 0.97  | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.775 | 0.782 | 0.02  | 0.748 | 0.815 | null        |
| sarc-UK | 0.781 | 0.783 | 0.038 | 0.719 | 0.844 | null        |
| score   | 0.847 | 0.843 | 0.015 | 0.818 | 0.865 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 48.4%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.913   | 0.975   | 0.803   | 0.794   | 0.854 |
| 1    | 0.927   | 0.967   | 0.785   | 0.797   | 0.856 |
| 2    | 0.922   | 0.961   | 0.787   | 0.761   | 0.841 |
| 3    | 0.917   | 0.978   | 0.733   | 0.788   | 0.825 |
| 4    | 0.92    | 0.967   | 0.764   | 0.764   | 0.842 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.479     | 0.92     | 0.902     | 0.929  | 0.979 | 0.977 | 0.057 | 0.015 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.525     | 0.97     | 0.969     | 0.974  | 0.996 | 0.996 | 0.021 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.747     | 0.97     | 0.976     | 0.992  | 0.996 | 0.998 | 0.017 | 0.011 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.334     | 0.874    | 0.812     | 0.85   | 0.955 | 0.914 | 0.079 | 0.025 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.758     | 0.964    | 0.976     | 0.989  | 0.995 | 0.998 | 0.019 | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.104     | 0.906    | 0.874     | 0.794  | 0.987 | 0.928 | 0.025 | 0.015 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.321     | 0.775    | 0.663     | 0.718  | 0.884 | 0.734 | 0.126 | 0.013 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.077     | 0.781    | 0.597     | 0.594  | 0.956 | 0.636 | 0.041 | 0.006 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.021     | 0.575    | 0.391     | 0.123  | 0.91  | 0.338 | 0.051 | 0.009 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.482     | 0.742    | 0.67      | 0.769  | 0.83  | 0.754 | 0.166 | 0.022 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.977 | 0.02  | 0.001 | 0.004 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.217     | 0.743    | 0.597     | 0.597  | 0.857 | 0.638 | 0.115 | 0.021 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.906 | 0.739 |
| 1-10  | UK      | 219 | 0.915 | 0.725 |
| 11-30 | AU      | 717 | 0.855 | 0.757 |
| 11-30 | UK      | 429 | 0.869 | 0.759 |
| 31-80 | AU      | 602 | 0.872 | 0.75  |
| 31-80 | UK      | 385 | 0.923 | 0.705 |
| >80   | AU      | 320 | 0.857 | 0.686 |
| >80   | UK      | 110 | 0.877 | 0.569 |
