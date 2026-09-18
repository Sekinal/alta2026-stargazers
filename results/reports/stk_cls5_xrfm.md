# Validation report: `stk_cls5_xrfm`

generated 2026-09-18 01:30, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.023 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.63 (valid 0.813, pool 0.850; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.926   | 0.975   | 0.788   | 0.763   | 0.845 | 0.009   | 0.785       |
| global+ukg    | 0.92    | 0.975   | 0.788   | 0.777   | 0.849 | 0.009   | 0.81        |
| variety+ukg   | 0.921   | 0.975   | 0.785   | 0.78    | 0.851 | 0.008   | 0.814       |
| segment+ukg   | 0.921   | 0.975   | 0.787   | 0.78    | 0.85  | 0.008   | 0.813       |
| prev+ukg      | 0.931   | 0.973   | 0.786   | 0.787   | 0.858 | 0.006   | 0.818       |
| prevtuned+ukg | 0.931   | 0.973   | 0.786   | 0.782   | 0.856 | 0.007   | 0.818       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8504**

## Expected test score for `segment+ukg`
**Point estimate 0.8504 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8219, 0.8789]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.921 | 0.922 | 0.013 | 0.9   | 0.943 | null        |
| sent-UK | 0.975 | 0.975 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.787 | 0.793 | 0.019 | 0.762 | 0.824 | null        |
| sarc-UK | 0.78  | 0.783 | 0.034 | 0.726 | 0.837 | null        |
| score   | 0.85  | 0.847 | 0.014 | 0.822 | 0.869 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 61.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.98    | 0.816   | 0.783   | 0.846 |
| 1    | 0.926   | 0.972   | 0.791   | 0.797   | 0.858 |
| 2    | 0.93    | 0.969   | 0.789   | 0.77    | 0.85  |
| 3    | 0.921   | 0.98    | 0.76    | 0.792   | 0.841 |
| 4    | 0.917   | 0.975   | 0.778   | 0.761   | 0.839 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.463     | 0.921    | 0.917     | 0.913  | 0.982 | 0.98  | 0.052 | 0.008 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.524     | 0.975    | 0.974     | 0.978  | 0.995 | 0.995 | 0.02  | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.969    | 0.985     | 0.982  | 0.998 | 0.999 | 0.014 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.318     | 0.875    | 0.832     | 0.828  | 0.96  | 0.923 | 0.073 | 0.008 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.753     | 0.974    | 0.984     | 0.99   | 0.994 | 0.998 | 0.017 | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.111     | 0.913    | 0.858     | 0.832  | 0.98  | 0.916 | 0.027 | 0.013 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.344     | 0.787    | 0.662     | 0.767  | 0.896 | 0.758 | 0.119 | 0.016 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.095     | 0.78     | 0.542     | 0.667  | 0.948 | 0.632 | 0.042 | 0.006 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.033     | 0.599    | 0.361     | 0.178  | 0.913 | 0.372 | 0.05  | 0.018 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.51      | 0.755    | 0.673     | 0.818  | 0.85  | 0.778 | 0.156 | 0.023 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.404 | 0.0   | 0.0   | 0.002 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.268     | 0.736    | 0.542     | 0.669  | 0.854 | 0.635 | 0.116 | 0.027 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.906 | 0.754 |
| 1-10  | UK      | 219 | 0.923 | 0.693 |
| 11-30 | AU      | 717 | 0.875 | 0.77  |
| 11-30 | UK      | 429 | 0.888 | 0.728 |
| 31-80 | AU      | 602 | 0.858 | 0.757 |
| 31-80 | UK      | 385 | 0.917 | 0.749 |
| >80   | AU      | 320 | 0.841 | 0.713 |
| >80   | UK      | 110 | 0.893 | 0.632 |
