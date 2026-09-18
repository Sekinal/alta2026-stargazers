# Validation report: `stk_v4_plus`

generated 2026-09-18 04:18, code 3cf922e-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.027 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.25 (valid 0.870, pool 0.867; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.933   | 0.976   | 0.804   | 0.815   | 0.869 | 0.011   | 0.849       |
| global+ukg    | 0.933   | 0.975   | 0.801   | 0.827   | 0.867 | 0.009   | 0.85        |
| variety+ukg   | 0.933   | 0.974   | 0.803   | 0.834   | 0.868 | 0.009   | 0.87        |
| segment+ukg   | 0.933   | 0.976   | 0.801   | 0.834   | 0.867 | 0.01    | 0.87        |
| prev+ukg      | 0.935   | 0.977   | 0.805   | 0.831   | 0.87  | 0.007   | 0.865       |
| prevtuned+ukg | 0.935   | 0.977   | 0.799   | 0.833   | 0.867 | 0.009   | 0.864       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8667**

## Expected test score for `segment+ukg`
**Point estimate 0.8667 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8421, 0.8913]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.933 | 0.934 | 0.012 | 0.913 | 0.954 | null        |
| sent-UK | 0.976 | 0.976 | 0.008 | 0.964 | 0.987 | null        |
| sarc-AU | 0.801 | 0.806 | 0.02  | 0.773 | 0.837 | null        |
| sarc-UK | 0.834 | 0.837 | 0.035 | 0.779 | 0.895 | null        |
| score   | 0.867 | 0.867 | 0.012 | 0.846 | 0.887 | 1.0         |

binding varieties: sent: AU binds 99.7%, sarc: UK binds 21.6%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.922   | 0.98    | 0.835   | 0.833   | 0.877 |
| 1    | 0.93    | 0.975   | 0.803   | 0.885   | 0.866 |
| 2    | 0.946   | 0.974   | 0.797   | 0.79    | 0.868 |
| 3    | 0.93    | 0.973   | 0.77    | 0.845   | 0.85  |
| 4    | 0.935   | 0.976   | 0.8     | 0.816   | 0.867 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.474     | 0.933    | 0.919     | 0.938  | 0.985 | 0.984 | 0.047 | 0.004 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.976    | 0.98      | 0.974  | 0.996 | 0.997 | 0.018 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.736     | 0.975    | 0.986     | 0.987  | 0.998 | 0.999 | 0.012 | 0.013 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.333     | 0.894    | 0.84      | 0.877  | 0.966 | 0.937 | 0.066 | 0.014 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.75      | 0.975    | 0.986     | 0.988  | 0.996 | 0.999 | 0.015 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.912    | 0.89      | 0.802  | 0.986 | 0.933 | 0.024 | 0.005 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.311     | 0.801    | 0.706     | 0.741  | 0.908 | 0.785 | 0.113 | 0.028 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.078     | 0.834    | 0.692     | 0.695  | 0.965 | 0.733 | 0.035 | 0.008 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.039     | 0.656    | 0.476     | 0.274  | 0.932 | 0.525 | 0.044 | 0.013 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.458     | 0.775    | 0.716     | 0.781  | 0.863 | 0.8   | 0.151 | 0.039 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.837 | 0.003 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.219     | 0.805    | 0.692     | 0.698  | 0.89  | 0.736 | 0.098 | 0.03  |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.912 | 0.769 |
| 1-10  | UK      | 219 | 0.93  | 0.774 |
| 11-30 | AU      | 717 | 0.889 | 0.798 |
| 11-30 | UK      | 429 | 0.889 | 0.796 |
| 31-80 | AU      | 602 | 0.886 | 0.782 |
| 31-80 | UK      | 385 | 0.906 | 0.831 |
| >80   | AU      | 320 | 0.879 | 0.711 |
| >80   | UK      | 110 | 0.877 | 0.647 |
