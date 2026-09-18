# Validation report: `stk_all_lr`

generated 2026-09-18 00:35, code 4c5c156-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.024 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.00 (valid 0.848, pool 0.859; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.927   | 0.977   | 0.788   | 0.781   | 0.854 | 0.01    | 0.824       |
| global+ukg    | 0.927   | 0.977   | 0.798   | 0.793   | 0.86  | 0.018   | 0.843       |
| variety+ukg   | 0.927   | 0.977   | 0.796   | 0.799   | 0.862 | 0.012   | 0.848       |
| segment+ukg   | 0.924   | 0.974   | 0.795   | 0.799   | 0.859 | 0.011   | 0.848       |
| prev+ukg      | 0.928   | 0.978   | 0.79    | 0.792   | 0.859 | 0.013   | 0.845       |
| prevtuned+ukg | 0.928   | 0.978   | 0.791   | 0.803   | 0.859 | 0.012   | 0.849       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8595**

## Expected test score for `segment+ukg`
**Point estimate 0.8595 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8345, 0.8845]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.924 | 0.925 | 0.013 | 0.903 | 0.946 | null        |
| sent-UK | 0.974 | 0.974 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.795 | 0.798 | 0.019 | 0.765 | 0.83  | null        |
| sarc-UK | 0.799 | 0.802 | 0.031 | 0.75  | 0.851 | null        |
| score   | 0.859 | 0.855 | 0.013 | 0.834 | 0.875 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 45.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.977   | 0.829   | 0.804   | 0.861 |
| 1    | 0.928   | 0.972   | 0.797   | 0.816   | 0.863 |
| 2    | 0.926   | 0.964   | 0.797   | 0.782   | 0.854 |
| 3    | 0.923   | 0.975   | 0.752   | 0.828   | 0.838 |
| 4    | 0.925   | 0.981   | 0.797   | 0.765   | 0.845 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.459     | 0.924    | 0.924     | 0.913  | 0.983 | 0.982 | 0.051 | 0.01  |
| sent-UK        | 3208 | 1676  | 0.522      | 0.521     | 0.974    | 0.976     | 0.974  | 0.996 | 0.997 | 0.018 | 0.001 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.728     | 0.969    | 0.989     | 0.979  | 0.998 | 0.999 | 0.014 | 0.004 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.315     | 0.881    | 0.843     | 0.833  | 0.962 | 0.932 | 0.071 | 0.013 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.974    | 0.988     | 0.986  | 0.996 | 0.998 | 0.014 | 0.002 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.114     | 0.907    | 0.838     | 0.832  | 0.984 | 0.923 | 0.025 | 0.01  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.344     | 0.795    | 0.671     | 0.779  | 0.9   | 0.773 | 0.117 | 0.012 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.106     | 0.799    | 0.55      | 0.751  | 0.959 | 0.667 | 0.04  | 0.004 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.061     | 0.719    | 0.5       | 0.452  | 0.928 | 0.464 | 0.046 | 0.011 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.497     | 0.76     | 0.682     | 0.807  | 0.852 | 0.79  | 0.155 | 0.015 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.457 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.297     | 0.755    | 0.55      | 0.754  | 0.871 | 0.67  | 0.11  | 0.015 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.914 | 0.751 |
| 1-10  | UK      | 219 | 0.916 | 0.746 |
| 11-30 | AU      | 717 | 0.871 | 0.771 |
| 11-30 | UK      | 429 | 0.866 | 0.744 |
| 31-80 | AU      | 602 | 0.871 | 0.769 |
| 31-80 | UK      | 385 | 0.934 | 0.748 |
| >80   | AU      | 320 | 0.857 | 0.719 |
| >80   | UK      | 110 | 0.865 | 0.695 |
