# Validation report: `stk_sel_lr2`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.019 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.34 (valid 0.837, pool 0.855; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.925   | 0.973   | 0.785   | 0.751   | 0.838 | 0.008   | 0.803       |
| global+ukg    | 0.925   | 0.974   | 0.791   | 0.787   | 0.856 | 0.015   | 0.835       |
| variety+ukg   | 0.925   | 0.973   | 0.793   | 0.787   | 0.856 | 0.014   | 0.839       |
| segment+ukg   | 0.923   | 0.974   | 0.789   | 0.787   | 0.855 | 0.014   | 0.837       |
| prev+ukg      | 0.933   | 0.973   | 0.787   | 0.787   | 0.86  | 0.014   | 0.829       |
| prevtuned+ukg | 0.933   | 0.973   | 0.791   | 0.792   | 0.862 | 0.018   | 0.84        |

**Headline policy: `segment+ukg`** (default policy); **score 0.8549**

## Expected test score for `segment+ukg`
**Point estimate 0.8549 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8265, 0.8833]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.923 | 0.924 | 0.013 | 0.902 | 0.946 | null        |
| sent-UK | 0.974 | 0.975 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.789 | 0.795 | 0.019 | 0.763 | 0.826 | null        |
| sarc-UK | 0.787 | 0.787 | 0.034 | 0.73  | 0.841 | null        |
| score   | 0.855 | 0.85  | 0.014 | 0.825 | 0.872 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 57.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.906   | 0.978   | 0.822   | 0.813   | 0.86  |
| 1    | 0.93    | 0.969   | 0.794   | 0.807   | 0.862 |
| 2    | 0.933   | 0.97    | 0.78    | 0.776   | 0.855 |
| 3    | 0.923   | 0.98    | 0.763   | 0.798   | 0.843 |
| 4    | 0.923   | 0.975   | 0.785   | 0.735   | 0.829 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.459     | 0.923    | 0.922     | 0.912  | 0.983 | 0.982 | 0.052 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.974    | 0.976     | 0.976  | 0.996 | 0.996 | 0.021 | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.969    | 0.985     | 0.982  | 0.998 | 0.999 | 0.014 | 0.007 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.312     | 0.879    | 0.843     | 0.825  | 0.962 | 0.928 | 0.072 | 0.011 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.755     | 0.971    | 0.981     | 0.99   | 0.995 | 0.998 | 0.017 | 0.001 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.102     | 0.914    | 0.897     | 0.802  | 0.982 | 0.908 | 0.028 | 0.006 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.332     | 0.789    | 0.673     | 0.753  | 0.894 | 0.759 | 0.119 | 0.015 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.098     | 0.787    | 0.548     | 0.691  | 0.953 | 0.649 | 0.041 | 0.003 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.024     | 0.571    | 0.346     | 0.123  | 0.909 | 0.389 | 0.051 | 0.02  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.497     | 0.759    | 0.682     | 0.807  | 0.849 | 0.781 | 0.156 | 0.021 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.17  | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.275     | 0.743    | 0.548     | 0.694  | 0.856 | 0.651 | 0.114 | 0.019 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.917 | 0.738 |
| 1-10  | UK      | 219 | 0.934 | 0.702 |
| 11-30 | AU      | 717 | 0.875 | 0.777 |
| 11-30 | UK      | 429 | 0.872 | 0.723 |
| 31-80 | AU      | 602 | 0.861 | 0.771 |
| 31-80 | UK      | 385 | 0.923 | 0.774 |
| >80   | AU      | 320 | 0.848 | 0.713 |
| >80   | UK      | 110 | 0.893 | 0.632 |
