# Validation report: `stk_v5r`

generated 2026-09-23 23:50, code 5f44a85-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] test split: 1573 rows bound to data/test.csv (f3ad7cc4f48da8e1)
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.028 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.56 (valid 0.851, pool 0.869; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.931   | 0.977   | 0.805   | 0.808   | 0.868 | 0.011   | 0.831       |
| global+ukg    | 0.936   | 0.976   | 0.806   | 0.821   | 0.871 | 0.011   | 0.842       |
| variety+ukg   | 0.935   | 0.976   | 0.806   | 0.841   | 0.87  | 0.008   | 0.852       |
| segment+ukg   | 0.934   | 0.976   | 0.805   | 0.841   | 0.869 | 0.009   | 0.851       |
| prev+ukg      | 0.934   | 0.978   | 0.806   | 0.838   | 0.87  | 0.006   | 0.861       |
| prevtuned+ukg | 0.934   | 0.978   | 0.806   | 0.835   | 0.87  | 0.006   | 0.867       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8693**

## Expected test score for `segment+ukg`
**Point estimate 0.8693 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8438, 0.8949]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.934 | 0.935 | 0.012 | 0.913 | 0.954 | null        |
| sent-UK | 0.976 | 0.976 | 0.007 | 0.964 | 0.987 | null        |
| sarc-AU | 0.805 | 0.809 | 0.021 | 0.774 | 0.843 | null        |
| sarc-UK | 0.841 | 0.843 | 0.035 | 0.781 | 0.899 | null        |
| score   | 0.869 | 0.869 | 0.013 | 0.848 | 0.89  | 1.0         |

binding varieties: sent: AU binds 99.8%, sarc: UK binds 19.3%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.922   | 0.984   | 0.843   | 0.866   | 0.883 |
| 1    | 0.928   | 0.973   | 0.805   | 0.878   | 0.867 |
| 2    | 0.948   | 0.978   | 0.801   | 0.792   | 0.87  |
| 3    | 0.935   | 0.97    | 0.783   | 0.846   | 0.859 |
| 4    | 0.936   | 0.972   | 0.791   | 0.809   | 0.864 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.475     | 0.934    | 0.919     | 0.941  | 0.984 | 0.984 | 0.049 | 0.012 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.976    | 0.981     | 0.973  | 0.996 | 0.997 | 0.018 | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.732     | 0.977    | 0.99      | 0.986  | 0.999 | 0.999 | 0.012 | 0.008 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.337     | 0.896    | 0.837     | 0.884  | 0.965 | 0.936 | 0.068 | 0.032 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.749     | 0.973    | 0.986     | 0.987  | 0.996 | 0.998 | 0.015 | 0.003 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.918    | 0.913     | 0.802  | 0.987 | 0.938 | 0.023 | 0.007 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.283     | 0.805    | 0.741     | 0.706  | 0.91  | 0.792 | 0.113 | 0.039 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.07      | 0.841    | 0.742     | 0.671  | 0.97  | 0.758 | 0.034 | 0.009 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.031     | 0.682    | 0.636     | 0.288  | 0.937 | 0.54  | 0.044 | 0.009 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.418     | 0.779    | 0.745     | 0.742  | 0.866 | 0.808 | 0.151 | 0.056 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.953 | 0.01  | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.197     | 0.815    | 0.742     | 0.673  | 0.906 | 0.761 | 0.094 | 0.027 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.907 | 0.771 |
| 1-10  | UK      | 219 | 0.935 | 0.761 |
| 11-30 | AU      | 717 | 0.894 | 0.797 |
| 11-30 | UK      | 429 | 0.896 | 0.81  |
| 31-80 | AU      | 602 | 0.884 | 0.791 |
| 31-80 | UK      | 385 | 0.921 | 0.845 |
| >80   | AU      | 320 | 0.888 | 0.72  |
| >80   | UK      | 110 | 0.847 | 0.714 |
