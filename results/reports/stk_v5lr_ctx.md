# Validation report: `stk_v5lr_ctx`

generated 2026-09-18 12:41, code ebc1a23-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.024 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.02 (valid 0.871, pool 0.871; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.93    | 0.977   | 0.804   | 0.82    | 0.867 | 0.01    | 0.862       |
| global+ukg    | 0.934   | 0.975   | 0.808   | 0.83    | 0.871 | 0.011   | 0.865       |
| variety+ukg   | 0.934   | 0.978   | 0.808   | 0.836   | 0.871 | 0.013   | 0.866       |
| segment+ukg   | 0.936   | 0.975   | 0.805   | 0.836   | 0.871 | 0.012   | 0.871       |
| prev+ukg      | 0.933   | 0.977   | 0.801   | 0.833   | 0.867 | 0.011   | 0.859       |
| prevtuned+ukg | 0.933   | 0.977   | 0.801   | 0.834   | 0.867 | 0.01    | 0.863       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8708**

## Expected test score for `segment+ukg`
**Point estimate 0.8708 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8460, 0.8957]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.936 | 0.937 | 0.012 | 0.916 | 0.957 | null        |
| sent-UK | 0.975 | 0.975 | 0.007 | 0.963 | 0.987 | null        |
| sarc-AU | 0.805 | 0.808 | 0.021 | 0.774 | 0.841 | null        |
| sarc-UK | 0.836 | 0.838 | 0.034 | 0.782 | 0.892 | null        |
| score   | 0.871 | 0.87  | 0.013 | 0.849 | 0.889 | 1.0         |

binding varieties: sent: AU binds 99.6%, sarc: UK binds 22.2%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.926   | 0.98    | 0.844   | 0.85    | 0.885 |
| 1    | 0.933   | 0.977   | 0.82    | 0.865   | 0.877 |
| 2    | 0.951   | 0.967   | 0.796   | 0.801   | 0.873 |
| 3    | 0.933   | 0.976   | 0.777   | 0.836   | 0.855 |
| 4    | 0.938   | 0.975   | 0.791   | 0.824   | 0.864 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.481     | 0.936    | 0.916     | 0.95   | 0.984 | 0.984 | 0.047 | 0.006 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.521     | 0.975    | 0.978     | 0.974  | 0.996 | 0.996 | 0.018 | 0.002 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.983    | 0.992     | 0.99   | 0.998 | 0.999 | 0.012 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.346     | 0.898    | 0.83      | 0.9    | 0.965 | 0.937 | 0.067 | 0.022 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.978    | 0.987     | 0.991  | 0.995 | 0.998 | 0.015 | 0.002 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.104     | 0.897    | 0.857     | 0.779  | 0.986 | 0.934 | 0.023 | 0.005 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.318     | 0.805    | 0.707     | 0.756  | 0.909 | 0.789 | 0.112 | 0.015 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.08      | 0.836    | 0.688     | 0.707  | 0.969 | 0.745 | 0.034 | 0.001 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.068     | 0.706    | 0.452     | 0.452  | 0.936 | 0.531 | 0.044 | 0.018 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.452     | 0.782    | 0.727     | 0.782  | 0.865 | 0.804 | 0.148 | 0.021 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.943 | 0.008 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.224     | 0.807    | 0.688     | 0.71   | 0.902 | 0.747 | 0.096 | 0.016 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.92  | 0.777 |
| 1-10  | UK      | 219 | 0.915 | 0.75  |
| 11-30 | AU      | 717 | 0.89  | 0.8   |
| 11-30 | UK      | 429 | 0.854 | 0.801 |
| 31-80 | AU      | 602 | 0.892 | 0.785 |
| 31-80 | UK      | 385 | 0.893 | 0.836 |
| >80   | AU      | 320 | 0.877 | 0.736 |
| >80   | UK      | 110 | 0.933 | 0.714 |
