# Validation report: `stk_mix2`

generated 2026-09-18 01:00, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.027 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -4.14 (valid 0.813, pool 0.867; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.932   | 0.976   | 0.801   | 0.804   | 0.867 | 0.012   | 0.807       |
| global+ukg    | 0.934   | 0.976   | 0.801   | 0.812   | 0.867 | 0.01    | 0.817       |
| variety+ukg   | 0.933   | 0.973   | 0.799   | 0.803   | 0.866 | 0.012   | 0.814       |
| segment+ukg   | 0.934   | 0.972   | 0.801   | 0.803   | 0.867 | 0.011   | 0.813       |
| prev+ukg      | 0.934   | 0.978   | 0.805   | 0.824   | 0.87  | 0.01    | 0.826       |
| prevtuned+ukg | 0.934   | 0.978   | 0.803   | 0.817   | 0.869 | 0.012   | 0.824       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8674**

## Expected test score for `segment+ukg`
**Point estimate 0.8674 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8398, 0.8950]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.934 | 0.935 | 0.012 | 0.913 | 0.954 | null        |
| sent-UK | 0.972 | 0.972 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.801 | 0.804 | 0.02  | 0.772 | 0.838 | null        |
| sarc-UK | 0.803 | 0.806 | 0.036 | 0.747 | 0.864 | null        |
| score   | 0.867 | 0.862 | 0.014 | 0.837 | 0.883 | 1.0         |

binding varieties: sent: AU binds 99.5%, sarc: UK binds 47.6%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.922   | 0.971   | 0.822   | 0.811   | 0.867 |
| 1    | 0.938   | 0.969   | 0.802   | 0.846   | 0.87  |
| 2    | 0.931   | 0.969   | 0.787   | 0.759   | 0.845 |
| 3    | 0.936   | 0.978   | 0.794   | 0.835   | 0.865 |
| 4    | 0.943   | 0.975   | 0.798   | 0.754   | 0.848 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.454     | 0.934    | 0.939     | 0.918  | 0.984 | 0.984 | 0.048 | 0.009 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.972    | 0.977     | 0.97   | 0.996 | 0.997 | 0.018 | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.734     | 0.982    | 0.991     | 0.99   | 0.998 | 0.999 | 0.013 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.304     | 0.891    | 0.872     | 0.83   | 0.966 | 0.937 | 0.067 | 0.014 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.745     | 0.978    | 0.991     | 0.986  | 0.996 | 0.999 | 0.014 | 0.003 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.111     | 0.882    | 0.803     | 0.779  | 0.979 | 0.919 | 0.026 | 0.012 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.325     | 0.801    | 0.694     | 0.76   | 0.906 | 0.79  | 0.113 | 0.011 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.082     | 0.803    | 0.622     | 0.655  | 0.964 | 0.729 | 0.036 | 0.007 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.055     | 0.742    | 0.576     | 0.466  | 0.931 | 0.536 | 0.043 | 0.015 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.471     | 0.767    | 0.701     | 0.786  | 0.861 | 0.805 | 0.15  | 0.019 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.446 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.229     | 0.768    | 0.622     | 0.657  | 0.891 | 0.732 | 0.101 | 0.018 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.895 | 0.763 |
| 1-10  | UK      | 219 | 0.876 | 0.738 |
| 11-30 | AU      | 717 | 0.89  | 0.775 |
| 11-30 | UK      | 429 | 0.839 | 0.758 |
| 31-80 | AU      | 602 | 0.884 | 0.788 |
| 31-80 | UK      | 385 | 0.908 | 0.782 |
| >80   | AU      | 320 | 0.886 | 0.704 |
| >80   | UK      | 110 | 0.933 | 0.714 |
