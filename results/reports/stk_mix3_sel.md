# Validation report: `stk_mix3_sel`

generated 2026-09-18 01:04, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.028 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -3.30 (valid 0.822, pool 0.868; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.935   | 0.976   | 0.799   | 0.811   | 0.867 | 0.011   | 0.82        |
| global+ukg    | 0.932   | 0.976   | 0.806   | 0.813   | 0.869 | 0.01    | 0.827       |
| variety+ukg   | 0.934   | 0.974   | 0.807   | 0.801   | 0.867 | 0.012   | 0.82        |
| segment+ukg   | 0.935   | 0.973   | 0.8     | 0.801   | 0.868 | 0.009   | 0.822       |
| prev+ukg      | 0.933   | 0.977   | 0.803   | 0.827   | 0.868 | 0.01    | 0.841       |
| prevtuned+ukg | 0.933   | 0.977   | 0.805   | 0.812   | 0.869 | 0.009   | 0.823       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8677**

## Expected test score for `segment+ukg`
**Point estimate 0.8677 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8390, 0.8965]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.935 | 0.936 | 0.012 | 0.916 | 0.957 | null        |
| sent-UK | 0.973 | 0.973 | 0.008 | 0.961 | 0.984 | null        |
| sarc-AU | 0.8   | 0.803 | 0.02  | 0.77  | 0.837 | null        |
| sarc-UK | 0.801 | 0.803 | 0.038 | 0.741 | 0.865 | null        |
| score   | 0.868 | 0.861 | 0.015 | 0.836 | 0.883 | 1.0         |

binding varieties: sent: AU binds 99.3%, sarc: UK binds 48.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.92    | 0.972   | 0.839   | 0.818   | 0.869 |
| 1    | 0.94    | 0.967   | 0.791   | 0.817   | 0.865 |
| 2    | 0.939   | 0.972   | 0.788   | 0.769   | 0.854 |
| 3    | 0.936   | 0.976   | 0.785   | 0.846   | 0.86  |
| 4    | 0.941   | 0.977   | 0.797   | 0.751   | 0.846 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.458     | 0.935    | 0.937     | 0.924  | 0.984 | 0.984 | 0.048 | 0.008 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.514     | 0.973    | 0.982     | 0.966  | 0.996 | 0.997 | 0.018 | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.732     | 0.982    | 0.992     | 0.989  | 0.998 | 0.999 | 0.013 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.311     | 0.894    | 0.867     | 0.844  | 0.966 | 0.936 | 0.067 | 0.015 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.745     | 0.972    | 0.988     | 0.984  | 0.996 | 0.998 | 0.014 | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.096     | 0.9      | 0.9       | 0.756  | 0.983 | 0.921 | 0.026 | 0.009 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.311     | 0.8      | 0.705     | 0.74   | 0.908 | 0.79  | 0.112 | 0.015 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.071     | 0.801    | 0.659     | 0.606  | 0.965 | 0.726 | 0.036 | 0.008 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.054     | 0.728    | 0.552     | 0.438  | 0.932 | 0.532 | 0.044 | 0.01  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.45      | 0.77     | 0.715     | 0.766  | 0.863 | 0.806 | 0.149 | 0.023 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.532 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.2       | 0.768    | 0.659     | 0.609  | 0.892 | 0.729 | 0.101 | 0.025 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.908 | 0.751 |
| 1-10  | UK      | 219 | 0.907 | 0.745 |
| 11-30 | AU      | 717 | 0.893 | 0.786 |
| 11-30 | UK      | 429 | 0.872 | 0.758 |
| 31-80 | AU      | 602 | 0.881 | 0.783 |
| 31-80 | UK      | 385 | 0.912 | 0.782 |
| >80   | AU      | 320 | 0.887 | 0.72  |
| >80   | UK      | 110 | 0.877 | 0.695 |
