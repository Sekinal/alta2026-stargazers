# Validation report: `stk_debcls_sel`

generated 2026-09-18 01:02, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.026 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.65 (valid 0.843, pool 0.865; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.932   | 0.977   | 0.795   | 0.807   | 0.864 | 0.011   | 0.824       |
| global+ukg    | 0.93    | 0.976   | 0.801   | 0.811   | 0.866 | 0.01    | 0.819       |
| variety+ukg   | 0.93    | 0.976   | 0.8     | 0.81    | 0.865 | 0.01    | 0.846       |
| segment+ukg   | 0.93    | 0.973   | 0.799   | 0.81    | 0.865 | 0.009   | 0.843       |
| prev+ukg      | 0.931   | 0.975   | 0.799   | 0.818   | 0.865 | 0.007   | 0.85        |
| prevtuned+ukg | 0.931   | 0.975   | 0.8     | 0.805   | 0.866 | 0.006   | 0.837       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8647**

## Expected test score for `segment+ukg`
**Point estimate 0.8647 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8375, 0.8919]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.93  | 0.931 | 0.013 | 0.911 | 0.951 | null        |
| sent-UK | 0.973 | 0.974 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.799 | 0.804 | 0.02  | 0.771 | 0.835 | null        |
| sarc-UK | 0.81  | 0.812 | 0.036 | 0.75  | 0.87  | null        |
| score   | 0.865 | 0.861 | 0.014 | 0.837 | 0.882 | 1.0         |

binding varieties: sent: AU binds 99.8%, sarc: UK binds 41.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.975   | 0.836   | 0.822   | 0.87  |
| 1    | 0.93    | 0.972   | 0.787   | 0.843   | 0.858 |
| 2    | 0.935   | 0.975   | 0.797   | 0.764   | 0.849 |
| 3    | 0.928   | 0.973   | 0.786   | 0.852   | 0.857 |
| 4    | 0.938   | 0.972   | 0.791   | 0.758   | 0.848 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.475     | 0.93     | 0.915     | 0.936  | 0.984 | 0.983 | 0.049 | 0.009 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.515     | 0.973    | 0.981     | 0.968  | 0.996 | 0.996 | 0.02  | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.736     | 0.98     | 0.989     | 0.99   | 0.998 | 0.999 | 0.013 | 0.007 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.335     | 0.888    | 0.829     | 0.87   | 0.965 | 0.935 | 0.068 | 0.016 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.746     | 0.972    | 0.987     | 0.984  | 0.996 | 0.998 | 0.016 | 0.002 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.098     | 0.906    | 0.902     | 0.771  | 0.982 | 0.916 | 0.027 | 0.007 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.318     | 0.799    | 0.698     | 0.748  | 0.906 | 0.787 | 0.113 | 0.014 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.075     | 0.81     | 0.66      | 0.639  | 0.963 | 0.708 | 0.037 | 0.006 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.039     | 0.665    | 0.5       | 0.288  | 0.928 | 0.515 | 0.045 | 0.013 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.468     | 0.771    | 0.707     | 0.788  | 0.861 | 0.803 | 0.15  | 0.022 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.756 | 0.002 | 0.001 | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.211     | 0.778    | 0.66      | 0.641  | 0.884 | 0.711 | 0.103 | 0.017 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.898 | 0.762 |
| 1-10  | UK      | 219 | 0.907 | 0.755 |
| 11-30 | AU      | 717 | 0.891 | 0.788 |
| 11-30 | UK      | 429 | 0.885 | 0.757 |
| 31-80 | AU      | 602 | 0.876 | 0.784 |
| 31-80 | UK      | 385 | 0.906 | 0.807 |
| >80   | AU      | 320 | 0.868 | 0.714 |
| >80   | UK      | 110 | 0.933 | 0.714 |
