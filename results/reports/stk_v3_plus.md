# Validation report: `stk_v3_plus`

generated 2026-09-18 02:00, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.024 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.71 (valid 0.848, pool 0.869; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.932   | 0.978   | 0.804   | 0.806   | 0.868 | 0.009   | 0.839       |
| global+ukg    | 0.934   | 0.976   | 0.801   | 0.817   | 0.868 | 0.009   | 0.848       |
| variety+ukg   | 0.934   | 0.976   | 0.8     | 0.828   | 0.867 | 0.006   | 0.848       |
| segment+ukg   | 0.934   | 0.978   | 0.803   | 0.828   | 0.869 | 0.005   | 0.848       |
| prev+ukg      | 0.935   | 0.975   | 0.799   | 0.829   | 0.867 | 0.008   | 0.856       |
| prevtuned+ukg | 0.935   | 0.975   | 0.797   | 0.832   | 0.866 | 0.01    | 0.855       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8688**

## Expected test score for `segment+ukg`
**Point estimate 0.8688 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8432, 0.8943]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.934 | 0.935 | 0.012 | 0.916 | 0.954 | null        |
| sent-UK | 0.978 | 0.978 | 0.007 | 0.966 | 0.99  | null        |
| sarc-AU | 0.803 | 0.807 | 0.021 | 0.773 | 0.842 | null        |
| sarc-UK | 0.828 | 0.832 | 0.035 | 0.771 | 0.891 | null        |
| score   | 0.869 | 0.868 | 0.013 | 0.846 | 0.888 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 26.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.92    | 0.981   | 0.832   | 0.833   | 0.876 |
| 1    | 0.932   | 0.977   | 0.798   | 0.859   | 0.865 |
| 2    | 0.943   | 0.98    | 0.796   | 0.801   | 0.87  |
| 3    | 0.941   | 0.975   | 0.788   | 0.846   | 0.864 |
| 4    | 0.935   | 0.976   | 0.801   | 0.801   | 0.868 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.463     | 0.934    | 0.931     | 0.928  | 0.984 | 0.984 | 0.048 | 0.005 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.978    | 0.983     | 0.975  | 0.996 | 0.997 | 0.018 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.731     | 0.974    | 0.989     | 0.984  | 0.998 | 0.999 | 0.013 | 0.013 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.319     | 0.897    | 0.859     | 0.859  | 0.965 | 0.936 | 0.067 | 0.018 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.975    | 0.986     | 0.989  | 0.996 | 0.999 | 0.015 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.099     | 0.927    | 0.938     | 0.809  | 0.987 | 0.928 | 0.024 | 0.01  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.28      | 0.803    | 0.742     | 0.7    | 0.906 | 0.783 | 0.114 | 0.028 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.075     | 0.828    | 0.696     | 0.671  | 0.963 | 0.724 | 0.036 | 0.008 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.035     | 0.682    | 0.579     | 0.301  | 0.93  | 0.515 | 0.044 | 0.01  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.412     | 0.779    | 0.749     | 0.735  | 0.861 | 0.799 | 0.152 | 0.038 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.478 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.21      | 0.799    | 0.696     | 0.673  | 0.885 | 0.727 | 0.1   | 0.025 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.912 | 0.765 |
| 1-10  | UK      | 219 | 0.948 | 0.805 |
| 11-30 | AU      | 717 | 0.895 | 0.801 |
| 11-30 | UK      | 429 | 0.913 | 0.792 |
| 31-80 | AU      | 602 | 0.885 | 0.79  |
| 31-80 | UK      | 385 | 0.912 | 0.804 |
| >80   | AU      | 320 | 0.886 | 0.718 |
| >80   | UK      | 110 | 0.877 | 0.632 |
