# Validation report: `stk_v4`

generated 2026-09-18 04:18, code 3cf922e-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.026 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.87 (valid 0.848, pool 0.870; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.933   | 0.977   | 0.802   | 0.813   | 0.868 | 0.01    | 0.834       |
| global+ukg    | 0.936   | 0.975   | 0.803   | 0.829   | 0.869 | 0.009   | 0.847       |
| variety+ukg   | 0.936   | 0.976   | 0.804   | 0.83    | 0.87  | 0.011   | 0.851       |
| segment+ukg   | 0.936   | 0.975   | 0.805   | 0.83    | 0.87  | 0.008   | 0.848       |
| prev+ukg      | 0.934   | 0.977   | 0.804   | 0.835   | 0.869 | 0.009   | 0.866       |
| prevtuned+ukg | 0.934   | 0.977   | 0.804   | 0.83    | 0.869 | 0.009   | 0.855       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8705**

## Expected test score for `segment+ukg`
**Point estimate 0.8705 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8451, 0.8958]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.936 | 0.937 | 0.012 | 0.916 | 0.957 | null        |
| sent-UK | 0.975 | 0.975 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.805 | 0.809 | 0.02  | 0.775 | 0.842 | null        |
| sarc-UK | 0.83  | 0.832 | 0.035 | 0.772 | 0.888 | null        |
| score   | 0.87  | 0.869 | 0.013 | 0.848 | 0.889 | 1.0         |

binding varieties: sent: AU binds 99.6%, sarc: UK binds 28.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.922   | 0.981   | 0.84    | 0.848   | 0.881 |
| 1    | 0.932   | 0.975   | 0.807   | 0.867   | 0.869 |
| 2    | 0.954   | 0.975   | 0.803   | 0.78    | 0.867 |
| 3    | 0.935   | 0.972   | 0.781   | 0.848   | 0.858 |
| 4    | 0.938   | 0.972   | 0.792   | 0.809   | 0.865 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.474     | 0.936    | 0.923     | 0.941  | 0.984 | 0.984 | 0.048 | 0.012 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.975    | 0.979     | 0.973  | 0.996 | 0.997 | 0.018 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.73      | 0.975    | 0.99      | 0.984  | 0.999 | 0.999 | 0.012 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.336     | 0.901    | 0.845     | 0.889  | 0.966 | 0.937 | 0.068 | 0.033 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.749     | 0.973    | 0.986     | 0.987  | 0.996 | 0.998 | 0.015 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.104     | 0.915    | 0.891     | 0.809  | 0.986 | 0.937 | 0.024 | 0.009 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.295     | 0.805    | 0.727     | 0.723  | 0.909 | 0.791 | 0.114 | 0.041 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.078     | 0.83     | 0.687     | 0.687  | 0.969 | 0.749 | 0.035 | 0.01  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.028     | 0.669    | 0.633     | 0.26   | 0.937 | 0.535 | 0.044 | 0.008 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.439     | 0.778    | 0.73      | 0.763  | 0.866 | 0.806 | 0.152 | 0.058 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.927 | 0.007 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.218     | 0.801    | 0.687     | 0.69   | 0.902 | 0.751 | 0.096 | 0.029 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.92  | 0.771 |
| 1-10  | UK      | 219 | 0.93  | 0.76  |
| 11-30 | AU      | 717 | 0.893 | 0.799 |
| 11-30 | UK      | 429 | 0.896 | 0.794 |
| 31-80 | AU      | 602 | 0.895 | 0.786 |
| 31-80 | UK      | 385 | 0.915 | 0.826 |
| >80   | AU      | 320 | 0.888 | 0.718 |
| >80   | UK      | 110 | 0.847 | 0.695 |
