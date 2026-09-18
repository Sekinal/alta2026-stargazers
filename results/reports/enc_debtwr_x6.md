# Validation report: `enc_debtwr_x6`

generated 2026-09-18 02:04, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.050 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.56 (valid 0.833, pool 0.866; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.928   | 0.971   | 0.797   | 0.797   | 0.862 | 0.017   | 0.815       |
| global+ukg    | 0.93    | 0.974   | 0.801   | 0.82    | 0.866 | 0.006   | 0.837       |
| variety+ukg   | 0.931   | 0.973   | 0.799   | 0.807   | 0.865 | 0.008   | 0.833       |
| segment+ukg   | 0.931   | 0.974   | 0.8     | 0.807   | 0.866 | 0.008   | 0.833       |
| prev+ukg      | 0.929   | 0.972   | 0.801   | 0.818   | 0.865 | 0.009   | 0.816       |
| prevtuned+ukg | 0.929   | 0.972   | 0.798   | 0.819   | 0.863 | 0.009   | 0.828       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8657**

## Expected test score for `segment+ukg`
**Point estimate 0.8657 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8387, 0.8926]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.931 | 0.932 | 0.012 | 0.91  | 0.951 | null        |
| sent-UK | 0.974 | 0.974 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.8   | 0.804 | 0.019 | 0.772 | 0.834 | null        |
| sarc-UK | 0.807 | 0.809 | 0.035 | 0.75  | 0.867 | null        |
| score   | 0.866 | 0.861 | 0.014 | 0.838 | 0.883 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 44.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.984   | 0.827   | 0.826   | 0.872 |
| 1    | 0.936   | 0.972   | 0.801   | 0.831   | 0.869 |
| 2    | 0.931   | 0.975   | 0.798   | 0.777   | 0.854 |
| 3    | 0.929   | 0.97    | 0.788   | 0.826   | 0.859 |
| 4    | 0.939   | 0.969   | 0.786   | 0.769   | 0.854 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.453     | 0.931    | 0.938     | 0.913  | 0.981 | 0.981 | 0.054 | 0.035 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.514     | 0.974    | 0.983     | 0.967  | 0.996 | 0.996 | 0.022 | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.981    | 0.987     | 0.992  | 0.998 | 0.999 | 0.013 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.299     | 0.886    | 0.871     | 0.816  | 0.96  | 0.927 | 0.076 | 0.043 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.742     | 0.971    | 0.989     | 0.981  | 0.995 | 0.998 | 0.018 | 0.007 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.916    | 0.905     | 0.802  | 0.984 | 0.924 | 0.029 | 0.013 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.354     | 0.8      | 0.672     | 0.801  | 0.91  | 0.791 | 0.125 | 0.082 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.085     | 0.807    | 0.617     | 0.679  | 0.962 | 0.697 | 0.04  | 0.017 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.046     | 0.677    | 0.48      | 0.329  | 0.926 | 0.533 | 0.049 | 0.033 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.519     | 0.767    | 0.681     | 0.842  | 0.866 | 0.806 | 0.166 | 0.114 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.992 | 0.056 | 0.001 | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.24      | 0.772    | 0.617     | 0.681  | 0.883 | 0.706 | 0.111 | 0.073 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.914 | 0.743 |
| 1-10  | UK      | 219 | 0.934 | 0.766 |
| 11-30 | AU      | 717 | 0.875 | 0.78  |
| 11-30 | UK      | 429 | 0.875 | 0.761 |
| 31-80 | AU      | 602 | 0.874 | 0.794 |
| 31-80 | UK      | 385 | 0.925 | 0.782 |
| >80   | AU      | 320 | 0.88  | 0.712 |
| >80   | UK      | 110 | 0.893 | 0.663 |
