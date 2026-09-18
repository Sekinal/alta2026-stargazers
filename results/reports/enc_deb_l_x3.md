# Validation report: `enc_deb_l_x3`

generated 2026-09-18 01:19, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.034 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -4.88 (valid 0.802, pool 0.866; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.933   | 0.973   | 0.795   | 0.794   | 0.864 | 0.011   | 0.816       |
| global+ukg    | 0.934   | 0.971   | 0.799   | 0.808   | 0.866 | 0.009   | 0.813       |
| variety+ukg   | 0.933   | 0.973   | 0.798   | 0.805   | 0.865 | 0.009   | 0.804       |
| segment+ukg   | 0.934   | 0.973   | 0.798   | 0.805   | 0.866 | 0.01    | 0.802       |
| prev+ukg      | 0.93    | 0.97    | 0.796   | 0.816   | 0.863 | 0.01    | 0.833       |
| prevtuned+ukg | 0.93    | 0.97    | 0.799   | 0.805   | 0.865 | 0.009   | 0.809       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8662**

## Expected test score for `segment+ukg`
**Point estimate 0.8662 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8384, 0.8939]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.934 | 0.935 | 0.012 | 0.914 | 0.954 | null        |
| sent-UK | 0.973 | 0.973 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.798 | 0.802 | 0.02  | 0.77  | 0.836 | null        |
| sarc-UK | 0.805 | 0.808 | 0.037 | 0.744 | 0.865 | null        |
| score   | 0.866 | 0.862 | 0.014 | 0.838 | 0.883 | 1.0         |

binding varieties: sent: AU binds 99.4%, sarc: UK binds 43.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.983   | 0.831   | 0.835   | 0.875 |
| 1    | 0.937   | 0.973   | 0.783   | 0.82    | 0.86  |
| 2    | 0.945   | 0.972   | 0.795   | 0.75    | 0.847 |
| 3    | 0.933   | 0.967   | 0.795   | 0.832   | 0.864 |
| 4    | 0.936   | 0.967   | 0.784   | 0.79    | 0.86  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.478     | 0.934    | 0.917     | 0.943  | 0.98  | 0.98  | 0.056 | 0.038 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.973    | 0.978     | 0.97   | 0.994 | 0.995 | 0.023 | 0.01  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.742     | 0.977    | 0.984     | 0.992  | 0.998 | 0.999 | 0.013 | 0.007 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.336     | 0.896    | 0.838     | 0.883  | 0.956 | 0.919 | 0.08  | 0.063 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.967    | 0.984     | 0.983  | 0.994 | 0.998 | 0.019 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.105     | 0.917    | 0.892     | 0.817  | 0.976 | 0.911 | 0.032 | 0.018 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.317     | 0.798    | 0.698     | 0.745  | 0.901 | 0.766 | 0.134 | 0.1   |
| sarc-UK        | 3208 | 249   | 0.078      | 0.082     | 0.805    | 0.626     | 0.659  | 0.957 | 0.701 | 0.042 | 0.02  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.034     | 0.665    | 0.541     | 0.274  | 0.924 | 0.538 | 0.05  | 0.036 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.469     | 0.769    | 0.704     | 0.786  | 0.853 | 0.78  | 0.179 | 0.134 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.795 | 0.002 | 0.002 | 0.003 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.229     | 0.77     | 0.626     | 0.661  | 0.873 | 0.714 | 0.116 | 0.087 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.901 | 0.759 |
| 1-10  | UK      | 219 | 0.936 | 0.711 |
| 11-30 | AU      | 717 | 0.895 | 0.785 |
| 11-30 | UK      | 429 | 0.868 | 0.754 |
| 31-80 | AU      | 602 | 0.884 | 0.783 |
| 31-80 | UK      | 385 | 0.932 | 0.811 |
| >80   | AU      | 320 | 0.897 | 0.713 |
| >80   | UK      | 110 | 0.893 | 0.75  |
