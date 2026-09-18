# Validation report: `enc_twr_l_x3`

generated 2026-09-18 02:04, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.019 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.94 (valid 0.831, pool 0.856; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.919   | 0.97    | 0.794   | 0.787   | 0.853 | 0.007   | 0.773       |
| global+ukg    | 0.914   | 0.969   | 0.798   | 0.806   | 0.856 | 0.01    | 0.833       |
| variety+ukg   | 0.914   | 0.972   | 0.798   | 0.808   | 0.856 | 0.01    | 0.83        |
| segment+ukg   | 0.914   | 0.972   | 0.799   | 0.808   | 0.856 | 0.009   | 0.831       |
| prev+ukg      | 0.919   | 0.97    | 0.794   | 0.809   | 0.856 | 0.013   | 0.828       |
| prevtuned+ukg | 0.919   | 0.97    | 0.797   | 0.803   | 0.858 | 0.007   | 0.839       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8561**

## Expected test score for `segment+ukg`
**Point estimate 0.8561 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8274, 0.8848]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.914 | 0.915 | 0.014 | 0.891 | 0.937 | null        |
| sent-UK | 0.972 | 0.972 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.799 | 0.804 | 0.019 | 0.771 | 0.834 | null        |
| sarc-UK | 0.808 | 0.809 | 0.036 | 0.747 | 0.867 | null        |
| score   | 0.856 | 0.852 | 0.015 | 0.827 | 0.875 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 44.3%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.904   | 0.984   | 0.84    | 0.83    | 0.867 |
| 1    | 0.922   | 0.97    | 0.792   | 0.833   | 0.857 |
| 2    | 0.916   | 0.969   | 0.788   | 0.788   | 0.852 |
| 3    | 0.908   | 0.97    | 0.779   | 0.809   | 0.843 |
| 4    | 0.918   | 0.967   | 0.793   | 0.773   | 0.846 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.453     | 0.914    | 0.918     | 0.895  | 0.978 | 0.978 | 0.067 | 0.051 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.515     | 0.972    | 0.98      | 0.967  | 0.994 | 0.995 | 0.025 | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.972    | 0.98      | 0.991  | 0.996 | 0.999 | 0.017 | 0.012 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.297     | 0.859    | 0.835     | 0.777  | 0.953 | 0.917 | 0.093 | 0.073 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.967    | 0.984     | 0.983  | 0.993 | 0.997 | 0.021 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.097     | 0.912    | 0.919     | 0.779  | 0.979 | 0.917 | 0.031 | 0.015 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.343     | 0.799    | 0.677     | 0.782  | 0.905 | 0.784 | 0.132 | 0.102 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.083     | 0.808    | 0.626     | 0.667  | 0.959 | 0.654 | 0.045 | 0.028 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.03      | 0.625    | 0.469     | 0.205  | 0.917 | 0.495 | 0.052 | 0.031 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.511     | 0.767    | 0.684     | 0.832  | 0.861 | 0.8   | 0.176 | 0.145 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.997 | 0.125 | 0.001 | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.232     | 0.773    | 0.626     | 0.669  | 0.872 | 0.66  | 0.124 | 0.092 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.894 | 0.739 |
| 1-10  | UK      | 219 | 0.922 | 0.757 |
| 11-30 | AU      | 717 | 0.836 | 0.777 |
| 11-30 | UK      | 429 | 0.878 | 0.773 |
| 31-80 | AU      | 602 | 0.854 | 0.794 |
| 31-80 | UK      | 385 | 0.923 | 0.783 |
| >80   | AU      | 320 | 0.851 | 0.721 |
| >80   | UK      | 110 | 0.877 | 0.626 |
