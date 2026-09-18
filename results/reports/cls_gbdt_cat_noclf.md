# Validation report: `cls_gbdt_cat_noclf`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.047 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.17 (valid 0.814, pool 0.817; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.91    | 0.964   | 0.733   | 0.54    | 0.725 | 0.019   | 0.697       |
| global+ukg    | 0.91    | 0.96    | 0.734   | 0.724   | 0.817 | 0.015   | 0.809       |
| variety+ukg   | 0.911   | 0.964   | 0.749   | 0.723   | 0.817 | 0.019   | 0.814       |
| segment+ukg   | 0.911   | 0.959   | 0.749   | 0.723   | 0.817 | 0.02    | 0.814       |
| prev+ukg      | 0.909   | 0.965   | 0.735   | 0.72    | 0.815 | 0.015   | 0.807       |
| prevtuned+ukg | 0.909   | 0.965   | 0.745   | 0.718   | 0.814 | 0.016   | 0.82        |

**Headline policy: `segment+ukg`** (default policy); **score 0.8167**

## Expected test score for `segment+ukg`
**Point estimate 0.8167 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7837, 0.8497]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.911 | 0.912 | 0.014 | 0.888 | 0.932 | null        |
| sent-UK | 0.959 | 0.959 | 0.009 | 0.943 | 0.974 | null        |
| sarc-AU | 0.749 | 0.756 | 0.02  | 0.72  | 0.789 | null        |
| sarc-UK | 0.723 | 0.723 | 0.035 | 0.664 | 0.777 | null        |
| score   | 0.817 | 0.815 | 0.017 | 0.786 | 0.841 | 0.987       |

binding varieties: sent: AU binds 99.6%, sarc: UK binds 78.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.888   | 0.96    | 0.752   | 0.694   | 0.791 |
| 1    | 0.922   | 0.953   | 0.763   | 0.767   | 0.842 |
| 2    | 0.926   | 0.961   | 0.744   | 0.714   | 0.82  |
| 3    | 0.907   | 0.961   | 0.738   | 0.742   | 0.822 |
| 4    | 0.91    | 0.959   | 0.746   | 0.692   | 0.801 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.463     | 0.911    | 0.905     | 0.903  | 0.972 | 0.971 | 0.064 | 0.012 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.521     | 0.959    | 0.962     | 0.959  | 0.993 | 0.994 | 0.027 | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.966    | 0.98      | 0.985  | 0.993 | 0.997 | 0.02  | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.315     | 0.858    | 0.812     | 0.802  | 0.942 | 0.897 | 0.088 | 0.023 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.753     | 0.954    | 0.974     | 0.98   | 0.993 | 0.997 | 0.023 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.863    | 0.797     | 0.718  | 0.972 | 0.87  | 0.036 | 0.011 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.344     | 0.749    | 0.613     | 0.711  | 0.858 | 0.679 | 0.139 | 0.021 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.113     | 0.723    | 0.421     | 0.61   | 0.925 | 0.434 | 0.053 | 0.014 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.017     | 0.547    | 0.333     | 0.082  | 0.905 | 0.355 | 0.054 | 0.038 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.52      | 0.702    | 0.618     | 0.766  | 0.787 | 0.696 | 0.185 | 0.016 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.946 | 0.009 | 0.001 | 0.006 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.316     | 0.659    | 0.421     | 0.613  | 0.757 | 0.435 | 0.148 | 0.038 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.867 | 0.686 |
| 1-10  | UK      | 219 | 0.905 | 0.629 |
| 11-30 | AU      | 717 | 0.852 | 0.721 |
| 11-30 | UK      | 429 | 0.824 | 0.656 |
| 31-80 | AU      | 602 | 0.847 | 0.713 |
| 31-80 | UK      | 385 | 0.845 | 0.632 |
| >80   | AU      | 320 | 0.862 | 0.649 |
| >80   | UK      | 110 | 0.761 | 0.576 |
