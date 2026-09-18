# Validation report: `cls_gbdt_lgbm_noclf`

generated 2026-09-17 23:55, code dc7268a-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.052 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.912   | 0.964   | 0.742   | 0.578   | 0.745 | 0.022   | 0.731       |
| global+ukg    | 0.907   | 0.964   | 0.725   | 0.72    | 0.813 | 0.01    | 0.797       |
| variety+ukg   | 0.907   | 0.964   | 0.756   | 0.721   | 0.814 | 0.013   | 0.807       |
| segment+ukg   | 0.907   | 0.961   | 0.755   | 0.721   | 0.814 | 0.014   | 0.806       |
| prev+ukg      | 0.909   | 0.965   | 0.744   | 0.694   | 0.801 | 0.016   | 0.779       |
| prevtuned+ukg | 0.909   | 0.965   | 0.75    | 0.728   | 0.819 | 0.011   | 0.816       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8136**

## Expected test score for `segment+ukg`
**Point estimate 0.8136 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7811, 0.8461]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.907 | 0.908 | 0.015 | 0.884 | 0.932 | null        |
| sent-UK | 0.961 | 0.96  | 0.009 | 0.945 | 0.977 | null        |
| sarc-AU | 0.755 | 0.761 | 0.02  | 0.728 | 0.795 | null        |
| sarc-UK | 0.721 | 0.722 | 0.033 | 0.669 | 0.774 | null        |
| score   | 0.814 | 0.813 | 0.016 | 0.785 | 0.84  | 0.986       |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 84.6%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.875   | 0.967   | 0.764   | 0.723   | 0.799 |
| 1    | 0.922   | 0.955   | 0.756   | 0.749   | 0.835 |
| 2    | 0.915   | 0.963   | 0.755   | 0.699   | 0.807 |
| 3    | 0.913   | 0.955   | 0.742   | 0.714   | 0.814 |
| 4    | 0.908   | 0.966   | 0.758   | 0.713   | 0.81  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.458     | 0.907    | 0.906     | 0.893  | 0.972 | 0.971 | 0.065 | 0.016 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.516     | 0.961    | 0.969     | 0.956  | 0.993 | 0.994 | 0.028 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.741     | 0.966    | 0.979     | 0.986  | 0.995 | 0.998 | 0.02  | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.306     | 0.85     | 0.811     | 0.778  | 0.941 | 0.896 | 0.089 | 0.025 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.743     | 0.957    | 0.982     | 0.975  | 0.993 | 0.997 | 0.025 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.106     | 0.871    | 0.802     | 0.74   | 0.968 | 0.868 | 0.034 | 0.008 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.361     | 0.755    | 0.612     | 0.744  | 0.861 | 0.686 | 0.137 | 0.014 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.127     | 0.721    | 0.401     | 0.659  | 0.927 | 0.439 | 0.052 | 0.012 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.036     | 0.623    | 0.41      | 0.219  | 0.908 | 0.393 | 0.051 | 0.028 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.535     | 0.707    | 0.619     | 0.79   | 0.79  | 0.703 | 0.184 | 0.018 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.862 | 0.004 | 0.001 | 0.007 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.358     | 0.649    | 0.401     | 0.661  | 0.761 | 0.44  | 0.146 | 0.031 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.87  | 0.692 |
| 1-10  | UK      | 219 | 0.912 | 0.592 |
| 11-30 | AU      | 717 | 0.835 | 0.721 |
| 11-30 | UK      | 429 | 0.835 | 0.62  |
| 31-80 | AU      | 602 | 0.852 | 0.719 |
| 31-80 | UK      | 385 | 0.855 | 0.674 |
| >80   | AU      | 320 | 0.831 | 0.665 |
| >80   | UK      | 110 | 0.761 | 0.466 |
