# Validation report: `cls_tfidf_nblr_seg`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.035 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.825   | 0.914   | 0.671   | 0.546   | 0.686 | 0.013   | 0.719       |
| global+ukg    | 0.825   | 0.911   | 0.655   | 0.646   | 0.735 | 0.009   | 0.732       |
| variety+ukg   | 0.825   | 0.915   | 0.697   | 0.659   | 0.742 | 0.01    | 0.754       |
| segment+ukg   | 0.841   | 0.917   | 0.696   | 0.659   | 0.75  | 0.008   | 0.762       |
| prev+ukg      | 0.84    | 0.916   | 0.681   | 0.632   | 0.736 | 0.012   | 0.738       |
| prevtuned+ukg | 0.837   | 0.912   | 0.683   | 0.66    | 0.749 | 0.008   | 0.752       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7498**

## Expected test score for `segment+ukg`
**Point estimate 0.7498 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7172, 0.7824]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.841 | 0.841 | 0.018 | 0.811 | 0.869 | null        |
| sent-UK | 0.917 | 0.917 | 0.012 | 0.896 | 0.935 | null        |
| sarc-AU | 0.696 | 0.703 | 0.021 | 0.671 | 0.737 | null        |
| sarc-UK | 0.659 | 0.66  | 0.031 | 0.61  | 0.711 | null        |
| score   | 0.75  | 0.749 | 0.017 | 0.721 | 0.776 | 0.066       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 88.6%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.817   | 0.923   | 0.69    | 0.658   | 0.737 |
| 1    | 0.844   | 0.92    | 0.692   | 0.664   | 0.754 |
| 2    | 0.854   | 0.911   | 0.72    | 0.641   | 0.747 |
| 3    | 0.845   | 0.912   | 0.675   | 0.681   | 0.76  |
| 4    | 0.844   | 0.918   | 0.703   | 0.65    | 0.747 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.435     | 0.841    | 0.852     | 0.799  | 0.91  | 0.913 | 0.12  | 0.042 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.512     | 0.917    | 0.929     | 0.911  | 0.967 | 0.975 | 0.066 | 0.027 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.737     | 0.883    | 0.937     | 0.939  | 0.961 | 0.985 | 0.075 | 0.068 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.273     | 0.768    | 0.73      | 0.625  | 0.851 | 0.755 | 0.145 | 0.063 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.764     | 0.9      | 0.942     | 0.961  | 0.973 | 0.99  | 0.056 | 0.038 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.059     | 0.685    | 0.627     | 0.321  | 0.845 | 0.526 | 0.083 | 0.055 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.384     | 0.696    | 0.531     | 0.688  | 0.798 | 0.575 | 0.167 | 0.054 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.161     | 0.659    | 0.296     | 0.614  | 0.887 | 0.334 | 0.063 | 0.042 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.006     | 0.507    | 0.333     | 0.027  | 0.776 | 0.216 | 0.06  | 0.032 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.588     | 0.618    | 0.532     | 0.746  | 0.691 | 0.59  | 0.224 | 0.073 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.452     | 0.549    | 0.296     | 0.617  | 0.637 | 0.335 | 0.176 | 0.115 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.711 | 0.561 |
| 1-10  | UK      | 219 | 0.72  | 0.437 |
| 11-30 | AU      | 717 | 0.783 | 0.652 |
| 11-30 | UK      | 429 | 0.66  | 0.531 |
| 31-80 | AU      | 602 | 0.772 | 0.613 |
| 31-80 | UK      | 385 | 0.661 | 0.562 |
| >80   | AU      | 320 | 0.774 | 0.614 |
| >80   | UK      | 110 | 0.481 | 0.458 |
