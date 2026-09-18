# Validation report: `cls_tfidf_nbsvm`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.028 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.852   | 0.913   | 0.706   | 0.63    | 0.741 | 0.014   | 0.788       |
| global+ukg    | 0.854   | 0.918   | 0.673   | 0.646   | 0.75  | 0.013   | 0.768       |
| variety+ukg   | 0.854   | 0.922   | 0.702   | 0.646   | 0.75  | 0.013   | 0.766       |
| segment+ukg   | 0.85    | 0.928   | 0.698   | 0.646   | 0.748 | 0.01    | 0.761       |
| prev+ukg      | 0.848   | 0.931   | 0.674   | 0.628   | 0.738 | 0.013   | 0.781       |
| prevtuned+ukg | 0.845   | 0.93    | 0.686   | 0.634   | 0.739 | 0.014   | 0.755       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7478**

## Expected test score for `segment+ukg`
**Point estimate 0.7478 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7161, 0.7795]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.85  | 0.85  | 0.017 | 0.822 | 0.878 | null        |
| sent-UK | 0.928 | 0.928 | 0.012 | 0.907 | 0.948 | null        |
| sarc-AU | 0.698 | 0.706 | 0.019 | 0.675 | 0.737 | null        |
| sarc-UK | 0.646 | 0.646 | 0.029 | 0.598 | 0.692 | null        |
| score   | 0.748 | 0.748 | 0.016 | 0.721 | 0.773 | 0.044       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 96.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.847   | 0.933   | 0.695   | 0.655   | 0.751 |
| 1    | 0.858   | 0.928   | 0.716   | 0.646   | 0.752 |
| 2    | 0.847   | 0.93    | 0.715   | 0.619   | 0.733 |
| 3    | 0.851   | 0.92    | 0.684   | 0.666   | 0.758 |
| 4    | 0.846   | 0.931   | 0.678   | 0.643   | 0.745 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.415     | 0.85     | 0.881     | 0.788  | 0.924 | 0.927 | 0.11  | 0.056 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.928    | 0.932     | 0.931  | 0.969 | 0.976 | 0.065 | 0.025 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.744     | 0.906    | 0.945     | 0.957  | 0.97  | 0.988 | 0.064 | 0.076 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.239     | 0.766    | 0.774     | 0.578  | 0.864 | 0.781 | 0.135 | 0.048 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.766     | 0.907    | 0.944     | 0.966  | 0.976 | 0.991 | 0.061 | 0.067 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.081     | 0.782    | 0.731     | 0.519  | 0.887 | 0.657 | 0.072 | 0.051 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.405     | 0.698    | 0.527     | 0.719  | 0.8   | 0.568 | 0.179 | 0.108 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.176     | 0.646    | 0.275     | 0.622  | 0.882 | 0.322 | 0.077 | 0.079 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.01      | 0.492    | 0.091     | 0.014  | 0.773 | 0.19  | 0.062 | 0.04  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.616     | 0.618    | 0.531     | 0.78   | 0.699 | 0.583 | 0.243 | 0.156 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.999 | 0.25  | 0.0   | 0.003 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.493     | 0.521    | 0.275     | 0.625  | 0.612 | 0.322 | 0.214 | 0.215 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.753 | 0.584 |
| 1-10  | UK      | 219 | 0.835 | 0.451 |
| 11-30 | AU      | 717 | 0.758 | 0.656 |
| 11-30 | UK      | 429 | 0.7   | 0.519 |
| 31-80 | AU      | 602 | 0.757 | 0.594 |
| 31-80 | UK      | 385 | 0.818 | 0.542 |
| >80   | AU      | 320 | 0.774 | 0.61  |
| >80   | UK      | 110 | 0.481 | 0.45  |
