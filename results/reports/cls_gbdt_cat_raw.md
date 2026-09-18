# Validation report: `cls_gbdt_cat_raw`

generated 2026-09-18 01:51, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.047 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.59 (valid 0.801, pool 0.810; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.913   | 0.967   | 0.732   | 0.567   | 0.74  | 0.019   | 0.718       |
| global+ukg    | 0.908   | 0.967   | 0.744   | 0.719   | 0.814 | 0.014   | 0.817       |
| variety+ukg   | 0.908   | 0.963   | 0.755   | 0.714   | 0.811 | 0.012   | 0.805       |
| segment+ukg   | 0.907   | 0.961   | 0.755   | 0.714   | 0.81  | 0.013   | 0.801       |
| prev+ukg      | 0.912   | 0.966   | 0.751   | 0.731   | 0.821 | 0.012   | 0.811       |
| prevtuned+ukg | 0.912   | 0.966   | 0.754   | 0.73    | 0.821 | 0.012   | 0.807       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8105**

## Expected test score for `segment+ukg`
**Point estimate 0.8105 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7748, 0.8461]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.907 | 0.908 | 0.014 | 0.886 | 0.932 | null        |
| sent-UK | 0.961 | 0.961 | 0.01  | 0.945 | 0.977 | null        |
| sarc-AU | 0.755 | 0.762 | 0.021 | 0.727 | 0.795 | null        |
| sarc-UK | 0.714 | 0.718 | 0.037 | 0.657 | 0.78  | null        |
| score   | 0.81  | 0.811 | 0.018 | 0.78  | 0.84  | 0.974       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 83.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.895   | 0.961   | 0.773   | 0.706   | 0.8   |
| 1    | 0.911   | 0.958   | 0.758   | 0.744   | 0.827 |
| 2    | 0.906   | 0.961   | 0.748   | 0.685   | 0.796 |
| 3    | 0.908   | 0.97    | 0.747   | 0.717   | 0.813 |
| 4    | 0.915   | 0.954   | 0.749   | 0.715   | 0.815 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.442     | 0.907    | 0.921     | 0.876  | 0.974 | 0.972 | 0.064 | 0.017 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.524     | 0.961    | 0.961     | 0.964  | 0.995 | 0.995 | 0.026 | 0.009 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.742     | 0.953    | 0.971     | 0.98   | 0.994 | 0.997 | 0.023 | 0.017 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.281     | 0.854    | 0.851     | 0.748  | 0.948 | 0.899 | 0.085 | 0.029 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.956    | 0.978     | 0.978  | 0.994 | 0.998 | 0.021 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.119     | 0.879    | 0.772     | 0.802  | 0.98  | 0.891 | 0.034 | 0.02  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.332     | 0.755    | 0.629     | 0.704  | 0.866 | 0.699 | 0.136 | 0.032 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.101     | 0.714    | 0.424     | 0.55   | 0.934 | 0.486 | 0.052 | 0.022 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.022     | 0.552    | 0.292     | 0.096  | 0.881 | 0.315 | 0.054 | 0.026 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.498     | 0.715    | 0.637     | 0.756  | 0.804 | 0.718 | 0.18  | 0.042 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.951 | 0.01  | 0.001 | 0.008 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.283     | 0.653    | 0.424     | 0.552  | 0.787 | 0.488 | 0.144 | 0.055 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.878 | 0.69  |
| 1-10  | UK      | 219 | 0.899 | 0.63  |
| 11-30 | AU      | 717 | 0.842 | 0.736 |
| 11-30 | UK      | 429 | 0.837 | 0.652 |
| 31-80 | AU      | 602 | 0.84  | 0.731 |
| 31-80 | UK      | 385 | 0.889 | 0.655 |
| >80   | AU      | 320 | 0.854 | 0.656 |
| >80   | UK      | 110 | 0.847 | 0.526 |
