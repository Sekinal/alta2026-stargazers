# Validation report: `stk_e5_feat`

generated 2026-09-18 02:11, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.035 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.34 (valid 0.842, pool 0.859; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.922   | 0.972   | 0.784   | 0.767   | 0.845 | 0.014   | 0.823       |
| global+ukg    | 0.919   | 0.973   | 0.788   | 0.796   | 0.854 | 0.01    | 0.841       |
| variety+ukg   | 0.921   | 0.972   | 0.793   | 0.8     | 0.857 | 0.01    | 0.841       |
| segment+ukg   | 0.924   | 0.968   | 0.793   | 0.8     | 0.859 | 0.011   | 0.842       |
| prev+ukg      | 0.921   | 0.972   | 0.783   | 0.787   | 0.852 | 0.01    | 0.842       |
| prevtuned+ukg | 0.921   | 0.972   | 0.79    | 0.796   | 0.856 | 0.006   | 0.836       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8587**

## Expected test score for `segment+ukg`
**Point estimate 0.8587 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8324, 0.8850]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.924 | 0.925 | 0.013 | 0.903 | 0.946 | null        |
| sent-UK | 0.968 | 0.968 | 0.008 | 0.953 | 0.982 | null        |
| sarc-AU | 0.793 | 0.799 | 0.019 | 0.768 | 0.83  | null        |
| sarc-UK | 0.8   | 0.803 | 0.033 | 0.749 | 0.855 | null        |
| score   | 0.859 | 0.855 | 0.013 | 0.833 | 0.875 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 44.2%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.912   | 0.967   | 0.806   | 0.807   | 0.859 |
| 1    | 0.933   | 0.964   | 0.794   | 0.827   | 0.863 |
| 2    | 0.928   | 0.964   | 0.796   | 0.798   | 0.862 |
| 3    | 0.928   | 0.972   | 0.785   | 0.814   | 0.856 |
| 4    | 0.921   | 0.972   | 0.783   | 0.751   | 0.836 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.445     | 0.924    | 0.938     | 0.898  | 0.98  | 0.979 | 0.055 | 0.006 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.968    | 0.975     | 0.964  | 0.995 | 0.996 | 0.022 | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.742     | 0.975    | 0.982     | 0.991  | 0.997 | 0.999 | 0.016 | 0.004 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.285     | 0.876    | 0.876     | 0.783  | 0.957 | 0.92  | 0.076 | 0.013 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.745     | 0.967    | 0.985     | 0.981  | 0.995 | 0.998 | 0.019 | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.885    | 0.839     | 0.756  | 0.983 | 0.918 | 0.027 | 0.007 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.337     | 0.793    | 0.674     | 0.766  | 0.895 | 0.761 | 0.12  | 0.017 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.099     | 0.8      | 0.566     | 0.723  | 0.958 | 0.647 | 0.041 | 0.002 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.025     | 0.622    | 0.519     | 0.192  | 0.921 | 0.435 | 0.048 | 0.013 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.504     | 0.759    | 0.679     | 0.816  | 0.845 | 0.778 | 0.159 | 0.027 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.991 | 0.05  | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.278     | 0.758    | 0.566     | 0.726  | 0.863 | 0.648 | 0.114 | 0.024 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.894 | 0.755 |
| 1-10  | UK      | 219 | 0.896 | 0.718 |
| 11-30 | AU      | 717 | 0.859 | 0.771 |
| 11-30 | UK      | 429 | 0.831 | 0.745 |
| 31-80 | AU      | 602 | 0.884 | 0.771 |
| 31-80 | UK      | 385 | 0.899 | 0.775 |
| >80   | AU      | 320 | 0.857 | 0.705 |
| >80   | UK      | 110 | 0.924 | 0.647 |
