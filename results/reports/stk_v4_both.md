# Validation report: `stk_v4_both`

generated 2026-09-18 04:18, code 3cf922e-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.019 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.56 (valid 0.849, pool 0.869; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.935   | 0.978   | 0.8     | 0.802   | 0.868 | 0.008   | 0.845       |
| global+ukg    | 0.936   | 0.978   | 0.805   | 0.825   | 0.871 | 0.009   | 0.852       |
| variety+ukg   | 0.937   | 0.976   | 0.802   | 0.826   | 0.87  | 0.009   | 0.851       |
| segment+ukg   | 0.936   | 0.975   | 0.802   | 0.826   | 0.869 | 0.005   | 0.849       |
| prev+ukg      | 0.937   | 0.978   | 0.801   | 0.827   | 0.869 | 0.01    | 0.855       |
| prevtuned+ukg | 0.937   | 0.978   | 0.801   | 0.827   | 0.869 | 0.008   | 0.858       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8691**

## Expected test score for `segment+ukg`
**Point estimate 0.8691 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8440, 0.8941]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.936 | 0.936 | 0.012 | 0.916 | 0.957 | null        |
| sent-UK | 0.975 | 0.975 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.802 | 0.807 | 0.019 | 0.774 | 0.838 | null        |
| sarc-UK | 0.826 | 0.829 | 0.036 | 0.767 | 0.884 | null        |
| score   | 0.869 | 0.868 | 0.013 | 0.846 | 0.888 | 1.0         |

binding varieties: sent: AU binds 99.7%, sarc: UK binds 29.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.981   | 0.828   | 0.856   | 0.874 |
| 1    | 0.933   | 0.973   | 0.803   | 0.861   | 0.868 |
| 2    | 0.953   | 0.969   | 0.803   | 0.779   | 0.866 |
| 3    | 0.936   | 0.973   | 0.783   | 0.841   | 0.86  |
| 4    | 0.938   | 0.977   | 0.793   | 0.789   | 0.863 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.461     | 0.936    | 0.935     | 0.927  | 0.985 | 0.984 | 0.047 | 0.009 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.975    | 0.98      | 0.971  | 0.997 | 0.997 | 0.018 | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.974    | 0.987     | 0.985  | 0.999 | 1.0   | 0.012 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.315     | 0.899    | 0.868     | 0.856  | 0.966 | 0.938 | 0.066 | 0.025 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.975    | 0.988     | 0.987  | 0.996 | 0.998 | 0.014 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.102     | 0.905    | 0.88      | 0.786  | 0.987 | 0.939 | 0.023 | 0.009 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.32      | 0.802    | 0.701     | 0.755  | 0.911 | 0.797 | 0.112 | 0.03  |
| sarc-UK        | 3208 | 249   | 0.078      | 0.075     | 0.826    | 0.692     | 0.667  | 0.969 | 0.751 | 0.035 | 0.009 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.031     | 0.673    | 0.606     | 0.274  | 0.938 | 0.53  | 0.044 | 0.01  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.475     | 0.771    | 0.704     | 0.797  | 0.867 | 0.813 | 0.149 | 0.042 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.959 | 0.012 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.21      | 0.797    | 0.692     | 0.669  | 0.902 | 0.753 | 0.097 | 0.037 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.922 | 0.755 |
| 1-10  | UK      | 219 | 0.924 | 0.79  |
| 11-30 | AU      | 717 | 0.891 | 0.787 |
| 11-30 | UK      | 429 | 0.861 | 0.782 |
| 31-80 | AU      | 602 | 0.887 | 0.786 |
| 31-80 | UK      | 385 | 0.912 | 0.815 |
| >80   | AU      | 320 | 0.895 | 0.721 |
| >80   | UK      | 110 | 0.877 | 0.695 |
