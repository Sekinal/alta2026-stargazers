# Validation report: `enc_twr_l_s14`

generated 2026-09-18 01:47, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.038 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -3.35 (valid 0.803, pool 0.851; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.916   | 0.969   | 0.795   | 0.788   | 0.852 | 0.014   | 0.769       |
| global+ukg    | 0.918   | 0.968   | 0.788   | 0.781   | 0.849 | 0.012   | 0.785       |
| variety+ukg   | 0.918   | 0.97    | 0.791   | 0.782   | 0.85  | 0.007   | 0.815       |
| segment+ukg   | 0.917   | 0.967   | 0.789   | 0.785   | 0.851 | 0.008   | 0.803       |
| prev+ukg      | 0.916   | 0.972   | 0.8     | 0.787   | 0.852 | 0.008   | 0.784       |
| prevtuned+ukg | 0.916   | 0.972   | 0.797   | 0.786   | 0.851 | 0.006   | 0.807       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8505**

## Expected test score for `segment+ukg`
**Point estimate 0.8505 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8200, 0.8810]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.917 | 0.918 | 0.014 | 0.894 | 0.938 | null        |
| sent-UK | 0.967 | 0.967 | 0.008 | 0.953 | 0.982 | null        |
| sarc-AU | 0.789 | 0.794 | 0.021 | 0.761 | 0.826 | null        |
| sarc-UK | 0.785 | 0.786 | 0.037 | 0.724 | 0.847 | null        |
| score   | 0.851 | 0.845 | 0.015 | 0.817 | 0.869 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 57.2%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.911   | 0.969   | 0.836   | 0.794   | 0.853 |
| 1    | 0.919   | 0.967   | 0.789   | 0.795   | 0.854 |
| 2    | 0.928   | 0.967   | 0.776   | 0.788   | 0.852 |
| 3    | 0.906   | 0.967   | 0.772   | 0.785   | 0.839 |
| 4    | 0.918   | 0.964   | 0.772   | 0.757   | 0.838 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.469     | 0.917    | 0.907     | 0.915  | 0.974 | 0.975 | 0.073 | 0.059 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.508     | 0.967    | 0.982     | 0.955  | 0.993 | 0.993 | 0.028 | 0.009 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.749     | 0.975    | 0.978     | 0.996  | 0.996 | 0.998 | 0.018 | 0.012 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.319     | 0.865    | 0.817     | 0.816  | 0.945 | 0.907 | 0.103 | 0.081 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.74      | 0.966    | 0.988     | 0.977  | 0.992 | 0.996 | 0.023 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.088     | 0.875    | 0.891     | 0.687  | 0.974 | 0.898 | 0.036 | 0.023 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.314     | 0.789    | 0.687     | 0.728  | 0.9   | 0.772 | 0.137 | 0.111 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.085     | 0.785    | 0.577     | 0.635  | 0.955 | 0.633 | 0.048 | 0.029 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.031     | 0.633    | 0.485     | 0.219  | 0.915 | 0.482 | 0.053 | 0.03  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.467     | 0.758    | 0.694     | 0.772  | 0.855 | 0.79  | 0.183 | 0.15  |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.998 | 0.167 | 0.001 | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.24      | 0.744    | 0.577     | 0.637  | 0.86  | 0.64  | 0.131 | 0.11  |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.884 | 0.711 |
| 1-10  | UK      | 219 | 0.895 | 0.739 |
| 11-30 | AU      | 717 | 0.854 | 0.783 |
| 11-30 | UK      | 429 | 0.815 | 0.744 |
| 31-80 | AU      | 602 | 0.86  | 0.783 |
| 31-80 | UK      | 385 | 0.903 | 0.757 |
| >80   | AU      | 320 | 0.854 | 0.707 |
| >80   | UK      | 110 | 0.761 | 0.553 |
