# Validation report: `enc_deb_l_ls05_s13`

generated 2026-09-18 02:39, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.046 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.23 (valid 0.831, pool 0.862; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.927   | 0.97    | 0.784   | 0.779   | 0.853 | 0.015   | 0.809       |
| global+ukg    | 0.927   | 0.969   | 0.794   | 0.812   | 0.861 | 0.008   | 0.833       |
| variety+ukg   | 0.926   | 0.97    | 0.793   | 0.809   | 0.86  | 0.008   | 0.831       |
| segment+ukg   | 0.927   | 0.968   | 0.797   | 0.809   | 0.862 | 0.006   | 0.831       |
| prev+ukg      | 0.926   | 0.971   | 0.794   | 0.814   | 0.86  | 0.006   | 0.834       |
| prevtuned+ukg | 0.926   | 0.971   | 0.798   | 0.814   | 0.862 | 0.007   | 0.835       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8615**

## Expected test score for `segment+ukg`
**Point estimate 0.8615 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8334, 0.8897]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.927 | 0.928 | 0.013 | 0.905 | 0.949 | null        |
| sent-UK | 0.968 | 0.968 | 0.008 | 0.953 | 0.982 | null        |
| sarc-AU | 0.797 | 0.8   | 0.02  | 0.765 | 0.832 | null        |
| sarc-UK | 0.809 | 0.811 | 0.038 | 0.744 | 0.873 | null        |
| score   | 0.862 | 0.858 | 0.014 | 0.833 | 0.88  | 1.0         |

binding varieties: sent: AU binds 99.7%, sarc: UK binds 37.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.907   | 0.967   | 0.834   | 0.836   | 0.871 |
| 1    | 0.928   | 0.973   | 0.788   | 0.831   | 0.858 |
| 2    | 0.943   | 0.966   | 0.785   | 0.781   | 0.862 |
| 3    | 0.928   | 0.969   | 0.792   | 0.813   | 0.86  |
| 4    | 0.926   | 0.964   | 0.782   | 0.78    | 0.853 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.464     | 0.927    | 0.922     | 0.92   | 0.978 | 0.977 | 0.064 | 0.046 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.513     | 0.968    | 0.978     | 0.961  | 0.993 | 0.994 | 0.028 | 0.01  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.745     | 0.982    | 0.984     | 0.997  | 0.997 | 0.999 | 0.015 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.312     | 0.879    | 0.843     | 0.825  | 0.951 | 0.912 | 0.091 | 0.079 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.744     | 0.968    | 0.986     | 0.981  | 0.993 | 0.997 | 0.022 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.096     | 0.877    | 0.855     | 0.718  | 0.963 | 0.881 | 0.038 | 0.022 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.317     | 0.797    | 0.695     | 0.742  | 0.887 | 0.758 | 0.137 | 0.096 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.07      | 0.809    | 0.683     | 0.614  | 0.919 | 0.661 | 0.043 | 0.017 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.041     | 0.679    | 0.523     | 0.315  | 0.893 | 0.537 | 0.049 | 0.031 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.465     | 0.766    | 0.704     | 0.779  | 0.839 | 0.774 | 0.184 | 0.135 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.875 | 0.004 | 0.002 | 0.021 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.196     | 0.778    | 0.683     | 0.617  | 0.864 | 0.689 | 0.118 | 0.079 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.9   | 0.758 |
| 1-10  | UK      | 219 | 0.88  | 0.724 |
| 11-30 | AU      | 717 | 0.858 | 0.78  |
| 11-30 | UK      | 429 | 0.832 | 0.779 |
| 31-80 | AU      | 602 | 0.878 | 0.784 |
| 31-80 | UK      | 385 | 0.906 | 0.793 |
| >80   | AU      | 320 | 0.882 | 0.708 |
| >80   | UK      | 110 | 0.847 | 0.75  |
