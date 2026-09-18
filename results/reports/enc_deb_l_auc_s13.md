# Validation report: `enc_deb_l_auc_s13`

generated 2026-09-18 05:46, code 3cf922e-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.045 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.75 (valid 0.848, pool 0.857; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.927   | 0.97    | 0.787   | 0.79    | 0.857 | 0.015   | 0.815       |
| global+ukg    | 0.925   | 0.97    | 0.793   | 0.802   | 0.859 | 0.007   | 0.847       |
| variety+ukg   | 0.925   | 0.97    | 0.788   | 0.809   | 0.857 | 0.004   | 0.845       |
| segment+ukg   | 0.926   | 0.971   | 0.789   | 0.81    | 0.857 | 0.005   | 0.848       |
| prev+ukg      | 0.924   | 0.97    | 0.797   | 0.809   | 0.86  | 0.008   | 0.848       |
| prevtuned+ukg | 0.924   | 0.97    | 0.795   | 0.807   | 0.859 | 0.009   | 0.845       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8574**

## Expected test score for `segment+ukg`
**Point estimate 0.8574 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8310, 0.8839]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.926 | 0.928 | 0.013 | 0.906 | 0.949 | null        |
| sent-UK | 0.971 | 0.971 | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.789 | 0.792 | 0.021 | 0.757 | 0.825 | null        |
| sarc-UK | 0.81  | 0.812 | 0.036 | 0.75  | 0.87  | null        |
| score   | 0.857 | 0.855 | 0.013 | 0.833 | 0.877 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 30.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.912   | 0.971   | 0.823   | 0.809   | 0.861 |
| 1    | 0.927   | 0.973   | 0.78    | 0.855   | 0.853 |
| 2    | 0.936   | 0.969   | 0.787   | 0.775   | 0.856 |
| 3    | 0.936   | 0.973   | 0.772   | 0.838   | 0.854 |
| 4    | 0.92    | 0.967   | 0.777   | 0.775   | 0.847 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.472     | 0.926    | 0.915     | 0.929  | 0.977 | 0.977 | 0.065 | 0.049 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.515     | 0.971    | 0.979     | 0.965  | 0.993 | 0.993 | 0.027 | 0.011 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.978    | 0.984     | 0.994  | 0.997 | 0.999 | 0.015 | 0.008 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.326     | 0.881    | 0.83      | 0.848  | 0.951 | 0.91  | 0.092 | 0.075 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.746     | 0.971    | 0.986     | 0.984  | 0.993 | 0.996 | 0.021 | 0.007 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.097     | 0.889    | 0.874     | 0.74   | 0.97  | 0.894 | 0.037 | 0.026 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.321     | 0.789    | 0.681     | 0.736  | 0.892 | 0.763 | 0.155 | 0.124 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.08      | 0.81     | 0.641     | 0.659  | 0.954 | 0.688 | 0.046 | 0.025 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.053     | 0.713    | 0.526     | 0.411  | 0.912 | 0.515 | 0.057 | 0.038 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.465     | 0.754    | 0.691     | 0.765  | 0.84  | 0.778 | 0.207 | 0.182 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.912 | 0.005 | 0.002 | 0.002 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.224     | 0.776    | 0.641     | 0.661  | 0.868 | 0.699 | 0.125 | 0.092 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.895 | 0.711 |
| 1-10  | UK      | 219 | 0.901 | 0.73  |
| 11-30 | AU      | 717 | 0.868 | 0.777 |
| 11-30 | UK      | 429 | 0.868 | 0.763 |
| 31-80 | AU      | 602 | 0.872 | 0.781 |
| 31-80 | UK      | 385 | 0.899 | 0.808 |
| >80   | AU      | 320 | 0.896 | 0.697 |
| >80   | UK      | 110 | 0.796 | 0.75  |
