# Validation report: `enc_deb_l_ukc_s15`

generated 2026-09-18 04:22, code 3cf922e-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.027 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.71 (valid 0.826, pool 0.861; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.932   | 0.973   | 0.785   | 0.8     | 0.858 | 0.01    | 0.808       |
| global+ukg    | 0.932   | 0.971   | 0.794   | 0.802   | 0.863 | 0.008   | 0.808       |
| variety+ukg   | 0.932   | 0.971   | 0.793   | 0.802   | 0.863 | 0.009   | 0.826       |
| segment+ukg   | 0.932   | 0.971   | 0.79    | 0.802   | 0.861 | 0.008   | 0.826       |
| prev+ukg      | 0.931   | 0.971   | 0.79    | 0.811   | 0.861 | 0.006   | 0.83        |
| prevtuned+ukg | 0.931   | 0.971   | 0.794   | 0.822   | 0.862 | 0.009   | 0.831       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8608**

## Expected test score for `segment+ukg`
**Point estimate 0.8608 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8336, 0.8881]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.932 | 0.932 | 0.013 | 0.911 | 0.951 | null        |
| sent-UK | 0.971 | 0.971 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.79  | 0.794 | 0.02  | 0.76  | 0.826 | null        |
| sarc-UK | 0.802 | 0.804 | 0.036 | 0.742 | 0.86  | null        |
| score   | 0.861 | 0.857 | 0.014 | 0.833 | 0.878 | 1.0         |

binding varieties: sent: AU binds 99.5%, sarc: UK binds 40.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.922   | 0.978   | 0.816   | 0.801   | 0.862 |
| 1    | 0.93    | 0.977   | 0.788   | 0.84    | 0.859 |
| 2    | 0.936   | 0.97    | 0.792   | 0.788   | 0.862 |
| 3    | 0.931   | 0.961   | 0.772   | 0.826   | 0.852 |
| 4    | 0.94    | 0.967   | 0.778   | 0.745   | 0.842 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.469     | 0.932    | 0.923     | 0.932  | 0.978 | 0.977 | 0.06  | 0.043 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.52      | 0.971    | 0.974     | 0.97   | 0.994 | 0.994 | 0.025 | 0.011 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.741     | 0.976    | 0.984     | 0.991  | 0.997 | 0.999 | 0.015 | 0.005 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.324     | 0.892    | 0.847     | 0.859  | 0.953 | 0.914 | 0.085 | 0.069 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.97     | 0.985     | 0.984  | 0.992 | 0.997 | 0.019 | 0.009 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.108     | 0.896    | 0.839     | 0.794  | 0.978 | 0.911 | 0.035 | 0.02  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.323     | 0.79     | 0.681     | 0.742  | 0.895 | 0.756 | 0.143 | 0.116 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.083     | 0.802    | 0.614     | 0.659  | 0.957 | 0.699 | 0.047 | 0.023 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.04      | 0.663    | 0.488     | 0.288  | 0.923 | 0.491 | 0.052 | 0.039 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.476     | 0.758    | 0.689     | 0.781  | 0.845 | 0.773 | 0.191 | 0.154 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.767 | 0.002 | 0.002 | 0.002 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.234     | 0.766    | 0.614     | 0.661  | 0.871 | 0.71  | 0.128 | 0.103 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.89  | 0.763 |
| 1-10  | UK      | 219 | 0.915 | 0.734 |
| 11-30 | AU      | 717 | 0.888 | 0.773 |
| 11-30 | UK      | 429 | 0.866 | 0.754 |
| 31-80 | AU      | 602 | 0.889 | 0.768 |
| 31-80 | UK      | 385 | 0.902 | 0.788 |
| >80   | AU      | 320 | 0.895 | 0.695 |
| >80   | UK      | 110 | 0.847 | 0.678 |
