# Validation report: `enc_rob_l_s13`

generated 2026-09-18 02:19, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.028 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.61 (valid 0.796, pool 0.836; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.91    | 0.968   | 0.786   | 0.753   | 0.832 | 0.011   | 0.784       |
| global+ukg    | 0.91    | 0.969   | 0.784   | 0.76    | 0.835 | 0.01    | 0.8         |
| variety+ukg   | 0.91    | 0.966   | 0.788   | 0.763   | 0.836 | 0.011   | 0.8         |
| segment+ukg   | 0.906   | 0.967   | 0.783   | 0.765   | 0.836 | 0.012   | 0.796       |
| prev+ukg      | 0.913   | 0.967   | 0.792   | 0.77    | 0.841 | 0.012   | 0.814       |
| prevtuned+ukg | 0.913   | 0.967   | 0.79    | 0.78    | 0.846 | 0.01    | 0.809       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8355**

## Expected test score for `segment+ukg`
**Point estimate 0.8355 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8023, 0.8688]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.906 | 0.907 | 0.015 | 0.883 | 0.93  | null        |
| sent-UK | 0.967 | 0.967 | 0.009 | 0.953 | 0.979 | null        |
| sarc-AU | 0.783 | 0.789 | 0.022 | 0.752 | 0.823 | null        |
| sarc-UK | 0.765 | 0.766 | 0.037 | 0.702 | 0.825 | null        |
| score   | 0.836 | 0.833 | 0.017 | 0.803 | 0.858 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 70.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.891   | 0.977   | 0.798   | 0.798   | 0.844 |
| 1    | 0.914   | 0.966   | 0.79    | 0.776   | 0.845 |
| 2    | 0.918   | 0.969   | 0.787   | 0.716   | 0.817 |
| 3    | 0.903   | 0.959   | 0.778   | 0.767   | 0.835 |
| 4    | 0.905   | 0.962   | 0.76    | 0.755   | 0.83  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.451     | 0.906    | 0.912     | 0.885  | 0.974 | 0.973 | 0.076 | 0.056 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.515     | 0.967    | 0.975     | 0.961  | 0.993 | 0.994 | 0.028 | 0.014 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.742     | 0.96     | 0.975     | 0.984  | 0.996 | 0.999 | 0.022 | 0.008 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.295     | 0.851    | 0.826     | 0.764  | 0.944 | 0.899 | 0.105 | 0.09  |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.744     | 0.969    | 0.987     | 0.982  | 0.993 | 0.997 | 0.021 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.866    | 0.81      | 0.718  | 0.977 | 0.885 | 0.039 | 0.028 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.287     | 0.783    | 0.705     | 0.682  | 0.895 | 0.773 | 0.138 | 0.102 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.088     | 0.765    | 0.535     | 0.606  | 0.946 | 0.577 | 0.051 | 0.029 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.025     | 0.612    | 0.481     | 0.178  | 0.918 | 0.461 | 0.054 | 0.035 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.428     | 0.756    | 0.712     | 0.725  | 0.847 | 0.792 | 0.183 | 0.136 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.993 | 0.062 | 0.001 | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.247     | 0.72     | 0.535     | 0.609  | 0.837 | 0.584 | 0.141 | 0.109 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.842 | 0.712 |
| 1-10  | UK      | 219 | 0.887 | 0.689 |
| 11-30 | AU      | 717 | 0.843 | 0.772 |
| 11-30 | UK      | 429 | 0.767 | 0.716 |
| 31-80 | AU      | 602 | 0.857 | 0.78  |
| 31-80 | UK      | 385 | 0.904 | 0.738 |
| >80   | AU      | 320 | 0.853 | 0.717 |
| >80   | UK      | 110 | 0.893 | 0.586 |
