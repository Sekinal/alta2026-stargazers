# Validation report: `cls_gbdt_xgb_nlog`

generated 2026-09-18 01:51, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.044 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -3.13 (valid 0.797, pool 0.842; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.917   | 0.972   | 0.774   | 0.77    | 0.844 | 0.016   | 0.802       |
| global+ukg    | 0.916   | 0.973   | 0.778   | 0.784   | 0.847 | 0.008   | 0.825       |
| variety+ukg   | 0.915   | 0.968   | 0.775   | 0.771   | 0.843 | 0.008   | 0.796       |
| segment+ukg   | 0.915   | 0.972   | 0.77    | 0.771   | 0.842 | 0.009   | 0.797       |
| prev+ukg      | 0.918   | 0.973   | 0.77    | 0.781   | 0.844 | 0.007   | 0.821       |
| prevtuned+ukg | 0.918   | 0.973   | 0.77    | 0.781   | 0.844 | 0.006   | 0.815       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8422**

## Expected test score for `segment+ukg`
**Point estimate 0.8422 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8131, 0.8714]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.915 | 0.916 | 0.013 | 0.892 | 0.937 | null        |
| sent-UK | 0.972 | 0.972 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.77  | 0.776 | 0.02  | 0.743 | 0.81  | null        |
| sarc-UK | 0.771 | 0.773 | 0.037 | 0.714 | 0.834 | null        |
| score   | 0.842 | 0.837 | 0.015 | 0.812 | 0.859 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 52.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.906   | 0.98    | 0.79    | 0.765   | 0.835 |
| 1    | 0.921   | 0.966   | 0.769   | 0.798   | 0.845 |
| 2    | 0.913   | 0.963   | 0.777   | 0.742   | 0.828 |
| 3    | 0.914   | 0.978   | 0.741   | 0.772   | 0.828 |
| 4    | 0.918   | 0.973   | 0.77    | 0.773   | 0.844 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.444     | 0.915    | 0.928     | 0.886  | 0.979 | 0.977 | 0.057 | 0.011 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.527     | 0.972    | 0.969     | 0.977  | 0.996 | 0.996 | 0.022 | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.742     | 0.972    | 0.981     | 0.99   | 0.996 | 0.999 | 0.018 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.283     | 0.859    | 0.854     | 0.758  | 0.955 | 0.915 | 0.079 | 0.023 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.762     | 0.968    | 0.975     | 0.994  | 0.995 | 0.998 | 0.02  | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.907    | 0.888     | 0.786  | 0.987 | 0.928 | 0.025 | 0.013 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.332     | 0.77     | 0.648     | 0.725  | 0.885 | 0.739 | 0.126 | 0.013 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.086     | 0.771    | 0.551     | 0.61   | 0.955 | 0.642 | 0.042 | 0.006 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.033     | 0.589    | 0.333     | 0.164  | 0.91  | 0.342 | 0.051 | 0.012 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.493     | 0.735    | 0.66      | 0.774  | 0.831 | 0.759 | 0.166 | 0.013 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.939 | 0.008 | 0.0   | 0.003 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.241     | 0.728    | 0.551     | 0.613  | 0.855 | 0.644 | 0.116 | 0.036 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.894 | 0.734 |
| 1-10  | UK      | 219 | 0.922 | 0.701 |
| 11-30 | AU      | 717 | 0.84  | 0.75  |
| 11-30 | UK      | 429 | 0.872 | 0.742 |
| 31-80 | AU      | 602 | 0.841 | 0.741 |
| 31-80 | UK      | 385 | 0.915 | 0.702 |
| >80   | AU      | 320 | 0.867 | 0.686 |
| >80   | UK      | 110 | 0.877 | 0.55  |
