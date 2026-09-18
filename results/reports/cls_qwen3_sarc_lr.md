# Validation report: `cls_qwen3_sarc_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.016 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.15 (valid 0.825, pool 0.827; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.903   | 0.954   | 0.738   | 0.733   | 0.818 | 0.007   | 0.821       |
| global+ukg    | 0.899   | 0.955   | 0.731   | 0.753   | 0.815 | 0.011   | 0.82        |
| variety+ukg   | 0.899   | 0.955   | 0.733   | 0.756   | 0.816 | 0.011   | 0.829       |
| segment+ukg   | 0.898   | 0.953   | 0.761   | 0.756   | 0.827 | 0.005   | 0.825       |
| prev+ukg      | 0.902   | 0.952   | 0.752   | 0.742   | 0.822 | 0.01    | 0.814       |
| prevtuned+ukg | 0.902   | 0.952   | 0.76    | 0.753   | 0.828 | 0.007   | 0.832       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8267**

## Expected test score for `segment+ukg`
**Point estimate 0.8267 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7959, 0.8574]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.898 | 0.899 | 0.015 | 0.873 | 0.924 | null        |
| sent-UK | 0.953 | 0.953 | 0.01  | 0.935 | 0.969 | null        |
| sarc-AU | 0.761 | 0.767 | 0.021 | 0.732 | 0.8   | null        |
| sarc-UK | 0.756 | 0.757 | 0.036 | 0.698 | 0.817 | null        |
| score   | 0.827 | 0.822 | 0.016 | 0.795 | 0.846 | 0.997       |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 57.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.888   | 0.952   | 0.771   | 0.766   | 0.827 |
| 1    | 0.9     | 0.952   | 0.76    | 0.757   | 0.829 |
| 2    | 0.91    | 0.947   | 0.766   | 0.744   | 0.827 |
| 3    | 0.889   | 0.958   | 0.742   | 0.759   | 0.815 |
| 4    | 0.902   | 0.958   | 0.764   | 0.751   | 0.826 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.455     | 0.898    | 0.899     | 0.881  | 0.968 | 0.968 | 0.07  | 0.01  |
| sent-UK        | 3208 | 1676  | 0.522      | 0.523     | 0.953    | 0.955     | 0.956  | 0.99  | 0.992 | 0.034 | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.966    | 0.98      | 0.985  | 0.996 | 0.998 | 0.021 | 0.011 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.303     | 0.834    | 0.792     | 0.752  | 0.932 | 0.883 | 0.096 | 0.021 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.754     | 0.945    | 0.969     | 0.976  | 0.989 | 0.996 | 0.03  | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.107     | 0.856    | 0.77      | 0.718  | 0.961 | 0.82  | 0.042 | 0.016 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.333     | 0.761    | 0.635     | 0.713  | 0.848 | 0.647 | 0.153 | 0.087 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.098     | 0.756    | 0.497     | 0.627  | 0.933 | 0.523 | 0.064 | 0.07  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.02      | 0.565    | 0.364     | 0.11   | 0.913 | 0.403 | 0.104 | 0.147 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.501     | 0.72     | 0.641     | 0.766  | 0.807 | 0.733 | 0.18  | 0.055 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.275     | 0.705    | 0.497     | 0.629  | 0.791 | 0.524 | 0.178 | 0.193 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.873 | 0.699 |
| 1-10  | UK      | 219 | 0.855 | 0.652 |
| 11-30 | AU      | 717 | 0.806 | 0.724 |
| 11-30 | UK      | 429 | 0.803 | 0.718 |
| 31-80 | AU      | 602 | 0.833 | 0.736 |
| 31-80 | UK      | 385 | 0.895 | 0.702 |
| >80   | AU      | 320 | 0.821 | 0.701 |
| >80   | UK      | 110 | 0.824 | 0.535 |
