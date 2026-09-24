# Validation report: `cls_e5me5ctx_lr`

generated 2026-09-18 12:46, code ebc1a23-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.012 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.67 (valid 0.815, pool 0.839; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.915   | 0.968   | 0.748   | 0.766   | 0.832 | 0.005   | 0.816       |
| global+ukg    | 0.913   | 0.97    | 0.744   | 0.759   | 0.829 | 0.004   | 0.82        |
| variety+ukg   | 0.911   | 0.97    | 0.746   | 0.768   | 0.828 | 0.005   | 0.815       |
| segment+ukg   | 0.911   | 0.972   | 0.771   | 0.768   | 0.839 | 0.006   | 0.815       |
| prev+ukg      | 0.918   | 0.97    | 0.751   | 0.766   | 0.835 | 0.01    | 0.825       |
| prevtuned+ukg | 0.918   | 0.97    | 0.759   | 0.77    | 0.839 | 0.01    | 0.823       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8394**

## Expected test score for `segment+ukg`
**Point estimate 0.8394 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8083, 0.8705]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.911 | 0.912 | 0.014 | 0.889 | 0.935 | null        |
| sent-UK | 0.972 | 0.972 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.771 | 0.778 | 0.019 | 0.746 | 0.809 | null        |
| sarc-UK | 0.768 | 0.769 | 0.037 | 0.706 | 0.828 | null        |
| score   | 0.839 | 0.834 | 0.016 | 0.807 | 0.858 | 0.998       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 57.7%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.905   | 0.974   | 0.795   | 0.772   | 0.838 |
| 1    | 0.908   | 0.969   | 0.784   | 0.783   | 0.845 |
| 2    | 0.916   | 0.969   | 0.766   | 0.752   | 0.834 |
| 3    | 0.915   | 0.976   | 0.744   | 0.761   | 0.83  |
| 4    | 0.912   | 0.97    | 0.767   | 0.769   | 0.839 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.466     | 0.911    | 0.904     | 0.906  | 0.977 | 0.976 | 0.059 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.526     | 0.972    | 0.97      | 0.976  | 0.994 | 0.995 | 0.024 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.966    | 0.977     | 0.987  | 0.996 | 0.999 | 0.018 | 0.014 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.317     | 0.86     | 0.811     | 0.806  | 0.952 | 0.911 | 0.081 | 0.019 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.963    | 0.979     | 0.984  | 0.993 | 0.998 | 0.022 | 0.009 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.118     | 0.928    | 0.859     | 0.885  | 0.978 | 0.898 | 0.028 | 0.011 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.344     | 0.771    | 0.642     | 0.744  | 0.861 | 0.691 | 0.144 | 0.061 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.088     | 0.768    | 0.541     | 0.61   | 0.94  | 0.534 | 0.053 | 0.045 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.02      | 0.598    | 0.5       | 0.151  | 0.897 | 0.357 | 0.085 | 0.102 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.517     | 0.729    | 0.645     | 0.795  | 0.814 | 0.738 | 0.176 | 0.044 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.246     | 0.723    | 0.541     | 0.613  | 0.812 | 0.536 | 0.149 | 0.126 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.873 | 0.705 |
| 1-10  | UK      | 219 | 0.938 | 0.675 |
| 11-30 | AU      | 717 | 0.846 | 0.743 |
| 11-30 | UK      | 429 | 0.872 | 0.721 |
| 31-80 | AU      | 602 | 0.866 | 0.742 |
| 31-80 | UK      | 385 | 0.956 | 0.723 |
| >80   | AU      | 320 | 0.842 | 0.695 |
| >80   | UK      | 110 | 0.933 | 0.647 |
