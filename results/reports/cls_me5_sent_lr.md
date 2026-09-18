# Validation report: `cls_me5_sent_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.009 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.01 (valid 0.809, pool 0.827; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.91    | 0.962   | 0.74    | 0.748   | 0.825 | 0.004   | 0.805       |
| global+ukg    | 0.912   | 0.963   | 0.742   | 0.749   | 0.827 | 0.005   | 0.803       |
| variety+ukg   | 0.912   | 0.963   | 0.742   | 0.743   | 0.827 | 0.007   | 0.811       |
| segment+ukg   | 0.911   | 0.963   | 0.772   | 0.743   | 0.827 | 0.01    | 0.809       |
| prev+ukg      | 0.91    | 0.962   | 0.761   | 0.748   | 0.829 | 0.01    | 0.819       |
| prevtuned+ukg | 0.91    | 0.962   | 0.765   | 0.744   | 0.827 | 0.011   | 0.805       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8267**

## Expected test score for `segment+ukg`
**Point estimate 0.8267 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7925, 0.8610]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.911 | 0.911 | 0.015 | 0.886 | 0.935 | null        |
| sent-UK | 0.963 | 0.964 | 0.009 | 0.948 | 0.979 | null        |
| sarc-AU | 0.772 | 0.778 | 0.021 | 0.743 | 0.812 | null        |
| sarc-UK | 0.743 | 0.744 | 0.036 | 0.685 | 0.801 | null        |
| score   | 0.827 | 0.825 | 0.017 | 0.795 | 0.851 | 0.996       |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 79.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.901   | 0.963   | 0.783   | 0.771   | 0.836 |
| 1    | 0.913   | 0.955   | 0.785   | 0.75    | 0.832 |
| 2    | 0.915   | 0.963   | 0.775   | 0.716   | 0.816 |
| 3    | 0.911   | 0.97    | 0.74    | 0.754   | 0.826 |
| 4    | 0.912   | 0.967   | 0.775   | 0.712   | 0.812 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.467     | 0.911    | 0.903     | 0.906  | 0.972 | 0.972 | 0.065 | 0.012 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.963    | 0.969     | 0.961  | 0.993 | 0.994 | 0.027 | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.745     | 0.963    | 0.974     | 0.987  | 0.995 | 0.998 | 0.021 | 0.011 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.317     | 0.86     | 0.813     | 0.806  | 0.943 | 0.896 | 0.089 | 0.023 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.955    | 0.975     | 0.979  | 0.992 | 0.997 | 0.025 | 0.012 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.098     | 0.892    | 0.875     | 0.748  | 0.977 | 0.894 | 0.031 | 0.015 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.307     | 0.772    | 0.67      | 0.693  | 0.848 | 0.648 | 0.153 | 0.085 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.096     | 0.743    | 0.479     | 0.594  | 0.94  | 0.585 | 0.057 | 0.057 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.012     | 0.528    | 0.308     | 0.055  | 0.893 | 0.341 | 0.11  | 0.145 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.465     | 0.74     | 0.675     | 0.748  | 0.816 | 0.733 | 0.175 | 0.053 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.27      | 0.69     | 0.479     | 0.597  | 0.813 | 0.587 | 0.159 | 0.158 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.87  | 0.697 |
| 1-10  | UK      | 219 | 0.908 | 0.671 |
| 11-30 | AU      | 717 | 0.863 | 0.757 |
| 11-30 | UK      | 429 | 0.848 | 0.683 |
| 31-80 | AU      | 602 | 0.851 | 0.754 |
| 31-80 | UK      | 385 | 0.921 | 0.684 |
| >80   | AU      | 320 | 0.83  | 0.709 |
| >80   | UK      | 110 | 0.761 | 0.607 |
