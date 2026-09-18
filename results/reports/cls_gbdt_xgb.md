# Validation report: `cls_gbdt_xgb`

generated 2026-09-18 01:51, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [WARN] per-fold score spread 0.079 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.99 (valid 0.788, pool 0.816; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.913   | 0.963   | 0.749   | 0.643   | 0.778 | 0.031   | 0.722       |
| global+ukg    | 0.909   | 0.962   | 0.739   | 0.727   | 0.818 | 0.016   | 0.79        |
| variety+ukg   | 0.908   | 0.963   | 0.767   | 0.724   | 0.816 | 0.021   | 0.788       |
| segment+ukg   | 0.908   | 0.963   | 0.766   | 0.724   | 0.816 | 0.02    | 0.788       |
| prev+ukg      | 0.913   | 0.966   | 0.752   | 0.722   | 0.817 | 0.019   | 0.764       |
| prevtuned+ukg | 0.913   | 0.966   | 0.753   | 0.731   | 0.822 | 0.012   | 0.825       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8162**

## Expected test score for `segment+ukg`
**Point estimate 0.8162 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7835, 0.8490]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.908 | 0.909 | 0.014 | 0.884 | 0.932 | null        |
| sent-UK | 0.963 | 0.963 | 0.009 | 0.948 | 0.977 | null        |
| sarc-AU | 0.766 | 0.772 | 0.02  | 0.738 | 0.804 | null        |
| sarc-UK | 0.724 | 0.726 | 0.033 | 0.672 | 0.78  | null        |
| score   | 0.816 | 0.817 | 0.017 | 0.789 | 0.842 | 0.991       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 89.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.896   | 0.967   | 0.775   | 0.722   | 0.809 |
| 1    | 0.902   | 0.95    | 0.775   | 0.789   | 0.838 |
| 2    | 0.913   | 0.966   | 0.775   | 0.653   | 0.783 |
| 3    | 0.911   | 0.973   | 0.743   | 0.728   | 0.82  |
| 4    | 0.92    | 0.961   | 0.76    | 0.715   | 0.817 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.452     | 0.908    | 0.914     | 0.888  | 0.975 | 0.973 | 0.063 | 0.016 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.963    | 0.969     | 0.961  | 0.994 | 0.995 | 0.026 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.959    | 0.976     | 0.981  | 0.994 | 0.998 | 0.022 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.297     | 0.856    | 0.831     | 0.773  | 0.948 | 0.902 | 0.085 | 0.024 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.956    | 0.976     | 0.98   | 0.994 | 0.998 | 0.022 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.098     | 0.887    | 0.866     | 0.74   | 0.977 | 0.89  | 0.035 | 0.014 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.347     | 0.766    | 0.633     | 0.741  | 0.866 | 0.692 | 0.135 | 0.02  |
| sarc-UK        | 3208 | 249   | 0.078      | 0.122     | 0.724    | 0.412     | 0.647  | 0.937 | 0.499 | 0.05  | 0.014 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.031     | 0.613    | 0.424     | 0.192  | 0.897 | 0.358 | 0.052 | 0.022 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.517     | 0.724    | 0.64      | 0.788  | 0.798 | 0.708 | 0.18  | 0.018 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.828 | 0.003 | 0.001 | 0.007 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.342     | 0.656    | 0.412     | 0.649  | 0.799 | 0.501 | 0.138 | 0.027 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.867 | 0.713 |
| 1-10  | UK      | 219 | 0.901 | 0.59  |
| 11-30 | AU      | 717 | 0.84  | 0.739 |
| 11-30 | UK      | 429 | 0.844 | 0.662 |
| 31-80 | AU      | 602 | 0.854 | 0.734 |
| 31-80 | UK      | 385 | 0.906 | 0.65  |
| >80   | AU      | 320 | 0.863 | 0.676 |
| >80   | UK      | 110 | 0.847 | 0.516 |
