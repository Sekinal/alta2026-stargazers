# Validation report: `cls_e5me5_xrfm`

generated 2026-09-18 01:59, code 1ddcc30, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [WARN] per-fold score spread 0.060 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.57 (valid 0.838, pool 0.846; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.911   | 0.965   | 0.762   | 0.728   | 0.82  | 0.028   | 0.803       |
| global+ukg    | 0.919   | 0.961   | 0.777   | 0.776   | 0.848 | 0.01    | 0.837       |
| variety+ukg   | 0.919   | 0.962   | 0.773   | 0.771   | 0.845 | 0.01    | 0.835       |
| segment+ukg   | 0.921   | 0.969   | 0.772   | 0.771   | 0.846 | 0.009   | 0.838       |
| prev+ukg      | 0.921   | 0.968   | 0.766   | 0.783   | 0.843 | 0.015   | 0.836       |
| prevtuned+ukg | 0.921   | 0.968   | 0.763   | 0.789   | 0.842 | 0.011   | 0.846       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8460**

## Expected test score for `segment+ukg`
**Point estimate 0.8460 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8160, 0.8760]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.921 | 0.922 | 0.014 | 0.9   | 0.943 | null        |
| sent-UK | 0.969 | 0.969 | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.772 | 0.78  | 0.019 | 0.748 | 0.812 | null        |
| sarc-UK | 0.771 | 0.771 | 0.036 | 0.71  | 0.831 | null        |
| score   | 0.846 | 0.84  | 0.015 | 0.813 | 0.863 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 57.7%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.972   | 0.782   | 0.792   | 0.845 |
| 1    | 0.92    | 0.964   | 0.789   | 0.754   | 0.837 |
| 2    | 0.922   | 0.969   | 0.762   | 0.743   | 0.832 |
| 3    | 0.928   | 0.969   | 0.741   | 0.757   | 0.834 |
| 4    | 0.925   | 0.972   | 0.786   | 0.798   | 0.855 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.461     | 0.921    | 0.918     | 0.911  | 0.977 | 0.976 | 0.071 | 0.08  |
| sent-UK        | 3208 | 1676  | 0.522      | 0.523     | 0.969    | 0.97      | 0.971  | 0.993 | 0.993 | 0.034 | 0.049 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.734     | 0.97     | 0.985     | 0.984  | 0.996 | 0.999 | 0.027 | 0.055 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.314     | 0.874    | 0.835     | 0.822  | 0.952 | 0.911 | 0.094 | 0.099 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.964    | 0.979     | 0.984  | 0.992 | 0.996 | 0.032 | 0.046 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.108     | 0.905    | 0.855     | 0.809  | 0.974 | 0.88  | 0.038 | 0.044 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.341     | 0.772    | 0.645     | 0.742  | 0.875 | 0.723 | 0.13  | 0.024 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.09      | 0.771    | 0.542     | 0.627  | 0.946 | 0.596 | 0.045 | 0.014 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.009     | 0.53     | 0.4       | 0.055  | 0.85  | 0.295 | 0.055 | 0.011 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.519     | 0.733    | 0.647     | 0.801  | 0.824 | 0.749 | 0.171 | 0.042 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.252     | 0.727    | 0.542     | 0.629  | 0.834 | 0.598 | 0.125 | 0.053 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.893 | 0.71  |
| 1-10  | UK      | 219 | 0.944 | 0.697 |
| 11-30 | AU      | 717 | 0.871 | 0.745 |
| 11-30 | UK      | 429 | 0.859 | 0.73  |
| 31-80 | AU      | 602 | 0.866 | 0.747 |
| 31-80 | UK      | 385 | 0.893 | 0.713 |
| >80   | AU      | 320 | 0.848 | 0.694 |
| >80   | UK      | 110 | 0.877 | 0.647 |
