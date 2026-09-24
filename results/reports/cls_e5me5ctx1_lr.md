# Validation report: `cls_e5me5ctx1_lr`

generated 2026-09-18 12:46, code ebc1a23-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.025 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.03 (valid 0.842, pool 0.841; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.92    | 0.969   | 0.749   | 0.753   | 0.835 | 0.01    | 0.828       |
| global+ukg    | 0.919   | 0.969   | 0.748   | 0.758   | 0.834 | 0.009   | 0.843       |
| variety+ukg   | 0.919   | 0.968   | 0.748   | 0.765   | 0.834 | 0.006   | 0.841       |
| segment+ukg   | 0.918   | 0.969   | 0.771   | 0.765   | 0.841 | 0.007   | 0.842       |
| prev+ukg      | 0.922   | 0.97    | 0.76    | 0.763   | 0.841 | 0.012   | 0.821       |
| prevtuned+ukg | 0.922   | 0.97    | 0.758   | 0.764   | 0.84  | 0.009   | 0.832       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8413**

## Expected test score for `segment+ukg`
**Point estimate 0.8413 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8122, 0.8704]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.918 | 0.919 | 0.014 | 0.895 | 0.94  | null        |
| sent-UK | 0.969 | 0.969 | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.771 | 0.777 | 0.02  | 0.743 | 0.811 | null        |
| sarc-UK | 0.765 | 0.765 | 0.033 | 0.711 | 0.821 | null        |
| score   | 0.841 | 0.837 | 0.015 | 0.811 | 0.86  | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 61.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.911   | 0.975   | 0.782   | 0.755   | 0.833 |
| 1    | 0.921   | 0.969   | 0.791   | 0.774   | 0.848 |
| 2    | 0.92    | 0.969   | 0.774   | 0.754   | 0.837 |
| 3    | 0.918   | 0.969   | 0.742   | 0.759   | 0.83  |
| 4    | 0.92    | 0.966   | 0.762   | 0.78    | 0.841 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.454     | 0.918    | 0.922     | 0.901  | 0.978 | 0.977 | 0.058 | 0.005 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.969    | 0.971     | 0.97   | 0.995 | 0.995 | 0.024 | 0.006 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.964    | 0.976     | 0.986  | 0.996 | 0.999 | 0.018 | 0.014 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.299     | 0.871    | 0.85      | 0.795  | 0.954 | 0.914 | 0.079 | 0.012 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.964    | 0.98      | 0.984  | 0.993 | 0.998 | 0.022 | 0.009 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.108     | 0.907    | 0.862     | 0.809  | 0.981 | 0.905 | 0.027 | 0.013 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.333     | 0.771    | 0.648     | 0.728  | 0.865 | 0.702 | 0.142 | 0.063 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.109     | 0.765    | 0.49      | 0.691  | 0.942 | 0.56  | 0.053 | 0.048 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.02      | 0.598    | 0.5       | 0.151  | 0.897 | 0.357 | 0.085 | 0.102 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.501     | 0.73     | 0.651     | 0.778  | 0.822 | 0.751 | 0.172 | 0.041 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.307     | 0.712    | 0.49      | 0.694  | 0.821 | 0.562 | 0.148 | 0.133 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.897 | 0.716 |
| 1-10  | UK      | 219 | 0.924 | 0.68  |
| 11-30 | AU      | 717 | 0.855 | 0.746 |
| 11-30 | UK      | 429 | 0.848 | 0.708 |
| 31-80 | AU      | 602 | 0.871 | 0.745 |
| 31-80 | UK      | 385 | 0.923 | 0.71  |
| >80   | AU      | 320 | 0.853 | 0.679 |
| >80   | UK      | 110 | 0.924 | 0.596 |
