# Validation report: `cls_gbdt_lgbm_raw2k`

generated 2026-09-18 01:51, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.035 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.60 (valid 0.829, pool 0.838; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.923   | 0.964   | 0.757   | 0.647   | 0.785 | 0.017   | 0.729       |
| global+ukg    | 0.921   | 0.965   | 0.765   | 0.754   | 0.837 | 0.008   | 0.828       |
| variety+ukg   | 0.921   | 0.967   | 0.774   | 0.755   | 0.838 | 0.013   | 0.828       |
| segment+ukg   | 0.921   | 0.969   | 0.768   | 0.755   | 0.838 | 0.014   | 0.829       |
| prev+ukg      | 0.921   | 0.967   | 0.759   | 0.75    | 0.836 | 0.017   | 0.813       |
| prevtuned+ukg | 0.921   | 0.967   | 0.762   | 0.758   | 0.84  | 0.016   | 0.821       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8381**

## Expected test score for `segment+ukg`
**Point estimate 0.8381 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8069, 0.8693]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.921 | 0.922 | 0.013 | 0.9   | 0.943 | null        |
| sent-UK | 0.969 | 0.969 | 0.009 | 0.953 | 0.982 | null        |
| sarc-AU | 0.768 | 0.774 | 0.02  | 0.742 | 0.806 | null        |
| sarc-UK | 0.755 | 0.758 | 0.038 | 0.694 | 0.819 | null        |
| score   | 0.838 | 0.835 | 0.016 | 0.807 | 0.859 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 64.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.974   | 0.782   | 0.742   | 0.83  |
| 1    | 0.919   | 0.967   | 0.785   | 0.811   | 0.852 |
| 2    | 0.921   | 0.966   | 0.772   | 0.719   | 0.82  |
| 3    | 0.918   | 0.97    | 0.725   | 0.747   | 0.821 |
| 4    | 0.928   | 0.967   | 0.773   | 0.754   | 0.841 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.441     | 0.921    | 0.938     | 0.89   | 0.978 | 0.977 | 0.058 | 0.019 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.526     | 0.969    | 0.967     | 0.973  | 0.995 | 0.995 | 0.025 | 0.006 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.742     | 0.972    | 0.981     | 0.99   | 0.997 | 0.999 | 0.017 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.279     | 0.87     | 0.877     | 0.767  | 0.954 | 0.91  | 0.081 | 0.031 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.758     | 0.961    | 0.974     | 0.987  | 0.993 | 0.998 | 0.023 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.106     | 0.911    | 0.876     | 0.809  | 0.984 | 0.916 | 0.027 | 0.01  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.342     | 0.768    | 0.639     | 0.736  | 0.88  | 0.741 | 0.129 | 0.032 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.093     | 0.755    | 0.507     | 0.606  | 0.947 | 0.57  | 0.047 | 0.016 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.014     | 0.597    | 0.667     | 0.137  | 0.9   | 0.388 | 0.052 | 0.028 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.518     | 0.723    | 0.639     | 0.788  | 0.824 | 0.76  | 0.17  | 0.034 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.998 | 0.2   | 0.001 | 0.005 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.261     | 0.706    | 0.507     | 0.609  | 0.827 | 0.571 | 0.13  | 0.042 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.891 | 0.713 |
| 1-10  | UK      | 219 | 0.919 | 0.662 |
| 11-30 | AU      | 717 | 0.856 | 0.734 |
| 11-30 | UK      | 429 | 0.882 | 0.716 |
| 31-80 | AU      | 602 | 0.877 | 0.717 |
| 31-80 | UK      | 385 | 0.921 | 0.708 |
| >80   | AU      | 320 | 0.842 | 0.716 |
| >80   | UK      | 110 | 0.877 | 0.535 |
