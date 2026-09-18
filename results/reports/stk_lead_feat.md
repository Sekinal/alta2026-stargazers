# Validation report: `stk_lead_feat`

generated 2026-09-18 02:11, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.026 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.03 (valid 0.852, pool 0.864; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.932   | 0.977   | 0.796   | 0.813   | 0.864 | 0.011   | 0.833       |
| global+ukg    | 0.93    | 0.977   | 0.801   | 0.814   | 0.865 | 0.007   | 0.844       |
| variety+ukg   | 0.93    | 0.976   | 0.801   | 0.823   | 0.865 | 0.007   | 0.852       |
| segment+ukg   | 0.93    | 0.977   | 0.799   | 0.823   | 0.864 | 0.006   | 0.852       |
| prev+ukg      | 0.933   | 0.975   | 0.797   | 0.82    | 0.865 | 0.009   | 0.85        |
| prevtuned+ukg | 0.933   | 0.975   | 0.798   | 0.808   | 0.865 | 0.008   | 0.846       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8642**

## Expected test score for `segment+ukg`
**Point estimate 0.8642 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8391, 0.8893]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.93  | 0.93  | 0.013 | 0.908 | 0.951 | null        |
| sent-UK | 0.977 | 0.977 | 0.007 | 0.964 | 0.99  | null        |
| sarc-AU | 0.799 | 0.803 | 0.019 | 0.771 | 0.834 | null        |
| sarc-UK | 0.823 | 0.825 | 0.035 | 0.765 | 0.881 | null        |
| score   | 0.864 | 0.863 | 0.013 | 0.84  | 0.882 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 27.2%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.916   | 0.977   | 0.823   | 0.845   | 0.869 |
| 1    | 0.928   | 0.975   | 0.802   | 0.861   | 0.865 |
| 2    | 0.936   | 0.973   | 0.79    | 0.772   | 0.854 |
| 3    | 0.935   | 0.98    | 0.778   | 0.84    | 0.856 |
| 4    | 0.933   | 0.98    | 0.8     | 0.794   | 0.863 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.467     | 0.93     | 0.922     | 0.927  | 0.983 | 0.982 | 0.05  | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.521     | 0.977    | 0.98      | 0.976  | 0.996 | 0.997 | 0.018 | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.737     | 0.974    | 0.985     | 0.987  | 0.998 | 0.999 | 0.013 | 0.007 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.322     | 0.889    | 0.845     | 0.853  | 0.962 | 0.93  | 0.07  | 0.014 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.753     | 0.975    | 0.985     | 0.991  | 0.996 | 0.998 | 0.015 | 0.003 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.918    | 0.913     | 0.802  | 0.982 | 0.927 | 0.024 | 0.005 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.333     | 0.799    | 0.685     | 0.769  | 0.905 | 0.78  | 0.115 | 0.023 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.08      | 0.823    | 0.664     | 0.683  | 0.965 | 0.726 | 0.036 | 0.006 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.041     | 0.679    | 0.523     | 0.315  | 0.933 | 0.5   | 0.045 | 0.011 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.491     | 0.767    | 0.692     | 0.809  | 0.858 | 0.796 | 0.152 | 0.023 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.911 | 0.005 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.224     | 0.791    | 0.664     | 0.685  | 0.888 | 0.728 | 0.101 | 0.021 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.906 | 0.754 |
| 1-10  | UK      | 219 | 0.927 | 0.764 |
| 11-30 | AU      | 717 | 0.879 | 0.78  |
| 11-30 | UK      | 429 | 0.899 | 0.781 |
| 31-80 | AU      | 602 | 0.884 | 0.777 |
| 31-80 | UK      | 385 | 0.912 | 0.807 |
| >80   | AU      | 320 | 0.883 | 0.724 |
| >80   | UK      | 110 | 0.933 | 0.714 |
