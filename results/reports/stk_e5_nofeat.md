# Validation report: `stk_e5_nofeat`

generated 2026-09-18 02:11, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.047 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.76 (valid 0.851, pool 0.841; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.917   | 0.969   | 0.762   | 0.723   | 0.82  | 0.017   | 0.798       |
| global+ukg    | 0.915   | 0.968   | 0.769   | 0.772   | 0.842 | 0.016   | 0.827       |
| variety+ukg   | 0.915   | 0.969   | 0.771   | 0.781   | 0.843 | 0.011   | 0.851       |
| segment+ukg   | 0.911   | 0.97    | 0.771   | 0.781   | 0.841 | 0.008   | 0.851       |
| prev+ukg      | 0.924   | 0.968   | 0.761   | 0.79    | 0.843 | 0.012   | 0.849       |
| prevtuned+ukg | 0.924   | 0.968   | 0.765   | 0.79    | 0.845 | 0.01    | 0.854       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8411**

## Expected test score for `segment+ukg`
**Point estimate 0.8411 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8128, 0.8694]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.911 | 0.912 | 0.014 | 0.889 | 0.935 | null        |
| sent-UK | 0.97  | 0.97  | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.771 | 0.777 | 0.02  | 0.745 | 0.809 | null        |
| sarc-UK | 0.781 | 0.782 | 0.036 | 0.719 | 0.84  | null        |
| score   | 0.841 | 0.838 | 0.014 | 0.813 | 0.86  | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 44.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.895   | 0.974   | 0.787   | 0.795   | 0.841 |
| 1    | 0.918   | 0.969   | 0.772   | 0.801   | 0.845 |
| 2    | 0.913   | 0.972   | 0.768   | 0.741   | 0.827 |
| 3    | 0.918   | 0.97    | 0.744   | 0.79    | 0.831 |
| 4    | 0.912   | 0.966   | 0.783   | 0.773   | 0.842 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.46      | 0.911    | 0.909     | 0.9    | 0.977 | 0.977 | 0.058 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.97     | 0.976     | 0.967  | 0.995 | 0.995 | 0.024 | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.74      | 0.965    | 0.979     | 0.985  | 0.996 | 0.999 | 0.018 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.31      | 0.859    | 0.82      | 0.795  | 0.952 | 0.913 | 0.079 | 0.01  |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.963    | 0.98      | 0.983  | 0.993 | 0.998 | 0.022 | 0.007 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.096     | 0.91     | 0.918     | 0.771  | 0.983 | 0.912 | 0.027 | 0.005 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.34      | 0.771    | 0.644     | 0.737  | 0.878 | 0.726 | 0.129 | 0.011 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.088     | 0.781    | 0.564     | 0.639  | 0.947 | 0.613 | 0.044 | 0.011 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.022     | 0.595    | 0.458     | 0.151  | 0.898 | 0.351 | 0.054 | 0.03  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.51      | 0.731    | 0.649     | 0.788  | 0.824 | 0.748 | 0.169 | 0.027 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.147 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.247     | 0.74     | 0.564     | 0.641  | 0.837 | 0.615 | 0.122 | 0.028 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.883 | 0.707 |
| 1-10  | UK      | 219 | 0.928 | 0.683 |
| 11-30 | AU      | 717 | 0.85  | 0.756 |
| 11-30 | UK      | 429 | 0.863 | 0.746 |
| 31-80 | AU      | 602 | 0.852 | 0.737 |
| 31-80 | UK      | 385 | 0.921 | 0.735 |
| >80   | AU      | 320 | 0.842 | 0.68  |
| >80   | UK      | 110 | 0.877 | 0.647 |
