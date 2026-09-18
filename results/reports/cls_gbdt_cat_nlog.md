# Validation report: `cls_gbdt_cat_nlog`

generated 2026-09-18 01:51, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.034 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.79 (valid 0.827, pool 0.852; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.92    | 0.972   | 0.771   | 0.768   | 0.844 | 0.016   | 0.834       |
| global+ukg    | 0.92    | 0.972   | 0.783   | 0.79    | 0.851 | 0.008   | 0.833       |
| variety+ukg   | 0.92    | 0.971   | 0.782   | 0.79    | 0.851 | 0.007   | 0.827       |
| segment+ukg   | 0.92    | 0.971   | 0.784   | 0.79    | 0.852 | 0.006   | 0.827       |
| prev+ukg      | 0.924   | 0.972   | 0.774   | 0.792   | 0.849 | 0.009   | 0.836       |
| prevtuned+ukg | 0.924   | 0.972   | 0.778   | 0.797   | 0.851 | 0.009   | 0.823       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8517**

## Expected test score for `segment+ukg`
**Point estimate 0.8517 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8227, 0.8806]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.92  | 0.921 | 0.014 | 0.897 | 0.943 | null        |
| sent-UK | 0.971 | 0.971 | 0.008 | 0.956 | 0.984 | null        |
| sarc-AU | 0.784 | 0.791 | 0.02  | 0.757 | 0.823 | null        |
| sarc-UK | 0.79  | 0.793 | 0.037 | 0.73  | 0.855 | null        |
| score   | 0.852 | 0.848 | 0.015 | 0.823 | 0.871 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 47.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.906   | 0.971   | 0.817   | 0.775   | 0.841 |
| 1    | 0.92    | 0.964   | 0.774   | 0.808   | 0.847 |
| 2    | 0.925   | 0.97    | 0.782   | 0.756   | 0.84  |
| 3    | 0.921   | 0.98    | 0.761   | 0.812   | 0.841 |
| 4    | 0.926   | 0.969   | 0.784   | 0.798   | 0.855 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.459     | 0.92     | 0.919     | 0.908  | 0.979 | 0.978 | 0.057 | 0.015 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.523     | 0.971    | 0.971     | 0.973  | 0.996 | 0.996 | 0.021 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.747     | 0.97     | 0.976     | 0.992  | 0.997 | 0.999 | 0.018 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.305     | 0.872    | 0.843     | 0.805  | 0.956 | 0.916 | 0.078 | 0.022 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.754     | 0.963    | 0.978     | 0.986  | 0.995 | 0.998 | 0.019 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.105     | 0.917    | 0.892     | 0.817  | 0.987 | 0.93  | 0.025 | 0.007 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.304     | 0.784    | 0.689     | 0.706  | 0.883 | 0.732 | 0.126 | 0.012 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.08      | 0.79     | 0.603     | 0.622  | 0.954 | 0.643 | 0.042 | 0.005 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.007     | 0.52     | 0.429     | 0.041  | 0.9   | 0.378 | 0.051 | 0.014 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.464     | 0.754    | 0.691     | 0.763  | 0.83  | 0.751 | 0.167 | 0.021 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.936 | 0.007 | 0.001 | 0.004 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.225     | 0.752    | 0.603     | 0.625  | 0.852 | 0.645 | 0.116 | 0.023 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.903 | 0.742 |
| 1-10  | UK      | 219 | 0.924 | 0.715 |
| 11-30 | AU      | 717 | 0.857 | 0.779 |
| 11-30 | UK      | 429 | 0.868 | 0.758 |
| 31-80 | AU      | 602 | 0.86  | 0.765 |
| 31-80 | UK      | 385 | 0.93  | 0.753 |
| >80   | AU      | 320 | 0.863 | 0.683 |
| >80   | UK      | 110 | 0.964 | 0.55  |
