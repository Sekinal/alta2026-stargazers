# Validation report: `cls_e5_large_lr`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.027 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.911   | 0.958   | 0.747   | 0.756   | 0.829 | 0.011   | 0.808       |
| global+ukg    | 0.915   | 0.959   | 0.751   | 0.756   | 0.833 | 0.007   | 0.807       |
| variety+ukg   | 0.915   | 0.959   | 0.752   | 0.753   | 0.834 | 0.009   | 0.824       |
| segment+ukg   | 0.919   | 0.963   | 0.77    | 0.753   | 0.836 | 0.007   | 0.828       |
| prev+ukg      | 0.919   | 0.963   | 0.745   | 0.757   | 0.832 | 0.013   | 0.823       |
| prevtuned+ukg | 0.919   | 0.963   | 0.762   | 0.756   | 0.837 | 0.011   | 0.825       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8360**

## Expected test score for `segment+ukg`
**Point estimate 0.8360 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8028, 0.8691]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.919 | 0.92  | 0.014 | 0.897 | 0.943 | null        |
| sent-UK | 0.963 | 0.962 | 0.009 | 0.946 | 0.977 | null        |
| sarc-AU | 0.77  | 0.777 | 0.019 | 0.745 | 0.808 | null        |
| sarc-UK | 0.753 | 0.755 | 0.038 | 0.692 | 0.815 | null        |
| score   | 0.836 | 0.834 | 0.017 | 0.804 | 0.859 | 0.998       |

binding varieties: sent: AU binds 99.5%, sarc: UK binds 71.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.971   | 0.772   | 0.742   | 0.825 |
| 1    | 0.928   | 0.959   | 0.771   | 0.755   | 0.842 |
| 2    | 0.923   | 0.963   | 0.759   | 0.754   | 0.839 |
| 3    | 0.915   | 0.962   | 0.749   | 0.754   | 0.832 |
| 4    | 0.92    | 0.959   | 0.797   | 0.758   | 0.839 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.454     | 0.919    | 0.923     | 0.902  | 0.973 | 0.971 | 0.069 | 0.064 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.963    | 0.968     | 0.961  | 0.992 | 0.993 | 0.033 | 0.039 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.742     | 0.965    | 0.977     | 0.986  | 0.995 | 0.998 | 0.026 | 0.052 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.3       | 0.872    | 0.85      | 0.798  | 0.947 | 0.907 | 0.092 | 0.077 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.749     | 0.964    | 0.981     | 0.983  | 0.994 | 0.998 | 0.027 | 0.044 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.858    | 0.788     | 0.71   | 0.979 | 0.874 | 0.045 | 0.078 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.338     | 0.77     | 0.644     | 0.733  | 0.854 | 0.659 | 0.148 | 0.079 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.079     | 0.753    | 0.539     | 0.55   | 0.938 | 0.541 | 0.059 | 0.062 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.007     | 0.532    | 0.5       | 0.055  | 0.88  | 0.315 | 0.093 | 0.129 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.515     | 0.729    | 0.645     | 0.792  | 0.812 | 0.729 | 0.177 | 0.06  |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.222     | 0.709    | 0.539     | 0.552  | 0.807 | 0.543 | 0.163 | 0.172 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.887 | 0.705 |
| 1-10  | UK      | 219 | 0.905 | 0.657 |
| 11-30 | AU      | 717 | 0.865 | 0.759 |
| 11-30 | UK      | 429 | 0.775 | 0.738 |
| 31-80 | AU      | 602 | 0.859 | 0.742 |
| 31-80 | UK      | 385 | 0.855 | 0.672 |
| >80   | AU      | 320 | 0.871 | 0.656 |
| >80   | UK      | 110 | 0.877 | 0.569 |
