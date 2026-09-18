# Validation report: `cls_e5gte_lr`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.017 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.919   | 0.969   | 0.752   | 0.744   | 0.831 | 0.007   | 0.83        |
| global+ukg    | 0.918   | 0.969   | 0.75    | 0.751   | 0.834 | 0.005   | 0.827       |
| variety+ukg   | 0.918   | 0.971   | 0.751   | 0.752   | 0.835 | 0.005   | 0.84        |
| segment+ukg   | 0.912   | 0.968   | 0.768   | 0.752   | 0.832 | 0.007   | 0.836       |
| prev+ukg      | 0.918   | 0.967   | 0.755   | 0.75    | 0.834 | 0.014   | 0.831       |
| prevtuned+ukg | 0.918   | 0.967   | 0.755   | 0.746   | 0.832 | 0.011   | 0.83        |

**Headline policy: `segment+ukg`** (default policy); **score 0.8320**

## Expected test score for `segment+ukg`
**Point estimate 0.8320 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7986, 0.8654]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.912 | 0.913 | 0.014 | 0.889 | 0.935 | null        |
| sent-UK | 0.968 | 0.968 | 0.009 | 0.953 | 0.982 | null        |
| sarc-AU | 0.768 | 0.773 | 0.021 | 0.738 | 0.806 | null        |
| sarc-UK | 0.752 | 0.752 | 0.039 | 0.689 | 0.814 | null        |
| score   | 0.832 | 0.828 | 0.017 | 0.798 | 0.854 | 0.997       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 67.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.902   | 0.972   | 0.785   | 0.748   | 0.825 |
| 1    | 0.923   | 0.961   | 0.764   | 0.754   | 0.839 |
| 2    | 0.92    | 0.969   | 0.757   | 0.736   | 0.828 |
| 3    | 0.908   | 0.969   | 0.746   | 0.744   | 0.826 |
| 4    | 0.907   | 0.969   | 0.786   | 0.774   | 0.84  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.455     | 0.912    | 0.915     | 0.895  | 0.974 | 0.974 | 0.061 | 0.008 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.968    | 0.973     | 0.965  | 0.995 | 0.995 | 0.024 | 0.017 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.738     | 0.958    | 0.976     | 0.98   | 0.995 | 0.998 | 0.021 | 0.017 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.302     | 0.864    | 0.835     | 0.791  | 0.947 | 0.908 | 0.082 | 0.011 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.963    | 0.98      | 0.983  | 0.994 | 0.998 | 0.02  | 0.013 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.099     | 0.894    | 0.876     | 0.756  | 0.977 | 0.889 | 0.032 | 0.019 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.327     | 0.768    | 0.649     | 0.714  | 0.86  | 0.683 | 0.143 | 0.066 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.083     | 0.752    | 0.526     | 0.562  | 0.933 | 0.515 | 0.058 | 0.058 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.025     | 0.591    | 0.407     | 0.151  | 0.893 | 0.35  | 0.08  | 0.104 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.489     | 0.73     | 0.656     | 0.763  | 0.81  | 0.73  | 0.177 | 0.046 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.233     | 0.706    | 0.526     | 0.565  | 0.789 | 0.517 | 0.163 | 0.159 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.881 | 0.684 |
| 1-10  | UK      | 219 | 0.916 | 0.645 |
| 11-30 | AU      | 717 | 0.849 | 0.768 |
| 11-30 | UK      | 429 | 0.844 | 0.736 |
| 31-80 | AU      | 602 | 0.864 | 0.745 |
| 31-80 | UK      | 385 | 0.921 | 0.681 |
| >80   | AU      | 320 | 0.849 | 0.664 |
| >80   | UK      | 110 | 0.761 | 0.559 |
