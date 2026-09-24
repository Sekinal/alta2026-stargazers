# Validation report: `enc_deb_l_x5`

generated 2026-09-18 11:14, code 1c9969c-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] test split: 757 rows bound to experiments/encoders/fake_test.csv (d9e22447974963d0)
  [OK  ] fold_hash matches current folds
  [OK  ] permutation canary passed: all members passed (enc_deb_l_s13, enc_deb_l_s14, enc_deb_l_s15, enc_deb_l_s16, enc_deb_l_s17)
  [OK  ] per-fold score spread 0.028 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.18 (valid 0.838, pool 0.863; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.931   | 0.974   | 0.793   | 0.788   | 0.86  | 0.009   | 0.804       |
| global+ukg    | 0.933   | 0.972   | 0.792   | 0.815   | 0.862 | 0.003   | 0.811       |
| variety+ukg   | 0.933   | 0.971   | 0.789   | 0.829   | 0.861 | 0.005   | 0.842       |
| segment+ukg   | 0.931   | 0.971   | 0.796   | 0.829   | 0.863 | 0.008   | 0.838       |
| prev+ukg      | 0.933   | 0.971   | 0.799   | 0.818   | 0.866 | 0.005   | 0.829       |
| prevtuned+ukg | 0.933   | 0.971   | 0.798   | 0.821   | 0.866 | 0.005   | 0.842       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8635**

## Expected test score for `segment+ukg`
**Point estimate 0.8635 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8381, 0.8889]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.931 | 0.932 | 0.013 | 0.911 | 0.951 | null        |
| sent-UK | 0.971 | 0.972 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.796 | 0.799 | 0.02  | 0.765 | 0.832 | null        |
| sarc-UK | 0.829 | 0.832 | 0.035 | 0.773 | 0.884 | null        |
| score   | 0.863 | 0.863 | 0.013 | 0.841 | 0.884 | 1.0         |

binding varieties: sent: AU binds 99.6%, sarc: UK binds 20.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.977   | 0.833   | 0.859   | 0.876 |
| 1    | 0.93    | 0.973   | 0.788   | 0.86    | 0.859 |
| 2    | 0.946   | 0.97    | 0.784   | 0.78    | 0.863 |
| 3    | 0.923   | 0.969   | 0.787   | 0.844   | 0.855 |
| 4    | 0.936   | 0.967   | 0.784   | 0.801   | 0.86  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.474     | 0.931    | 0.917     | 0.936  | 0.98  | 0.98  | 0.056 | 0.035 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.971    | 0.978     | 0.967  | 0.995 | 0.995 | 0.023 | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.734     | 0.975    | 0.987     | 0.986  | 0.998 | 0.999 | 0.013 | 0.006 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.335     | 0.892    | 0.835     | 0.875  | 0.957 | 0.922 | 0.079 | 0.063 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.969    | 0.985     | 0.983  | 0.994 | 0.998 | 0.018 | 0.007 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.902    | 0.879     | 0.779  | 0.979 | 0.916 | 0.03  | 0.015 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.326     | 0.796    | 0.687     | 0.755  | 0.903 | 0.768 | 0.132 | 0.099 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.079     | 0.829    | 0.68      | 0.691  | 0.96  | 0.705 | 0.042 | 0.02  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.057     | 0.69     | 0.459     | 0.384  | 0.928 | 0.555 | 0.049 | 0.034 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.471     | 0.767    | 0.701     | 0.787  | 0.854 | 0.781 | 0.177 | 0.132 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.948 | 0.009 | 0.002 | 0.002 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.221     | 0.799    | 0.68      | 0.694  | 0.881 | 0.719 | 0.113 | 0.085 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.893 | 0.747 |
| 1-10  | UK      | 219 | 0.921 | 0.768 |
| 11-30 | AU      | 717 | 0.891 | 0.79  |
| 11-30 | UK      | 429 | 0.861 | 0.8   |
| 31-80 | AU      | 602 | 0.889 | 0.778 |
| 31-80 | UK      | 385 | 0.908 | 0.804 |
| >80   | AU      | 320 | 0.879 | 0.713 |
| >80   | UK      | 110 | 0.893 | 0.75  |
