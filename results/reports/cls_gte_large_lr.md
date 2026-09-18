# Validation report: `cls_gte_large_lr`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.019 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.897   | 0.958   | 0.723   | 0.692   | 0.795 | 0.007   | 0.785       |
| global+ukg    | 0.9     | 0.952   | 0.723   | 0.686   | 0.793 | 0.008   | 0.804       |
| variety+ukg   | 0.901   | 0.959   | 0.717   | 0.686   | 0.794 | 0.008   | 0.802       |
| segment+ukg   | 0.905   | 0.96    | 0.749   | 0.686   | 0.796 | 0.008   | 0.808       |
| prev+ukg      | 0.905   | 0.961   | 0.727   | 0.691   | 0.798 | 0.005   | 0.803       |
| prevtuned+ukg | 0.905   | 0.961   | 0.745   | 0.693   | 0.799 | 0.005   | 0.804       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7955**

## Expected test score for `segment+ukg`
**Point estimate 0.7955 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7583, 0.8328]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.905 | 0.906 | 0.015 | 0.881 | 0.929 | null        |
| sent-UK | 0.96  | 0.959 | 0.01  | 0.943 | 0.974 | null        |
| sarc-AU | 0.749 | 0.754 | 0.019 | 0.724 | 0.785 | null        |
| sarc-UK | 0.686 | 0.685 | 0.036 | 0.626 | 0.744 | null        |
| score   | 0.796 | 0.795 | 0.019 | 0.763 | 0.825 | 0.866       |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 95.2%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.894   | 0.971   | 0.757   | 0.673   | 0.784 |
| 1    | 0.922   | 0.952   | 0.747   | 0.683   | 0.802 |
| 2    | 0.918   | 0.961   | 0.742   | 0.665   | 0.792 |
| 3    | 0.903   | 0.959   | 0.735   | 0.704   | 0.804 |
| 4    | 0.886   | 0.956   | 0.764   | 0.701   | 0.793 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.458     | 0.905    | 0.904     | 0.891  | 0.964 | 0.963 | 0.079 | 0.062 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.96     | 0.965     | 0.958  | 0.989 | 0.991 | 0.039 | 0.046 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.962    | 0.981     | 0.979  | 0.991 | 0.996 | 0.031 | 0.056 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.31      | 0.85     | 0.805     | 0.783  | 0.931 | 0.883 | 0.104 | 0.08  |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.743     | 0.959    | 0.982     | 0.975  | 0.992 | 0.997 | 0.028 | 0.045 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.115     | 0.862    | 0.756     | 0.756  | 0.968 | 0.823 | 0.058 | 0.101 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.398     | 0.749    | 0.587     | 0.788  | 0.836 | 0.629 | 0.157 | 0.092 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.106     | 0.686    | 0.372     | 0.51   | 0.909 | 0.409 | 0.071 | 0.074 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.037     | 0.64     | 0.45      | 0.247  | 0.888 | 0.318 | 0.101 | 0.164 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.592     | 0.689    | 0.591     | 0.835  | 0.786 | 0.699 | 0.188 | 0.062 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.298     | 0.617    | 0.372     | 0.512  | 0.71  | 0.41  | 0.198 | 0.207 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.862 | 0.654 |
| 1-10  | UK      | 219 | 0.879 | 0.565 |
| 11-30 | AU      | 717 | 0.822 | 0.705 |
| 11-30 | UK      | 429 | 0.813 | 0.645 |
| 31-80 | AU      | 602 | 0.863 | 0.714 |
| 31-80 | UK      | 385 | 0.869 | 0.59  |
| >80   | AU      | 320 | 0.848 | 0.635 |
| >80   | UK      | 110 | 0.877 | 0.587 |
