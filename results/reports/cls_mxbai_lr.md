# Validation report: `cls_mxbai_lr`

generated 2026-09-17 23:49, code 6844a92-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.015 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.907   | 0.963   | 0.722   | 0.718   | 0.813 | 0.005   | 0.82        |
| global+ukg    | 0.904   | 0.963   | 0.724   | 0.712   | 0.808 | 0.007   | 0.828       |
| variety+ukg   | 0.903   | 0.962   | 0.722   | 0.711   | 0.807 | 0.006   | 0.824       |
| segment+ukg   | 0.903   | 0.963   | 0.748   | 0.711   | 0.807 | 0.005   | 0.823       |
| prev+ukg      | 0.906   | 0.961   | 0.736   | 0.711   | 0.809 | 0.006   | 0.829       |
| prevtuned+ukg | 0.906   | 0.961   | 0.742   | 0.71    | 0.808 | 0.007   | 0.826       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8070**

## Expected test score for `segment+ukg`
**Point estimate 0.8070 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7711, 0.8429]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.903 | 0.903 | 0.014 | 0.88  | 0.927 | null        |
| sent-UK | 0.963 | 0.963 | 0.009 | 0.948 | 0.977 | null        |
| sarc-AU | 0.748 | 0.753 | 0.019 | 0.721 | 0.784 | null        |
| sarc-UK | 0.711 | 0.712 | 0.038 | 0.649 | 0.772 | null        |
| score   | 0.807 | 0.805 | 0.018 | 0.773 | 0.833 | 0.948       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 83.5%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.884   | 0.967   | 0.752   | 0.72    | 0.802 |
| 1    | 0.922   | 0.961   | 0.752   | 0.699   | 0.81  |
| 2    | 0.912   | 0.961   | 0.745   | 0.699   | 0.805 |
| 3    | 0.888   | 0.961   | 0.735   | 0.716   | 0.802 |
| 4    | 0.907   | 0.966   | 0.756   | 0.72    | 0.813 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.45      | 0.903    | 0.908     | 0.881  | 0.966 | 0.967 | 0.069 | 0.012 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.523     | 0.963    | 0.964     | 0.965  | 0.993 | 0.994 | 0.029 | 0.023 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.735     | 0.955    | 0.976     | 0.976  | 0.99  | 0.996 | 0.027 | 0.024 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.297     | 0.848    | 0.819     | 0.762  | 0.934 | 0.889 | 0.092 | 0.011 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.758     | 0.963    | 0.975     | 0.988  | 0.993 | 0.997 | 0.024 | 0.025 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.098     | 0.86     | 0.812     | 0.695  | 0.968 | 0.86  | 0.039 | 0.02  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.394     | 0.748    | 0.587     | 0.78   | 0.834 | 0.628 | 0.159 | 0.094 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.096     | 0.711    | 0.427     | 0.53   | 0.915 | 0.428 | 0.069 | 0.073 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.042     | 0.641    | 0.422     | 0.26   | 0.892 | 0.367 | 0.102 | 0.166 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.583     | 0.69     | 0.594     | 0.825  | 0.784 | 0.7   | 0.189 | 0.055 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.27      | 0.652    | 0.427     | 0.532  | 0.731 | 0.429 | 0.193 | 0.204 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.862 | 0.701 |
| 1-10  | UK      | 219 | 0.887 | 0.592 |
| 11-30 | AU      | 717 | 0.83  | 0.698 |
| 11-30 | UK      | 429 | 0.815 | 0.662 |
| 31-80 | AU      | 602 | 0.846 | 0.703 |
| 31-80 | UK      | 385 | 0.861 | 0.62  |
| >80   | AU      | 320 | 0.848 | 0.623 |
| >80   | UK      | 110 | 0.761 | 0.647 |
