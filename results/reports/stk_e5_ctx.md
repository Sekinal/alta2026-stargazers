# Validation report: `stk_e5_ctx`

generated 2026-09-18 12:47, code ebc1a23-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.044 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.54 (valid 0.847, pool 0.840; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.919   | 0.969   | 0.765   | 0.721   | 0.82  | 0.016   | 0.8         |
| global+ukg    | 0.913   | 0.967   | 0.768   | 0.771   | 0.84  | 0.011   | 0.845       |
| variety+ukg   | 0.914   | 0.971   | 0.773   | 0.785   | 0.843 | 0.007   | 0.847       |
| segment+ukg   | 0.911   | 0.968   | 0.769   | 0.785   | 0.84  | 0.005   | 0.847       |
| prev+ukg      | 0.922   | 0.967   | 0.764   | 0.783   | 0.843 | 0.009   | 0.847       |
| prevtuned+ukg | 0.922   | 0.967   | 0.766   | 0.783   | 0.844 | 0.009   | 0.855       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8400**

## Expected test score for `segment+ukg`
**Point estimate 0.8400 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8126, 0.8675]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.911 | 0.912 | 0.014 | 0.886 | 0.935 | null        |
| sent-UK | 0.968 | 0.968 | 0.008 | 0.953 | 0.982 | null        |
| sarc-AU | 0.769 | 0.776 | 0.02  | 0.743 | 0.807 | null        |
| sarc-UK | 0.785 | 0.788 | 0.035 | 0.729 | 0.845 | null        |
| score   | 0.84  | 0.838 | 0.014 | 0.815 | 0.86  | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 37.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.893   | 0.972   | 0.78    | 0.783   | 0.837 |
| 1    | 0.918   | 0.966   | 0.773   | 0.785   | 0.845 |
| 2    | 0.916   | 0.967   | 0.767   | 0.781   | 0.842 |
| 3    | 0.915   | 0.97    | 0.753   | 0.778   | 0.834 |
| 4    | 0.913   | 0.967   | 0.769   | 0.798   | 0.841 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.465     | 0.911    | 0.905     | 0.905  | 0.977 | 0.977 | 0.058 | 0.005 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.968    | 0.973     | 0.967  | 0.995 | 0.995 | 0.024 | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.966    | 0.98      | 0.985  | 0.996 | 0.999 | 0.018 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.317     | 0.86     | 0.811     | 0.806  | 0.952 | 0.912 | 0.079 | 0.009 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.964    | 0.98      | 0.984  | 0.993 | 0.998 | 0.022 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.099     | 0.894    | 0.876     | 0.756  | 0.983 | 0.907 | 0.028 | 0.006 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.325     | 0.769    | 0.652     | 0.713  | 0.879 | 0.733 | 0.128 | 0.012 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.095     | 0.785    | 0.553     | 0.675  | 0.946 | 0.603 | 0.044 | 0.009 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.022     | 0.595    | 0.458     | 0.151  | 0.898 | 0.35  | 0.053 | 0.03  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.487     | 0.731    | 0.657     | 0.762  | 0.825 | 0.754 | 0.169 | 0.022 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.147 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.266     | 0.742    | 0.553     | 0.677  | 0.833 | 0.605 | 0.123 | 0.021 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.892 | 0.71  |
| 1-10  | UK      | 219 | 0.915 | 0.711 |
| 11-30 | AU      | 717 | 0.848 | 0.754 |
| 11-30 | UK      | 429 | 0.832 | 0.74  |
| 31-80 | AU      | 602 | 0.852 | 0.728 |
| 31-80 | UK      | 385 | 0.915 | 0.738 |
| >80   | AU      | 320 | 0.836 | 0.694 |
| >80   | UK      | 110 | 0.877 | 0.647 |
