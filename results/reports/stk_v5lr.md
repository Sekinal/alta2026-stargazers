# Validation report: `stk_v5lr`

generated 2026-09-18 12:41, code ebc1a23-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.023 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.09 (valid 0.871, pool 0.872; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.933   | 0.977   | 0.807   | 0.822   | 0.87  | 0.01    | 0.845       |
| global+ukg    | 0.934   | 0.977   | 0.812   | 0.827   | 0.873 | 0.012   | 0.862       |
| variety+ukg   | 0.935   | 0.978   | 0.812   | 0.84    | 0.874 | 0.011   | 0.867       |
| segment+ukg   | 0.937   | 0.978   | 0.807   | 0.84    | 0.872 | 0.011   | 0.871       |
| prev+ukg      | 0.934   | 0.978   | 0.805   | 0.842   | 0.87  | 0.011   | 0.859       |
| prevtuned+ukg | 0.934   | 0.978   | 0.804   | 0.838   | 0.869 | 0.009   | 0.865       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8719**

## Expected test score for `segment+ukg`
**Point estimate 0.8719 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8470, 0.8969]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.937 | 0.938 | 0.012 | 0.916 | 0.957 | null        |
| sent-UK | 0.978 | 0.978 | 0.007 | 0.966 | 0.99  | null        |
| sarc-AU | 0.807 | 0.81  | 0.021 | 0.775 | 0.842 | null        |
| sarc-UK | 0.84  | 0.843 | 0.033 | 0.785 | 0.896 | null        |
| score   | 0.872 | 0.872 | 0.013 | 0.85  | 0.891 | 1.0         |

binding varieties: sent: AU binds 99.7%, sarc: UK binds 18.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.927   | 0.981   | 0.846   | 0.847   | 0.887 |
| 1    | 0.935   | 0.978   | 0.808   | 0.889   | 0.871 |
| 2    | 0.951   | 0.975   | 0.805   | 0.801   | 0.876 |
| 3    | 0.935   | 0.976   | 0.779   | 0.832   | 0.857 |
| 4    | 0.938   | 0.976   | 0.794   | 0.831   | 0.866 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.476     | 0.937    | 0.922     | 0.945  | 0.984 | 0.984 | 0.047 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.978    | 0.982     | 0.975  | 0.996 | 0.996 | 0.018 | 0.002 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.986    | 0.994     | 0.991  | 0.999 | 0.999 | 0.011 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.338     | 0.898    | 0.839     | 0.888  | 0.965 | 0.937 | 0.067 | 0.018 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.979    | 0.988     | 0.991  | 0.995 | 0.998 | 0.015 | 0.002 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.1       | 0.911    | 0.904     | 0.786  | 0.986 | 0.935 | 0.023 | 0.005 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.309     | 0.807    | 0.716     | 0.746  | 0.91  | 0.791 | 0.112 | 0.013 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.085     | 0.84     | 0.676     | 0.739  | 0.97  | 0.746 | 0.034 | 0.002 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.07      | 0.688    | 0.413     | 0.425  | 0.936 | 0.533 | 0.044 | 0.018 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.438     | 0.788    | 0.741     | 0.774  | 0.865 | 0.805 | 0.148 | 0.03  |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.899 | 0.005 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.238     | 0.811    | 0.676     | 0.742  | 0.904 | 0.748 | 0.094 | 0.009 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.926 | 0.779 |
| 1-10  | UK      | 219 | 0.93  | 0.78  |
| 11-30 | AU      | 717 | 0.891 | 0.81  |
| 11-30 | UK      | 429 | 0.897 | 0.805 |
| 31-80 | AU      | 602 | 0.887 | 0.79  |
| 31-80 | UK      | 385 | 0.899 | 0.831 |
| >80   | AU      | 320 | 0.875 | 0.739 |
| >80   | UK      | 110 | 0.847 | 0.695 |
