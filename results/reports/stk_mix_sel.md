# Validation report: `stk_mix_sel`

generated 2026-09-18 00:48, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.035 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.56 (valid 0.847, pool 0.867; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.933   | 0.975   | 0.794   | 0.801   | 0.864 | 0.014   | 0.823       |
| global+ukg    | 0.931   | 0.976   | 0.799   | 0.817   | 0.865 | 0.009   | 0.837       |
| variety+ukg   | 0.931   | 0.973   | 0.799   | 0.808   | 0.865 | 0.009   | 0.845       |
| segment+ukg   | 0.932   | 0.971   | 0.802   | 0.808   | 0.867 | 0.009   | 0.847       |
| prev+ukg      | 0.933   | 0.978   | 0.798   | 0.824   | 0.866 | 0.008   | 0.84        |
| prevtuned+ukg | 0.933   | 0.978   | 0.8     | 0.814   | 0.867 | 0.008   | 0.834       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8669**

## Expected test score for `segment+ukg`
**Point estimate 0.8669 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8396, 0.8943]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.932 | 0.933 | 0.013 | 0.913 | 0.954 | null        |
| sent-UK | 0.971 | 0.971 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.802 | 0.804 | 0.02  | 0.771 | 0.836 | null        |
| sarc-UK | 0.808 | 0.81  | 0.035 | 0.75  | 0.867 | null        |
| score   | 0.867 | 0.862 | 0.014 | 0.838 | 0.884 | 1.0         |

binding varieties: sent: AU binds 99.2%, sarc: UK binds 42.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.922   | 0.969   | 0.834   | 0.815   | 0.868 |
| 1    | 0.936   | 0.969   | 0.795   | 0.834   | 0.866 |
| 2    | 0.931   | 0.966   | 0.797   | 0.76    | 0.846 |
| 3    | 0.936   | 0.978   | 0.78    | 0.851   | 0.858 |
| 4    | 0.936   | 0.975   | 0.801   | 0.776   | 0.856 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.454     | 0.932    | 0.938     | 0.916  | 0.984 | 0.983 | 0.049 | 0.009 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.971    | 0.977     | 0.968  | 0.996 | 0.997 | 0.018 | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.736     | 0.98     | 0.989     | 0.99   | 0.998 | 0.999 | 0.014 | 0.006 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.302     | 0.889    | 0.871     | 0.825  | 0.965 | 0.935 | 0.068 | 0.013 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.745     | 0.977    | 0.99      | 0.986  | 0.996 | 0.998 | 0.014 | 0.002 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.108     | 0.876    | 0.805     | 0.756  | 0.981 | 0.92  | 0.027 | 0.01  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.33      | 0.802    | 0.691     | 0.768  | 0.903 | 0.783 | 0.115 | 0.014 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.081     | 0.808    | 0.633     | 0.659  | 0.964 | 0.73  | 0.036 | 0.007 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.067     | 0.73     | 0.5       | 0.493  | 0.929 | 0.514 | 0.044 | 0.015 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.471     | 0.771    | 0.706     | 0.792  | 0.856 | 0.799 | 0.153 | 0.022 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.354 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.227     | 0.773    | 0.633     | 0.661  | 0.89  | 0.732 | 0.101 | 0.018 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.903 | 0.772 |
| 1-10  | UK      | 219 | 0.882 | 0.747 |
| 11-30 | AU      | 717 | 0.884 | 0.79  |
| 11-30 | UK      | 429 | 0.824 | 0.753 |
| 31-80 | AU      | 602 | 0.886 | 0.779 |
| 31-80 | UK      | 385 | 0.908 | 0.804 |
| >80   | AU      | 320 | 0.872 | 0.707 |
| >80   | UK      | 110 | 0.865 | 0.714 |
