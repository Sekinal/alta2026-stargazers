# Validation report: `stk_cls4`

generated 2026-09-18 00:45, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] permutation canary passed (0.6443 vs floor 0.6407)
  [OK  ] per-fold score spread 0.017 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.04 (valid 0.824, pool 0.853; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.928   | 0.973   | 0.785   | 0.752   | 0.84  | 0.006   | 0.811       |
| global+ukg    | 0.924   | 0.973   | 0.789   | 0.789   | 0.856 | 0.012   | 0.837       |
| variety+ukg   | 0.923   | 0.972   | 0.793   | 0.785   | 0.854 | 0.012   | 0.826       |
| segment+ukg   | 0.922   | 0.973   | 0.792   | 0.785   | 0.853 | 0.011   | 0.824       |
| prev+ukg      | 0.933   | 0.973   | 0.786   | 0.785   | 0.859 | 0.014   | 0.831       |
| prevtuned+ukg | 0.933   | 0.973   | 0.791   | 0.79    | 0.862 | 0.014   | 0.829       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8534**

## Expected test score for `segment+ukg`
**Point estimate 0.8534 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8233, 0.8834]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.922 | 0.923 | 0.013 | 0.9   | 0.943 | null        |
| sent-UK | 0.973 | 0.973 | 0.008 | 0.961 | 0.984 | null        |
| sarc-AU | 0.792 | 0.798 | 0.019 | 0.767 | 0.828 | null        |
| sarc-UK | 0.785 | 0.786 | 0.036 | 0.726 | 0.842 | null        |
| score   | 0.853 | 0.849 | 0.015 | 0.822 | 0.873 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 61.7%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.906   | 0.977   | 0.822   | 0.811   | 0.858 |
| 1    | 0.928   | 0.969   | 0.798   | 0.79    | 0.859 |
| 2    | 0.93    | 0.966   | 0.775   | 0.775   | 0.852 |
| 3    | 0.923   | 0.978   | 0.767   | 0.805   | 0.845 |
| 4    | 0.923   | 0.975   | 0.796   | 0.743   | 0.833 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.459     | 0.922    | 0.921     | 0.911  | 0.983 | 0.982 | 0.052 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.973    | 0.974     | 0.974  | 0.996 | 0.996 | 0.02  | 0.004 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.969    | 0.985     | 0.982  | 0.998 | 0.999 | 0.014 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.312     | 0.877    | 0.842     | 0.822  | 0.962 | 0.929 | 0.072 | 0.011 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.755     | 0.971    | 0.981     | 0.99   | 0.995 | 0.998 | 0.017 | 0.002 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.902    | 0.879     | 0.779  | 0.984 | 0.913 | 0.027 | 0.005 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.335     | 0.792    | 0.674     | 0.761  | 0.894 | 0.758 | 0.12  | 0.015 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.088     | 0.785    | 0.569     | 0.647  | 0.953 | 0.652 | 0.041 | 0.006 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.013     | 0.551    | 0.429     | 0.082  | 0.909 | 0.392 | 0.053 | 0.029 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.508     | 0.759    | 0.677     | 0.82   | 0.85  | 0.782 | 0.156 | 0.019 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.169 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.248     | 0.744    | 0.569     | 0.649  | 0.857 | 0.654 | 0.114 | 0.018 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.914 | 0.748 |
| 1-10  | UK      | 219 | 0.921 | 0.688 |
| 11-30 | AU      | 717 | 0.873 | 0.778 |
| 11-30 | UK      | 429 | 0.85  | 0.734 |
| 31-80 | AU      | 602 | 0.861 | 0.767 |
| 31-80 | UK      | 385 | 0.917 | 0.768 |
| >80   | AU      | 320 | 0.845 | 0.707 |
| >80   | UK      | 110 | 0.893 | 0.647 |
