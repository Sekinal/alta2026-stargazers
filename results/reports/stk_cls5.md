# Validation report: `stk_cls5`

generated 2026-09-18 00:45, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] permutation canary passed (0.6437 vs floor 0.6407)
  [OK  ] per-fold score spread 0.018 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.42 (valid 0.834, pool 0.854; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.929   | 0.974   | 0.783   | 0.764   | 0.847 | 0.008   | 0.802       |
| global+ukg    | 0.925   | 0.975   | 0.787   | 0.787   | 0.856 | 0.01    | 0.83        |
| variety+ukg   | 0.926   | 0.973   | 0.787   | 0.783   | 0.855 | 0.011   | 0.835       |
| segment+ukg   | 0.925   | 0.97    | 0.788   | 0.783   | 0.854 | 0.01    | 0.834       |
| prev+ukg      | 0.932   | 0.973   | 0.787   | 0.79    | 0.859 | 0.01    | 0.823       |
| prevtuned+ukg | 0.932   | 0.973   | 0.788   | 0.787   | 0.86  | 0.013   | 0.834       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8541**

## Expected test score for `segment+ukg`
**Point estimate 0.8541 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8256, 0.8826]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.925 | 0.926 | 0.013 | 0.903 | 0.946 | null        |
| sent-UK | 0.97  | 0.97  | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.788 | 0.794 | 0.02  | 0.76  | 0.827 | null        |
| sarc-UK | 0.783 | 0.786 | 0.034 | 0.73  | 0.842 | null        |
| score   | 0.854 | 0.85  | 0.014 | 0.825 | 0.872 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 58.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.971   | 0.819   | 0.808   | 0.858 |
| 1    | 0.928   | 0.966   | 0.788   | 0.804   | 0.858 |
| 2    | 0.936   | 0.964   | 0.78    | 0.753   | 0.844 |
| 3    | 0.923   | 0.976   | 0.762   | 0.808   | 0.843 |
| 4    | 0.928   | 0.973   | 0.79    | 0.746   | 0.837 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.457     | 0.925    | 0.927     | 0.911  | 0.983 | 0.982 | 0.052 | 0.008 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.97     | 0.972     | 0.971  | 0.996 | 0.996 | 0.02  | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.729     | 0.97     | 0.989     | 0.98   | 0.998 | 0.999 | 0.014 | 0.007 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.311     | 0.882    | 0.849     | 0.827  | 0.963 | 0.929 | 0.072 | 0.01  |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.972    | 0.984     | 0.988  | 0.996 | 0.998 | 0.016 | 0.002 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.107     | 0.882    | 0.82      | 0.763  | 0.983 | 0.913 | 0.027 | 0.004 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.306     | 0.788    | 0.693     | 0.714  | 0.895 | 0.761 | 0.119 | 0.011 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.096     | 0.783    | 0.547     | 0.675  | 0.955 | 0.646 | 0.041 | 0.004 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.01      | 0.566    | 0.636     | 0.096  | 0.915 | 0.405 | 0.051 | 0.023 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.465     | 0.757    | 0.694     | 0.768  | 0.849 | 0.782 | 0.156 | 0.018 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.178 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.269     | 0.739    | 0.547     | 0.677  | 0.861 | 0.648 | 0.114 | 0.013 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.92  | 0.753 |
| 1-10  | UK      | 219 | 0.901 | 0.668 |
| 11-30 | AU      | 717 | 0.873 | 0.769 |
| 11-30 | UK      | 429 | 0.846 | 0.729 |
| 31-80 | AU      | 602 | 0.87  | 0.766 |
| 31-80 | UK      | 385 | 0.886 | 0.773 |
| >80   | AU      | 320 | 0.848 | 0.71  |
| >80   | UK      | 110 | 0.865 | 0.665 |
