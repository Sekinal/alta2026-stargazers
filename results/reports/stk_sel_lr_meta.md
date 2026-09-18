# Validation report: `stk_sel_lr_meta`

generated 2026-09-18 00:34, code 94275f4-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.024 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.01 (valid 0.857, pool 0.857; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.925   | 0.973   | 0.784   | 0.764   | 0.844 | 0.011   | 0.822       |
| global+ukg    | 0.929   | 0.974   | 0.789   | 0.786   | 0.857 | 0.008   | 0.849       |
| variety+ukg   | 0.928   | 0.973   | 0.788   | 0.797   | 0.858 | 0.014   | 0.858       |
| segment+ukg   | 0.927   | 0.974   | 0.787   | 0.797   | 0.857 | 0.013   | 0.857       |
| prev+ukg      | 0.928   | 0.972   | 0.787   | 0.785   | 0.857 | 0.008   | 0.843       |
| prevtuned+ukg | 0.928   | 0.972   | 0.788   | 0.79    | 0.858 | 0.01    | 0.842       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8570**

## Expected test score for `segment+ukg`
**Point estimate 0.8570 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8312, 0.8828]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.927 | 0.928 | 0.013 | 0.905 | 0.949 | null        |
| sent-UK | 0.974 | 0.974 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.787 | 0.794 | 0.02  | 0.76  | 0.825 | null        |
| sarc-UK | 0.797 | 0.8   | 0.033 | 0.744 | 0.855 | null        |
| score   | 0.857 | 0.855 | 0.013 | 0.833 | 0.875 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 44.2%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.914   | 0.98    | 0.818   | 0.825   | 0.866 |
| 1    | 0.936   | 0.967   | 0.792   | 0.804   | 0.864 |
| 2    | 0.935   | 0.964   | 0.776   | 0.788   | 0.855 |
| 3    | 0.928   | 0.983   | 0.757   | 0.815   | 0.843 |
| 4    | 0.922   | 0.976   | 0.789   | 0.752   | 0.837 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.467     | 0.927    | 0.919     | 0.925  | 0.982 | 0.981 | 0.053 | 0.009 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.974    | 0.976     | 0.975  | 0.995 | 0.996 | 0.021 | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.73      | 0.973    | 0.989     | 0.982  | 0.998 | 0.999 | 0.014 | 0.007 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.326     | 0.885    | 0.836     | 0.853  | 0.96  | 0.925 | 0.074 | 0.013 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.754     | 0.972    | 0.982     | 0.99   | 0.995 | 0.998 | 0.017 | 0.003 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.912    | 0.89      | 0.802  | 0.98  | 0.906 | 0.028 | 0.005 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.327     | 0.787    | 0.674     | 0.743  | 0.893 | 0.755 | 0.121 | 0.014 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.098     | 0.797    | 0.565     | 0.711  | 0.952 | 0.633 | 0.042 | 0.004 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.017     | 0.558    | 0.389     | 0.096  | 0.91  | 0.396 | 0.051 | 0.021 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.494     | 0.756    | 0.679     | 0.799  | 0.846 | 0.776 | 0.158 | 0.017 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.147 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.274     | 0.756    | 0.565     | 0.714  | 0.853 | 0.635 | 0.116 | 0.019 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.915 | 0.748 |
| 1-10  | UK      | 219 | 0.928 | 0.702 |
| 11-30 | AU      | 717 | 0.882 | 0.768 |
| 11-30 | UK      | 429 | 0.908 | 0.737 |
| 31-80 | AU      | 602 | 0.883 | 0.766 |
| 31-80 | UK      | 385 | 0.908 | 0.797 |
| >80   | AU      | 320 | 0.831 | 0.708 |
| >80   | UK      | 110 | 0.796 | 0.678 |
