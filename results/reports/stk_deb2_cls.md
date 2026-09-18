# Validation report: `stk_deb2_cls`

generated 2026-09-18 01:00, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.032 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.38 (valid 0.835, pool 0.865; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.931   | 0.974   | 0.797   | 0.815   | 0.864 | 0.014   | 0.822       |
| global+ukg    | 0.93    | 0.974   | 0.802   | 0.816   | 0.866 | 0.017   | 0.826       |
| variety+ukg   | 0.929   | 0.972   | 0.802   | 0.811   | 0.866 | 0.012   | 0.838       |
| segment+ukg   | 0.928   | 0.973   | 0.803   | 0.811   | 0.865 | 0.01    | 0.835       |
| prev+ukg      | 0.931   | 0.976   | 0.801   | 0.818   | 0.866 | 0.013   | 0.836       |
| prevtuned+ukg | 0.931   | 0.976   | 0.799   | 0.809   | 0.865 | 0.01    | 0.843       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8654**

## Expected test score for `segment+ukg`
**Point estimate 0.8654 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8385, 0.8923]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.928 | 0.929 | 0.013 | 0.908 | 0.949 | null        |
| sent-UK | 0.973 | 0.973 | 0.008 | 0.961 | 0.984 | null        |
| sarc-AU | 0.803 | 0.807 | 0.019 | 0.775 | 0.838 | null        |
| sarc-UK | 0.811 | 0.813 | 0.035 | 0.755 | 0.871 | null        |
| score   | 0.865 | 0.861 | 0.014 | 0.838 | 0.882 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 44.4%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.911   | 0.977   | 0.844   | 0.823   | 0.867 |
| 1    | 0.93    | 0.972   | 0.801   | 0.855   | 0.865 |
| 2    | 0.93    | 0.974   | 0.794   | 0.765   | 0.847 |
| 3    | 0.931   | 0.97    | 0.784   | 0.847   | 0.858 |
| 4    | 0.938   | 0.973   | 0.791   | 0.75    | 0.844 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.468     | 0.928    | 0.92      | 0.926  | 0.984 | 0.983 | 0.049 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.973    | 0.979     | 0.969  | 0.996 | 0.996 | 0.02  | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.974    | 0.987     | 0.985  | 0.998 | 0.999 | 0.014 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.325     | 0.886    | 0.839     | 0.853  | 0.965 | 0.934 | 0.068 | 0.015 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.746     | 0.975    | 0.989     | 0.986  | 0.996 | 0.998 | 0.016 | 0.003 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.894    | 0.856     | 0.771  | 0.982 | 0.919 | 0.026 | 0.006 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.331     | 0.803    | 0.691     | 0.772  | 0.904 | 0.778 | 0.114 | 0.014 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.084     | 0.811    | 0.628     | 0.679  | 0.962 | 0.726 | 0.036 | 0.004 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.044     | 0.726    | 0.617     | 0.397  | 0.928 | 0.512 | 0.045 | 0.014 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.486     | 0.768    | 0.695     | 0.805  | 0.859 | 0.794 | 0.151 | 0.024 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.214 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.235     | 0.776    | 0.628     | 0.681  | 0.887 | 0.729 | 0.101 | 0.017 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.901 | 0.754 |
| 1-10  | UK      | 219 | 0.909 | 0.72  |
| 11-30 | AU      | 717 | 0.884 | 0.785 |
| 11-30 | UK      | 429 | 0.854 | 0.761 |
| 31-80 | AU      | 602 | 0.877 | 0.785 |
| 31-80 | UK      | 385 | 0.899 | 0.812 |
| >80   | AU      | 320 | 0.871 | 0.708 |
| >80   | UK      | 110 | 0.893 | 0.736 |
