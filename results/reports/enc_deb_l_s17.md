# Validation report: `enc_deb_l_s17`

generated 2026-09-18 11:14, code 1c9969c-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] test split: 757 rows bound to experiments/encoders/fake_test.csv (d9e22447974963d0)
  [OK  ] fold_hash matches current folds
  [OK  ] permutation canary passed: 0.6314 vs floor 0.6407
  [OK  ] per-fold score spread 0.020 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.65 (valid 0.826, pool 0.859; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.927   | 0.973   | 0.785   | 0.78    | 0.853 | 0.006   | 0.784       |
| global+ukg    | 0.927   | 0.973   | 0.789   | 0.813   | 0.858 | 0.008   | 0.83        |
| variety+ukg   | 0.928   | 0.97    | 0.79    | 0.817   | 0.859 | 0.007   | 0.827       |
| segment+ukg   | 0.925   | 0.97    | 0.793   | 0.817   | 0.859 | 0.007   | 0.826       |
| prev+ukg      | 0.929   | 0.97    | 0.797   | 0.82    | 0.863 | 0.008   | 0.838       |
| prevtuned+ukg | 0.929   | 0.97    | 0.8     | 0.818   | 0.864 | 0.007   | 0.832       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8593**

## Expected test score for `segment+ukg`
**Point estimate 0.8593 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8318, 0.8868]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.925 | 0.926 | 0.013 | 0.903 | 0.946 | null        |
| sent-UK | 0.97  | 0.971 | 0.008 | 0.956 | 0.984 | null        |
| sarc-AU | 0.793 | 0.797 | 0.021 | 0.761 | 0.83  | null        |
| sarc-UK | 0.817 | 0.818 | 0.037 | 0.756 | 0.877 | null        |
| score   | 0.859 | 0.857 | 0.014 | 0.834 | 0.879 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 30.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.917   | 0.977   | 0.829   | 0.82    | 0.868 |
| 1    | 0.92    | 0.97    | 0.777   | 0.827   | 0.848 |
| 2    | 0.925   | 0.97    | 0.797   | 0.797   | 0.861 |
| 3    | 0.933   | 0.967   | 0.777   | 0.833   | 0.855 |
| 4    | 0.931   | 0.967   | 0.784   | 0.805   | 0.858 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.462     | 0.925    | 0.923     | 0.917  | 0.977 | 0.978 | 0.065 | 0.046 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.514     | 0.97     | 0.979     | 0.964  | 0.993 | 0.994 | 0.024 | 0.01  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.736     | 0.973    | 0.985     | 0.986  | 0.998 | 0.999 | 0.017 | 0.005 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.314     | 0.881    | 0.844     | 0.831  | 0.95  | 0.913 | 0.091 | 0.075 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.739     | 0.971    | 0.991     | 0.979  | 0.993 | 0.996 | 0.02  | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.107     | 0.891    | 0.836     | 0.779  | 0.972 | 0.903 | 0.031 | 0.022 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.312     | 0.793    | 0.695     | 0.731  | 0.896 | 0.758 | 0.143 | 0.114 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.074     | 0.817    | 0.676     | 0.647  | 0.954 | 0.679 | 0.045 | 0.022 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.042     | 0.695    | 0.556     | 0.342  | 0.92  | 0.539 | 0.05  | 0.032 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.457     | 0.761    | 0.702     | 0.765  | 0.844 | 0.772 | 0.192 | 0.155 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.846 | 0.003 | 0.002 | 0.003 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.208     | 0.786    | 0.676     | 0.649  | 0.866 | 0.696 | 0.122 | 0.092 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.895 | 0.751 |
| 1-10  | UK      | 219 | 0.896 | 0.72  |
| 11-30 | AU      | 717 | 0.875 | 0.774 |
| 11-30 | UK      | 429 | 0.879 | 0.795 |
| 31-80 | AU      | 602 | 0.871 | 0.772 |
| 31-80 | UK      | 385 | 0.902 | 0.798 |
| >80   | AU      | 320 | 0.88  | 0.719 |
| >80   | UK      | 110 | 0.796 | 0.714 |
