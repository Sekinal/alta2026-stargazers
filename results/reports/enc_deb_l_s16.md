# Validation report: `enc_deb_l_s16`

generated 2026-09-18 11:14, code 1c9969c-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] test split: 757 rows bound to experiments/encoders/fake_test.csv (d9e22447974963d0)
  [OK  ] fold_hash matches current folds
  [OK  ] permutation canary passed: 0.6314 vs floor 0.6407
  [OK  ] per-fold score spread 0.014 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.68 (valid 0.824, pool 0.858; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.924   | 0.971   | 0.791   | 0.796   | 0.857 | 0.005   | 0.786       |
| global+ukg    | 0.925   | 0.972   | 0.792   | 0.814   | 0.859 | 0.006   | 0.83        |
| variety+ukg   | 0.924   | 0.969   | 0.79    | 0.812   | 0.857 | 0.004   | 0.824       |
| segment+ukg   | 0.924   | 0.968   | 0.792   | 0.812   | 0.858 | 0.003   | 0.824       |
| prev+ukg      | 0.924   | 0.971   | 0.792   | 0.814   | 0.858 | 0.007   | 0.828       |
| prevtuned+ukg | 0.924   | 0.971   | 0.788   | 0.814   | 0.856 | 0.005   | 0.828       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8579**

## Expected test score for `segment+ukg`
**Point estimate 0.8579 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8303, 0.8854]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.924 | 0.924 | 0.013 | 0.902 | 0.946 | null        |
| sent-UK | 0.968 | 0.968 | 0.008 | 0.953 | 0.982 | null        |
| sarc-AU | 0.792 | 0.797 | 0.02  | 0.763 | 0.829 | null        |
| sarc-UK | 0.812 | 0.814 | 0.037 | 0.75  | 0.87  | null        |
| score   | 0.858 | 0.856 | 0.014 | 0.832 | 0.877 | 1.0         |

binding varieties: sent: AU binds 99.8%, sarc: UK binds 33.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.902   | 0.975   | 0.819   | 0.838   | 0.861 |
| 1    | 0.923   | 0.97    | 0.786   | 0.825   | 0.855 |
| 2    | 0.935   | 0.972   | 0.798   | 0.783   | 0.859 |
| 3    | 0.93    | 0.958   | 0.777   | 0.831   | 0.853 |
| 4    | 0.928   | 0.966   | 0.778   | 0.78    | 0.853 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.453     | 0.924    | 0.929     | 0.906  | 0.976 | 0.975 | 0.066 | 0.049 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.968    | 0.975     | 0.964  | 0.993 | 0.994 | 0.026 | 0.014 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.974    | 0.987     | 0.985  | 0.997 | 0.999 | 0.016 | 0.005 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.302     | 0.877    | 0.853     | 0.808  | 0.947 | 0.905 | 0.094 | 0.074 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.966    | 0.984     | 0.982  | 0.992 | 0.997 | 0.021 | 0.011 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.101     | 0.889    | 0.853     | 0.756  | 0.969 | 0.898 | 0.034 | 0.019 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.332     | 0.792    | 0.677     | 0.758  | 0.897 | 0.753 | 0.141 | 0.112 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.074     | 0.812    | 0.668     | 0.639  | 0.956 | 0.677 | 0.044 | 0.02  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.031     | 0.682    | 0.636     | 0.288  | 0.924 | 0.511 | 0.05  | 0.03  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.495     | 0.755    | 0.678     | 0.799  | 0.846 | 0.768 | 0.19  | 0.156 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.983 | 0.028 | 0.002 | 0.002 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.208     | 0.78     | 0.668     | 0.641  | 0.871 | 0.693 | 0.12  | 0.09  |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.892 | 0.749 |
| 1-10  | UK      | 219 | 0.915 | 0.75  |
| 11-30 | AU      | 717 | 0.869 | 0.772 |
| 11-30 | UK      | 429 | 0.872 | 0.77  |
| 31-80 | AU      | 602 | 0.868 | 0.774 |
| 31-80 | UK      | 385 | 0.877 | 0.798 |
| >80   | AU      | 320 | 0.867 | 0.681 |
| >80   | UK      | 110 | 0.796 | 0.75  |
