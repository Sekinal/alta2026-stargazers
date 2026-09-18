# Validation report: `stk_v3_catn`

generated 2026-09-18 02:00, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.028 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.26 (valid 0.838, pool 0.865; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.935   | 0.977   | 0.794   | 0.784   | 0.859 | 0.011   | 0.817       |
| global+ukg    | 0.931   | 0.976   | 0.797   | 0.83    | 0.864 | 0.008   | 0.835       |
| variety+ukg   | 0.931   | 0.978   | 0.798   | 0.828   | 0.864 | 0.008   | 0.841       |
| segment+ukg   | 0.93    | 0.979   | 0.8     | 0.828   | 0.865 | 0.007   | 0.838       |
| prev+ukg      | 0.935   | 0.976   | 0.802   | 0.831   | 0.869 | 0.006   | 0.85        |
| prevtuned+ukg | 0.935   | 0.976   | 0.798   | 0.83    | 0.867 | 0.005   | 0.857       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8647**

## Expected test score for `segment+ukg`
**Point estimate 0.8647 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8397, 0.8898]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.93  | 0.931 | 0.013 | 0.911 | 0.951 | null        |
| sent-UK | 0.979 | 0.979 | 0.007 | 0.969 | 0.99  | null        |
| sarc-AU | 0.8   | 0.804 | 0.02  | 0.771 | 0.836 | null        |
| sarc-UK | 0.828 | 0.83  | 0.035 | 0.769 | 0.884 | null        |
| score   | 0.865 | 0.864 | 0.013 | 0.842 | 0.883 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 24.6%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.917   | 0.983   | 0.835   | 0.843   | 0.876 |
| 1    | 0.927   | 0.978   | 0.791   | 0.867   | 0.859 |
| 2    | 0.938   | 0.98    | 0.793   | 0.779   | 0.858 |
| 3    | 0.935   | 0.976   | 0.784   | 0.856   | 0.859 |
| 4    | 0.933   | 0.978   | 0.794   | 0.79    | 0.861 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.469     | 0.93     | 0.92      | 0.93   | 0.984 | 0.984 | 0.05  | 0.015 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.979    | 0.983     | 0.977  | 0.997 | 0.997 | 0.018 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.739     | 0.974    | 0.984     | 0.989  | 0.999 | 0.999 | 0.013 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.325     | 0.89     | 0.843     | 0.858  | 0.966 | 0.936 | 0.069 | 0.035 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.977    | 0.986     | 0.991  | 0.996 | 0.999 | 0.015 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.099     | 0.927    | 0.938     | 0.809  | 0.988 | 0.937 | 0.024 | 0.008 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.319     | 0.8      | 0.698     | 0.749  | 0.908 | 0.79  | 0.117 | 0.046 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.076     | 0.828    | 0.689     | 0.675  | 0.968 | 0.747 | 0.036 | 0.017 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.035     | 0.72     | 0.684     | 0.356  | 0.936 | 0.532 | 0.046 | 0.024 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.471     | 0.764    | 0.698     | 0.784  | 0.863 | 0.805 | 0.155 | 0.062 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.969 | 0.016 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.213     | 0.798    | 0.689     | 0.677  | 0.897 | 0.75  | 0.101 | 0.047 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.903 | 0.757 |
| 1-10  | UK      | 219 | 0.934 | 0.755 |
| 11-30 | AU      | 717 | 0.887 | 0.785 |
| 11-30 | UK      | 429 | 0.903 | 0.796 |
| 31-80 | AU      | 602 | 0.879 | 0.778 |
| 31-80 | UK      | 385 | 0.93  | 0.815 |
| >80   | AU      | 320 | 0.879 | 0.695 |
| >80   | UK      | 110 | 0.924 | 0.695 |
