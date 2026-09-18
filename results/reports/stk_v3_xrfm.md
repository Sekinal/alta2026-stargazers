# Validation report: `stk_v3_xrfm`

generated 2026-09-18 02:00, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.014 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.76 (valid 0.844, pool 0.865; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.934   | 0.977   | 0.798   | 0.799   | 0.866 | 0.005   | 0.822       |
| global+ukg    | 0.934   | 0.977   | 0.797   | 0.824   | 0.866 | 0.003   | 0.842       |
| variety+ukg   | 0.934   | 0.977   | 0.797   | 0.829   | 0.866 | 0.005   | 0.848       |
| segment+ukg   | 0.933   | 0.979   | 0.796   | 0.829   | 0.865 | 0.008   | 0.844       |
| prev+ukg      | 0.937   | 0.977   | 0.802   | 0.827   | 0.869 | 0.006   | 0.857       |
| prevtuned+ukg | 0.937   | 0.977   | 0.802   | 0.831   | 0.87  | 0.007   | 0.852       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8645**

## Expected test score for `segment+ukg`
**Point estimate 0.8645 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8393, 0.8898]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.933 | 0.934 | 0.012 | 0.913 | 0.954 | null        |
| sent-UK | 0.979 | 0.979 | 0.007 | 0.966 | 0.99  | null        |
| sarc-AU | 0.796 | 0.8   | 0.021 | 0.766 | 0.835 | null        |
| sarc-UK | 0.829 | 0.832 | 0.036 | 0.772 | 0.892 | null        |
| score   | 0.865 | 0.864 | 0.013 | 0.843 | 0.884 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 20.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.924   | 0.989   | 0.831   | 0.834   | 0.878 |
| 1    | 0.93    | 0.977   | 0.793   | 0.863   | 0.861 |
| 2    | 0.939   | 0.98    | 0.787   | 0.805   | 0.863 |
| 3    | 0.933   | 0.973   | 0.778   | 0.849   | 0.855 |
| 4    | 0.939   | 0.975   | 0.791   | 0.79    | 0.865 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.466     | 0.933    | 0.927     | 0.93   | 0.984 | 0.983 | 0.049 | 0.008 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.979    | 0.98      | 0.979  | 0.996 | 0.997 | 0.019 | 0.006 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.738     | 0.982    | 0.989     | 0.992  | 0.999 | 0.999 | 0.012 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.32      | 0.891    | 0.85      | 0.853  | 0.965 | 0.935 | 0.068 | 0.025 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.752     | 0.975    | 0.985     | 0.99   | 0.996 | 0.999 | 0.015 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.106     | 0.933    | 0.917     | 0.847  | 0.986 | 0.921 | 0.025 | 0.005 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.306     | 0.796    | 0.705     | 0.725  | 0.907 | 0.784 | 0.116 | 0.046 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.074     | 0.829    | 0.7       | 0.667  | 0.963 | 0.724 | 0.037 | 0.015 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.033     | 0.667    | 0.556     | 0.274  | 0.933 | 0.534 | 0.046 | 0.028 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.452     | 0.767    | 0.71      | 0.765  | 0.862 | 0.799 | 0.154 | 0.057 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.766 | 0.002 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.207     | 0.8      | 0.7       | 0.669  | 0.885 | 0.727 | 0.102 | 0.046 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.909 | 0.762 |
| 1-10  | UK      | 219 | 0.943 | 0.805 |
| 11-30 | AU      | 717 | 0.887 | 0.792 |
| 11-30 | UK      | 429 | 0.923 | 0.793 |
| 31-80 | AU      | 602 | 0.874 | 0.78  |
| 31-80 | UK      | 385 | 0.93  | 0.798 |
| >80   | AU      | 320 | 0.892 | 0.683 |
| >80   | UK      | 110 | 0.893 | 0.695 |
