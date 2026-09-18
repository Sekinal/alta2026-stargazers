# Validation report: `enc_deb_l_ls05_s14`

generated 2026-09-18 03:21, code ae44cce-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.013 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -3.67 (valid 0.813, pool 0.861; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.928   | 0.973   | 0.789   | 0.792   | 0.858 | 0.005   | 0.8         |
| global+ukg    | 0.928   | 0.973   | 0.788   | 0.806   | 0.858 | 0.009   | 0.813       |
| variety+ukg   | 0.927   | 0.972   | 0.789   | 0.803   | 0.858 | 0.008   | 0.813       |
| segment+ukg   | 0.927   | 0.973   | 0.795   | 0.803   | 0.861 | 0.007   | 0.813       |
| prev+ukg      | 0.925   | 0.972   | 0.79    | 0.8     | 0.857 | 0.007   | 0.822       |
| prevtuned+ukg | 0.925   | 0.972   | 0.792   | 0.81    | 0.859 | 0.008   | 0.845       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8606**

## Expected test score for `segment+ukg`
**Point estimate 0.8606 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8322, 0.8890]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.927 | 0.927 | 0.013 | 0.905 | 0.948 | null        |
| sent-UK | 0.973 | 0.973 | 0.008 | 0.959 | 0.984 | null        |
| sarc-AU | 0.795 | 0.8   | 0.02  | 0.766 | 0.831 | null        |
| sarc-UK | 0.803 | 0.806 | 0.037 | 0.741 | 0.864 | null        |
| score   | 0.861 | 0.856 | 0.014 | 0.831 | 0.878 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 43.3%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.904   | 0.974   | 0.831   | 0.819   | 0.861 |
| 1    | 0.935   | 0.975   | 0.792   | 0.82    | 0.864 |
| 2    | 0.928   | 0.98    | 0.788   | 0.765   | 0.846 |
| 3    | 0.933   | 0.967   | 0.781   | 0.835   | 0.857 |
| 4    | 0.933   | 0.969   | 0.781   | 0.774   | 0.853 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.458     | 0.927    | 0.927     | 0.915  | 0.977 | 0.976 | 0.064 | 0.045 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.51      | 0.973    | 0.986     | 0.962  | 0.993 | 0.995 | 0.024 | 0.009 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.736     | 0.975    | 0.986     | 0.987  | 0.996 | 0.998 | 0.016 | 0.005 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.309     | 0.882    | 0.852     | 0.825  | 0.951 | 0.909 | 0.09  | 0.077 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.742     | 0.973    | 0.99      | 0.982  | 0.994 | 0.998 | 0.019 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.09      | 0.895    | 0.922     | 0.725  | 0.963 | 0.892 | 0.035 | 0.018 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.331     | 0.795    | 0.681     | 0.76   | 0.894 | 0.751 | 0.134 | 0.085 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.078     | 0.803    | 0.636     | 0.639  | 0.917 | 0.662 | 0.043 | 0.017 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.029     | 0.646    | 0.548     | 0.233  | 0.896 | 0.483 | 0.051 | 0.029 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.494     | 0.761    | 0.685     | 0.806  | 0.848 | 0.767 | 0.178 | 0.128 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.332 | 0.001 | 0.002 | 0.021 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.219     | 0.769    | 0.636     | 0.641  | 0.863 | 0.69  | 0.116 | 0.076 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.884 | 0.754 |
| 1-10  | UK      | 219 | 0.907 | 0.711 |
| 11-30 | AU      | 717 | 0.884 | 0.782 |
| 11-30 | UK      | 429 | 0.889 | 0.761 |
| 31-80 | AU      | 602 | 0.869 | 0.773 |
| 31-80 | UK      | 385 | 0.903 | 0.805 |
| >80   | AU      | 320 | 0.881 | 0.696 |
| >80   | UK      | 110 | 0.736 | 0.713 |
