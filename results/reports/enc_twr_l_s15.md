# Validation report: `enc_twr_l_s15`

generated 2026-09-18 02:04, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.035 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.66 (valid 0.819, pool 0.855; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.917   | 0.969   | 0.791   | 0.78    | 0.849 | 0.014   | 0.774       |
| global+ukg    | 0.917   | 0.97    | 0.803   | 0.797   | 0.857 | 0.012   | 0.816       |
| variety+ukg   | 0.917   | 0.967   | 0.807   | 0.797   | 0.857 | 0.012   | 0.816       |
| segment+ukg   | 0.913   | 0.967   | 0.804   | 0.797   | 0.855 | 0.01    | 0.819       |
| prev+ukg      | 0.918   | 0.969   | 0.794   | 0.8     | 0.856 | 0.009   | 0.82        |
| prevtuned+ukg | 0.918   | 0.969   | 0.804   | 0.796   | 0.857 | 0.007   | 0.824       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8548**

## Expected test score for `segment+ukg`
**Point estimate 0.8548 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8237, 0.8859]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.913 | 0.914 | 0.014 | 0.891 | 0.935 | null        |
| sent-UK | 0.967 | 0.967 | 0.009 | 0.951 | 0.982 | null        |
| sarc-AU | 0.804 | 0.809 | 0.019 | 0.776 | 0.84  | null        |
| sarc-UK | 0.797 | 0.798 | 0.037 | 0.737 | 0.857 | null        |
| score   | 0.855 | 0.85  | 0.016 | 0.823 | 0.874 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 60.2%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.892   | 0.978   | 0.84    | 0.845   | 0.866 |
| 1    | 0.925   | 0.966   | 0.804   | 0.794   | 0.86  |
| 2    | 0.915   | 0.963   | 0.807   | 0.794   | 0.854 |
| 3    | 0.911   | 0.959   | 0.783   | 0.784   | 0.847 |
| 4    | 0.92    | 0.967   | 0.786   | 0.763   | 0.842 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.445     | 0.913    | 0.924     | 0.886  | 0.976 | 0.974 | 0.073 | 0.057 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.967    | 0.971     | 0.965  | 0.994 | 0.994 | 0.027 | 0.01  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.744     | 0.971    | 0.979     | 0.991  | 0.992 | 0.996 | 0.019 | 0.013 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.285     | 0.856    | 0.848     | 0.756  | 0.951 | 0.909 | 0.101 | 0.083 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.961    | 0.979     | 0.983  | 0.993 | 0.997 | 0.023 | 0.009 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.1       | 0.893    | 0.868     | 0.756  | 0.973 | 0.909 | 0.034 | 0.02  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.328     | 0.804    | 0.696     | 0.769  | 0.901 | 0.769 | 0.139 | 0.118 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.083     | 0.797    | 0.608     | 0.647  | 0.956 | 0.633 | 0.047 | 0.028 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.033     | 0.667    | 0.556     | 0.274  | 0.916 | 0.47  | 0.053 | 0.027 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.486     | 0.774    | 0.701     | 0.812  | 0.855 | 0.787 | 0.186 | 0.163 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.998 | 0.2   | 0.001 | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.232     | 0.76     | 0.608     | 0.649  | 0.863 | 0.638 | 0.131 | 0.107 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.88  | 0.744 |
| 1-10  | UK      | 219 | 0.917 | 0.734 |
| 11-30 | AU      | 717 | 0.836 | 0.788 |
| 11-30 | UK      | 429 | 0.821 | 0.761 |
| 31-80 | AU      | 602 | 0.847 | 0.802 |
| 31-80 | UK      | 385 | 0.917 | 0.763 |
| >80   | AU      | 320 | 0.867 | 0.72  |
| >80   | UK      | 110 | 0.824 | 0.697 |
