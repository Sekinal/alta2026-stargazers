# Validation report: `llm_q35_9b_probe`

generated 2026-09-18 04:18, code 3cf922e-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [OK  ] permutation canary passed (0.6422 vs floor 0.6407)
  [OK  ] per-fold score spread 0.021 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.16 (valid 0.855, pool 0.853; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.92    | 0.974   | 0.789   | 0.795   | 0.854 | 0.01    | 0.846       |
| global+ukg    | 0.917   | 0.972   | 0.791   | 0.801   | 0.854 | 0.01    | 0.854       |
| variety+ukg   | 0.917   | 0.972   | 0.791   | 0.806   | 0.854 | 0.01    | 0.854       |
| segment+ukg   | 0.916   | 0.973   | 0.789   | 0.806   | 0.853 | 0.009   | 0.855       |
| prev+ukg      | 0.921   | 0.973   | 0.787   | 0.809   | 0.854 | 0.013   | 0.838       |
| prevtuned+ukg | 0.921   | 0.973   | 0.787   | 0.801   | 0.854 | 0.011   | 0.856       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8529**

## Expected test score for `segment+ukg`
**Point estimate 0.8529 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8275, 0.8783]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.916 | 0.917 | 0.014 | 0.894 | 0.938 | null        |
| sent-UK | 0.973 | 0.972 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.789 | 0.795 | 0.02  | 0.763 | 0.827 | null        |
| sarc-UK | 0.806 | 0.808 | 0.033 | 0.755 | 0.864 | null        |
| score   | 0.853 | 0.851 | 0.013 | 0.83  | 0.872 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 37.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.899   | 0.977   | 0.824   | 0.823   | 0.861 |
| 1    | 0.924   | 0.969   | 0.794   | 0.832   | 0.859 |
| 2    | 0.928   | 0.967   | 0.789   | 0.788   | 0.858 |
| 3    | 0.917   | 0.973   | 0.762   | 0.784   | 0.839 |
| 4    | 0.915   | 0.976   | 0.777   | 0.805   | 0.846 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.46      | 0.916    | 0.915     | 0.906  | 0.979 | 0.979 | 0.055 | 0.01  |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.973    | 0.978     | 0.97   | 0.995 | 0.995 | 0.02  | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.736     | 0.97     | 0.984     | 0.985  | 0.998 | 0.999 | 0.016 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.312     | 0.867    | 0.827     | 0.808  | 0.956 | 0.923 | 0.076 | 0.023 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.969    | 0.985     | 0.984  | 0.994 | 0.997 | 0.017 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.104     | 0.91     | 0.882     | 0.802  | 0.982 | 0.92  | 0.027 | 0.008 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.331     | 0.789    | 0.674     | 0.753  | 0.899 | 0.773 | 0.118 | 0.026 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.093     | 0.806    | 0.592     | 0.711  | 0.962 | 0.668 | 0.039 | 0.003 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.023     | 0.604    | 0.48      | 0.164  | 0.924 | 0.492 | 0.046 | 0.012 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.497     | 0.756    | 0.679     | 0.804  | 0.851 | 0.79  | 0.156 | 0.034 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.998 | 0.167 | 0.0   | 0.0   |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.262     | 0.768    | 0.592     | 0.714  | 0.878 | 0.67  | 0.108 | 0.018 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.878 | 0.755 |
| 1-10  | UK      | 219 | 0.924 | 0.747 |
| 11-30 | AU      | 717 | 0.865 | 0.763 |
| 11-30 | UK      | 429 | 0.892 | 0.755 |
| 31-80 | AU      | 602 | 0.854 | 0.769 |
| 31-80 | UK      | 385 | 0.899 | 0.779 |
| >80   | AU      | 320 | 0.858 | 0.714 |
| >80   | UK      | 110 | 0.893 | 0.73  |
