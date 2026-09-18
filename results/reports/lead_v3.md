# Validation report: `lead_v3`

generated 2026-09-18 02:00, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.042 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.59 (valid 0.845, pool 0.864; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.934   | 0.977   | 0.8     | 0.789   | 0.862 | 0.016   | 0.824       |
| global+ukg    | 0.932   | 0.975   | 0.799   | 0.814   | 0.866 | 0.008   | 0.835       |
| variety+ukg   | 0.932   | 0.976   | 0.799   | 0.826   | 0.866 | 0.009   | 0.847       |
| segment+ukg   | 0.932   | 0.978   | 0.797   | 0.826   | 0.864 | 0.006   | 0.845       |
| prev+ukg      | 0.936   | 0.975   | 0.8     | 0.829   | 0.868 | 0.008   | 0.853       |
| prevtuned+ukg | 0.936   | 0.975   | 0.799   | 0.825   | 0.867 | 0.007   | 0.849       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8643**

## Expected test score for `segment+ukg`
**Point estimate 0.8643 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8393, 0.8893]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.932 | 0.933 | 0.012 | 0.911 | 0.951 | null        |
| sent-UK | 0.978 | 0.979 | 0.007 | 0.966 | 0.99  | null        |
| sarc-AU | 0.797 | 0.801 | 0.02  | 0.768 | 0.832 | null        |
| sarc-UK | 0.826 | 0.829 | 0.036 | 0.769 | 0.884 | null        |
| score   | 0.864 | 0.864 | 0.013 | 0.842 | 0.883 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 23.6%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.917   | 0.986   | 0.828   | 0.833   | 0.873 |
| 1    | 0.925   | 0.978   | 0.788   | 0.878   | 0.856 |
| 2    | 0.941   | 0.977   | 0.794   | 0.776   | 0.858 |
| 3    | 0.939   | 0.972   | 0.782   | 0.861   | 0.861 |
| 4    | 0.936   | 0.98    | 0.789   | 0.782   | 0.859 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.466     | 0.932    | 0.926     | 0.928  | 0.984 | 0.984 | 0.049 | 0.013 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.52      | 0.978    | 0.981     | 0.977  | 0.996 | 0.997 | 0.019 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.974    | 0.987     | 0.985  | 0.999 | 0.999 | 0.012 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.322     | 0.893    | 0.851     | 0.858  | 0.965 | 0.935 | 0.069 | 0.033 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.975    | 0.986     | 0.99   | 0.996 | 0.998 | 0.015 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.93     | 0.924     | 0.832  | 0.987 | 0.934 | 0.025 | 0.01  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.32      | 0.797    | 0.693     | 0.747  | 0.908 | 0.79  | 0.115 | 0.04  |
| sarc-UK        | 3208 | 249   | 0.078      | 0.077     | 0.826    | 0.683     | 0.675  | 0.966 | 0.743 | 0.036 | 0.009 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.039     | 0.683    | 0.548     | 0.315  | 0.935 | 0.527 | 0.044 | 0.006 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.471     | 0.765    | 0.699     | 0.785  | 0.863 | 0.805 | 0.154 | 0.053 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.766 | 0.002 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.215     | 0.796    | 0.683     | 0.677  | 0.895 | 0.745 | 0.099 | 0.025 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.906 | 0.756 |
| 1-10  | UK      | 219 | 0.936 | 0.771 |
| 11-30 | AU      | 717 | 0.891 | 0.782 |
| 11-30 | UK      | 429 | 0.913 | 0.788 |
| 31-80 | AU      | 602 | 0.877 | 0.783 |
| 31-80 | UK      | 385 | 0.93  | 0.812 |
| >80   | AU      | 320 | 0.891 | 0.7   |
| >80   | UK      | 110 | 0.924 | 0.695 |
