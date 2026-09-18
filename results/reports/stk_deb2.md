# Validation report: `stk_deb2`

generated 2026-09-18 00:58, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.030 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -3.37 (valid 0.821, pool 0.865; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.929   | 0.972   | 0.787   | 0.794   | 0.858 | 0.01    | 0.812       |
| global+ukg    | 0.927   | 0.971   | 0.804   | 0.811   | 0.865 | 0.013   | 0.833       |
| variety+ukg   | 0.927   | 0.969   | 0.804   | 0.805   | 0.865 | 0.012   | 0.823       |
| segment+ukg   | 0.928   | 0.971   | 0.803   | 0.805   | 0.865 | 0.012   | 0.821       |
| prev+ukg      | 0.928   | 0.97    | 0.801   | 0.818   | 0.865 | 0.013   | 0.843       |
| prevtuned+ukg | 0.928   | 0.97    | 0.799   | 0.81    | 0.864 | 0.013   | 0.827       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8651**

## Expected test score for `segment+ukg`
**Point estimate 0.8651 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8373, 0.8930]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.928 | 0.929 | 0.013 | 0.908 | 0.949 | null        |
| sent-UK | 0.971 | 0.971 | 0.008 | 0.956 | 0.984 | null        |
| sarc-AU | 0.803 | 0.807 | 0.019 | 0.775 | 0.837 | null        |
| sarc-UK | 0.805 | 0.808 | 0.036 | 0.749 | 0.867 | null        |
| score   | 0.865 | 0.86  | 0.014 | 0.836 | 0.881 | 1.0         |

binding varieties: sent: AU binds 99.8%, sarc: UK binds 48.4%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.909   | 0.971   | 0.842   | 0.831   | 0.87  |
| 1    | 0.932   | 0.973   | 0.795   | 0.838   | 0.864 |
| 2    | 0.938   | 0.973   | 0.788   | 0.746   | 0.842 |
| 3    | 0.931   | 0.967   | 0.8     | 0.84    | 0.866 |
| 4    | 0.928   | 0.969   | 0.787   | 0.771   | 0.85  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.469     | 0.928    | 0.919     | 0.927  | 0.979 | 0.979 | 0.062 | 0.043 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.971    | 0.973     | 0.971  | 0.994 | 0.995 | 0.025 | 0.01  |
| sent:AU-google | 1076 | 791   | 0.735      | 0.737     | 0.978    | 0.987     | 0.99   | 0.998 | 0.999 | 0.014 | 0.008 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.324     | 0.883    | 0.835     | 0.848  | 0.954 | 0.916 | 0.087 | 0.077 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.755     | 0.97     | 0.98      | 0.99   | 0.994 | 0.997 | 0.02  | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.1       | 0.893    | 0.868     | 0.756  | 0.972 | 0.906 | 0.033 | 0.02  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.321     | 0.803    | 0.7       | 0.757  | 0.899 | 0.76  | 0.136 | 0.106 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.082     | 0.805    | 0.626     | 0.659  | 0.957 | 0.702 | 0.044 | 0.02  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.034     | 0.655    | 0.514     | 0.26   | 0.923 | 0.547 | 0.05  | 0.035 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.475     | 0.774    | 0.707     | 0.8    | 0.85  | 0.774 | 0.183 | 0.144 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.857 | 0.003 | 0.002 | 0.003 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.229     | 0.77     | 0.626     | 0.661  | 0.874 | 0.715 | 0.119 | 0.092 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.892 | 0.762 |
| 1-10  | UK      | 219 | 0.887 | 0.706 |
| 11-30 | AU      | 717 | 0.88  | 0.787 |
| 11-30 | UK      | 429 | 0.878 | 0.762 |
| 31-80 | AU      | 602 | 0.87  | 0.791 |
| 31-80 | UK      | 385 | 0.899 | 0.801 |
| >80   | AU      | 320 | 0.887 | 0.726 |
| >80   | UK      | 110 | 0.893 | 0.75  |
