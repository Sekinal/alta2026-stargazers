# Validation report: `cls_tw_sent_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.017 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.91 (valid 0.776, pool 0.820; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.912   | 0.96    | 0.744   | 0.738   | 0.825 | 0.008   | 0.785       |
| global+ukg    | 0.909   | 0.96    | 0.745   | 0.729   | 0.819 | 0.005   | 0.777       |
| variety+ukg   | 0.911   | 0.959   | 0.738   | 0.732   | 0.821 | 0.006   | 0.777       |
| segment+ukg   | 0.909   | 0.959   | 0.77    | 0.732   | 0.82  | 0.004   | 0.776       |
| prev+ukg      | 0.911   | 0.962   | 0.764   | 0.724   | 0.817 | 0.005   | 0.78        |
| prevtuned+ukg | 0.911   | 0.962   | 0.768   | 0.74    | 0.825 | 0.008   | 0.787       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8204**

## Expected test score for `segment+ukg`
**Point estimate 0.8204 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7864, 0.8544]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.909 | 0.911 | 0.015 | 0.886 | 0.935 | null        |
| sent-UK | 0.959 | 0.959 | 0.01  | 0.943 | 0.974 | null        |
| sarc-AU | 0.77  | 0.776 | 0.019 | 0.744 | 0.807 | null        |
| sarc-UK | 0.732 | 0.731 | 0.034 | 0.673 | 0.79  | null        |
| score   | 0.82  | 0.82  | 0.017 | 0.79  | 0.847 | 0.994       |

binding varieties: sent: AU binds 99.8%, sarc: UK binds 87.2%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.891   | 0.964   | 0.77    | 0.75    | 0.821 |
| 1    | 0.923   | 0.952   | 0.786   | 0.732   | 0.827 |
| 2    | 0.918   | 0.956   | 0.763   | 0.714   | 0.816 |
| 3    | 0.905   | 0.962   | 0.757   | 0.73    | 0.817 |
| 4    | 0.908   | 0.962   | 0.775   | 0.73    | 0.819 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.457     | 0.909    | 0.909     | 0.895  | 0.973 | 0.972 | 0.064 | 0.014 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.525     | 0.959    | 0.959     | 0.964  | 0.992 | 0.993 | 0.032 | 0.006 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.746     | 0.959    | 0.971     | 0.986  | 0.994 | 0.998 | 0.024 | 0.006 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.302     | 0.858    | 0.827     | 0.783  | 0.945 | 0.902 | 0.086 | 0.018 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.756     | 0.953    | 0.971     | 0.982  | 0.991 | 0.997 | 0.028 | 0.003 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.107     | 0.873    | 0.803     | 0.748  | 0.971 | 0.854 | 0.038 | 0.014 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.347     | 0.77     | 0.639     | 0.746  | 0.855 | 0.667 | 0.149 | 0.081 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.114     | 0.732    | 0.433     | 0.635  | 0.931 | 0.526 | 0.062 | 0.067 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.015     | 0.525    | 0.25      | 0.055  | 0.894 | 0.334 | 0.105 | 0.142 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.525     | 0.731    | 0.644     | 0.806  | 0.821 | 0.75  | 0.172 | 0.048 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.319     | 0.669    | 0.433     | 0.637  | 0.784 | 0.528 | 0.175 | 0.185 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.85  | 0.723 |
| 1-10  | UK      | 219 | 0.878 | 0.632 |
| 11-30 | AU      | 717 | 0.852 | 0.741 |
| 11-30 | UK      | 429 | 0.84  | 0.678 |
| 31-80 | AU      | 602 | 0.846 | 0.747 |
| 31-80 | UK      | 385 | 0.884 | 0.663 |
| >80   | AU      | 320 | 0.89  | 0.684 |
| >80   | UK      | 110 | 0.877 | 0.539 |
