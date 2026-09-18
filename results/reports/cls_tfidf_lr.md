# Validation report: `cls_tfidf_lr`

generated 2026-09-18 01:06, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] test split: 757 rows bound to /home/ieqr/Desktop/research/ALTA/experiments/classical/cache/fake_test.csv (d9e22447974963d0)
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.026 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.17 (valid 0.737, pool 0.753; > 2 suggests valid rows leaked into training)
  [WARN] sent AU-reddit: P(p>=0.5) test 0.320 vs OOF 0.234; prefer a prevalence policy
  [WARN] sent UK-reddit: P(p>=0.5) test 0.116 vs OOF 0.040; prefer a prevalence policy
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.845   | 0.918   | 0.692   | 0.655   | 0.75  | 0.011   | 0.747       |
| global+ukg    | 0.843   | 0.919   | 0.699   | 0.656   | 0.749 | 0.008   | 0.735       |
| variety+ukg   | 0.843   | 0.916   | 0.69    | 0.656   | 0.749 | 0.007   | 0.735       |
| segment+ukg   | 0.85    | 0.921   | 0.691   | 0.656   | 0.753 | 0.009   | 0.737       |
| prev+ukg      | 0.843   | 0.921   | 0.682   | 0.646   | 0.744 | 0.007   | 0.742       |
| prevtuned+ukg | 0.844   | 0.921   | 0.682   | 0.657   | 0.751 | 0.004   | 0.748       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7530**

## Expected test score for `segment+ukg`
**Point estimate 0.7530 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7226, 0.7835]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.85  | 0.851 | 0.017 | 0.823 | 0.88  | null        |
| sent-UK | 0.921 | 0.921 | 0.013 | 0.899 | 0.94  | null        |
| sarc-AU | 0.691 | 0.698 | 0.021 | 0.664 | 0.733 | null        |
| sarc-UK | 0.656 | 0.657 | 0.027 | 0.612 | 0.7   | null        |
| score   | 0.753 | 0.753 | 0.015 | 0.726 | 0.778 | 0.086       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 88.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.836   | 0.915   | 0.688   | 0.659   | 0.748 |
| 1    | 0.844   | 0.924   | 0.69    | 0.673   | 0.759 |
| 2    | 0.851   | 0.92    | 0.7     | 0.628   | 0.74  |
| 3    | 0.853   | 0.926   | 0.672   | 0.677   | 0.762 |
| 4    | 0.865   | 0.919   | 0.702   | 0.642   | 0.754 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.42      | 0.85     | 0.877     | 0.792  | 0.922 | 0.926 | 0.11  | 0.024 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.512     | 0.921    | 0.933     | 0.914  | 0.974 | 0.98  | 0.06  | 0.012 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.718     | 0.902    | 0.957     | 0.936  | 0.972 | 0.989 | 0.059 | 0.036 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.259     | 0.774    | 0.758     | 0.616  | 0.86  | 0.775 | 0.137 | 0.044 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.898    | 0.947     | 0.95   | 0.975 | 0.991 | 0.056 | 0.028 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.08      | 0.764    | 0.696     | 0.489  | 0.892 | 0.648 | 0.066 | 0.038 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.354     | 0.691    | 0.536     | 0.639  | 0.803 | 0.572 | 0.174 | 0.108 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.18      | 0.656    | 0.287     | 0.667  | 0.889 | 0.328 | 0.094 | 0.118 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.008     | 0.531    | 0.444     | 0.055  | 0.838 | 0.281 | 0.058 | 0.054 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.539     | 0.619    | 0.537     | 0.69   | 0.698 | 0.586 | 0.237 | 0.143 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.88  | 0.004 | 0.001 | 0.011 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.507     | 0.531    | 0.287     | 0.669  | 0.638 | 0.328 | 0.262 | 0.311 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.761 | 0.605 |
| 1-10  | UK      | 219 | 0.829 | 0.378 |
| 11-30 | AU      | 717 | 0.756 | 0.62  |
| 11-30 | UK      | 429 | 0.677 | 0.525 |
| 31-80 | AU      | 602 | 0.785 | 0.623 |
| 31-80 | UK      | 385 | 0.779 | 0.55  |
| >80   | AU      | 320 | 0.786 | 0.612 |
| >80   | UK      | 110 | 0.481 | 0.442 |
