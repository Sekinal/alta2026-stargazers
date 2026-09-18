# Validation report: `cls_e5_base_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.021 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.50 (valid 0.798, pool 0.821; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.911   | 0.961   | 0.726   | 0.735   | 0.818 | 0.008   | 0.806       |
| global+ukg    | 0.91    | 0.96    | 0.729   | 0.732   | 0.82  | 0.01    | 0.79        |
| variety+ukg   | 0.91    | 0.958   | 0.731   | 0.733   | 0.82  | 0.009   | 0.798       |
| segment+ukg   | 0.909   | 0.957   | 0.748   | 0.733   | 0.821 | 0.01    | 0.798       |
| prev+ukg      | 0.913   | 0.963   | 0.733   | 0.724   | 0.818 | 0.006   | 0.792       |
| prevtuned+ukg | 0.913   | 0.963   | 0.739   | 0.737   | 0.825 | 0.006   | 0.804       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8210**

## Expected test score for `segment+ukg`
**Point estimate 0.8210 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7891, 0.8528]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.909 | 0.911 | 0.015 | 0.886 | 0.935 | null        |
| sent-UK | 0.957 | 0.957 | 0.01  | 0.94  | 0.971 | null        |
| sarc-AU | 0.748 | 0.756 | 0.019 | 0.724 | 0.786 | null        |
| sarc-UK | 0.733 | 0.732 | 0.035 | 0.675 | 0.79  | null        |
| score   | 0.821 | 0.818 | 0.016 | 0.789 | 0.842 | 0.991       |

binding varieties: sent: AU binds 99.5%, sarc: UK binds 72.7%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.907   | 0.95    | 0.756   | 0.728   | 0.817 |
| 1    | 0.92    | 0.953   | 0.757   | 0.761   | 0.839 |
| 2    | 0.908   | 0.959   | 0.748   | 0.72    | 0.814 |
| 3    | 0.903   | 0.958   | 0.73    | 0.729   | 0.816 |
| 4    | 0.907   | 0.966   | 0.748   | 0.728   | 0.817 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.456     | 0.909    | 0.91      | 0.894  | 0.969 | 0.969 | 0.066 | 0.01  |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.957    | 0.963     | 0.955  | 0.992 | 0.993 | 0.03  | 0.024 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.749     | 0.953    | 0.967     | 0.985  | 0.991 | 0.997 | 0.028 | 0.019 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.299     | 0.86     | 0.835     | 0.781  | 0.941 | 0.895 | 0.087 | 0.013 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.743     | 0.953    | 0.979     | 0.973  | 0.991 | 0.996 | 0.027 | 0.022 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.11      | 0.862    | 0.77      | 0.74   | 0.968 | 0.869 | 0.036 | 0.015 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.358     | 0.748    | 0.605     | 0.73   | 0.838 | 0.63  | 0.155 | 0.084 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.105     | 0.733    | 0.447     | 0.606  | 0.93  | 0.509 | 0.065 | 0.073 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.005     | 0.508    | 0.4       | 0.027  | 0.867 | 0.288 | 0.097 | 0.14  |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.548     | 0.696    | 0.606     | 0.791  | 0.791 | 0.706 | 0.186 | 0.055 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.296     | 0.674    | 0.447     | 0.609  | 0.779 | 0.511 | 0.181 | 0.202 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.867 | 0.652 |
| 1-10  | UK      | 219 | 0.917 | 0.652 |
| 11-30 | AU      | 717 | 0.858 | 0.714 |
| 11-30 | UK      | 429 | 0.778 | 0.693 |
| 31-80 | AU      | 602 | 0.853 | 0.713 |
| 31-80 | UK      | 385 | 0.881 | 0.643 |
| >80   | AU      | 320 | 0.846 | 0.662 |
| >80   | UK      | 110 | 0.761 | 0.516 |
