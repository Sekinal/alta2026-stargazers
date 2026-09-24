# Validation report: `cls_subreddit_prior`

generated 2026-09-18 12:39, code b83c698-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.002 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.08 (valid 0.656, pool 0.655; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.689   | 0.791   | 0.67    | 0.48    | 0.584 | 0.001   | 0.585       |
| global+ukg    | 0.688   | 0.791   | 0.598   | 0.619   | 0.643 | 0.026   | 0.648       |
| variety+ukg   | 0.688   | 0.791   | 0.67    | 0.627   | 0.657 | 0.015   | 0.656       |
| segment+ukg   | 0.688   | 0.73    | 0.622   | 0.627   | 0.655 | 0.077   | 0.656       |
| prev+ukg      | 0.635   | 0.688   | 0.653   | 0.617   | 0.626 | 0.018   | 0.461       |
| prevtuned+ukg | 0.659   | 0.749   | 0.654   | 0.624   | 0.642 | 0.019   | 0.6         |

**Headline policy: `segment+ukg`** (default policy); **score 0.6547**

## Expected test score for `segment+ukg`
**Point estimate 0.6547 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.6291, 0.6803]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.688 | 0.689 | 0.01  | 0.673 | 0.706 | null        |
| sent-UK | 0.73  | 0.731 | 0.017 | 0.702 | 0.758 | null        |
| sarc-AU | 0.622 | 0.625 | 0.027 | 0.582 | 0.668 | null        |
| sarc-UK | 0.627 | 0.629 | 0.031 | 0.578 | 0.679 | null        |
| score   | 0.655 | 0.65  | 0.013 | 0.628 | 0.671 | 0.0         |

binding varieties: sent: AU binds 97.8%, sarc: UK binds 45.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.689   | 0.787   | 0.653   | 0.606   | 0.647 |
| 1    | 0.671   | 0.791   | 0.66    | 0.594   | 0.632 |
| 2    | 0.688   | 0.78    | 0.446   | 0.658   | 0.567 |
| 3    | 0.69    | 0.792   | 0.669   | 0.629   | 0.66  |
| 4    | 0.69    | 0.323   | 0.682   | 0.625   | 0.474 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.391     | 0.688    | 0.703     | 0.592  | 0.758 | 0.681 | 0.191 | 0.018 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.73     | 0.744     | 0.738  | 0.796 | 0.739 | 0.157 | 0.014 |
| sent:AU-google | 1076 | 791   | 0.735      | 1.0       | 0.424    | 0.735     | 1.0    | 0.496 | 0.733 | 0.195 | 0.024 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.064     | 0.471    | 0.434     | 0.088  | 0.684 | 0.436 | 0.19  | 0.019 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.8       | 0.497    | 0.748     | 0.8    | 0.498 | 0.748 | 0.188 | 0.015 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.008     | 0.475    | 0.111     | 0.008  | 0.603 | 0.159 | 0.099 | 0.034 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.285     | 0.622    | 0.473     | 0.454  | 0.751 | 0.513 | 0.17  | 0.013 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.155     | 0.627    | 0.256     | 0.51   | 0.863 | 0.254 | 0.06  | 0.005 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.2       | 0.479    | 0.065     | 0.192  | 0.487 | 0.066 | 0.063 | 0.021 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.33      | 0.627    | 0.606     | 0.477  | 0.621 | 0.529 | 0.227 | 0.016 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.435     | 0.511    | 0.256     | 0.512  | 0.559 | 0.255 | 0.168 | 0.032 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.411 | 0.589 |
| 1-10  | UK      | 219 | 0.427 | 0.509 |
| 11-30 | AU      | 717 | 0.493 | 0.585 |
| 11-30 | UK      | 429 | 0.507 | 0.503 |
| 31-80 | AU      | 602 | 0.483 | 0.68  |
| 31-80 | UK      | 385 | 0.473 | 0.49  |
| >80   | AU      | 320 | 0.449 | 0.616 |
| >80   | UK      | 110 | 0.481 | 0.467 |
