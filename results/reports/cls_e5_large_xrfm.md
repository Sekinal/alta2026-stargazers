# Validation report: `cls_e5_large_xrfm`

generated 2026-09-18 01:23, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.034 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.64 (valid 0.827, pool 0.846; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.915   | 0.959   | 0.749   | 0.78    | 0.832 | 0.014   | 0.818       |
| global+ukg    | 0.917   | 0.959   | 0.774   | 0.796   | 0.845 | 0.008   | 0.824       |
| variety+ukg   | 0.916   | 0.958   | 0.771   | 0.792   | 0.844 | 0.01    | 0.824       |
| segment+ukg   | 0.918   | 0.963   | 0.775   | 0.792   | 0.846 | 0.007   | 0.827       |
| prev+ukg      | 0.917   | 0.963   | 0.746   | 0.787   | 0.832 | 0.011   | 0.822       |
| prevtuned+ukg | 0.917   | 0.963   | 0.757   | 0.788   | 0.837 | 0.009   | 0.814       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8464**

## Expected test score for `segment+ukg`
**Point estimate 0.8464 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8203, 0.8725]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.918 | 0.919 | 0.014 | 0.895 | 0.94  | null        |
| sent-UK | 0.963 | 0.963 | 0.009 | 0.948 | 0.977 | null        |
| sarc-AU | 0.775 | 0.782 | 0.018 | 0.752 | 0.812 | null        |
| sarc-UK | 0.792 | 0.795 | 0.035 | 0.737 | 0.851 | null        |
| score   | 0.846 | 0.846 | 0.013 | 0.823 | 0.866 | 1.0         |

binding varieties: sent: AU binds 99.6%, sarc: UK binds 35.7%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.898   | 0.971   | 0.788   | 0.777   | 0.838 |
| 1    | 0.926   | 0.955   | 0.774   | 0.823   | 0.85  |
| 2    | 0.926   | 0.961   | 0.759   | 0.76    | 0.843 |
| 3    | 0.918   | 0.966   | 0.763   | 0.801   | 0.84  |
| 4    | 0.922   | 0.964   | 0.789   | 0.794   | 0.855 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.463     | 0.918    | 0.914     | 0.911  | 0.973 | 0.973 | 0.074 | 0.081 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.515     | 0.963    | 0.972     | 0.958  | 0.99  | 0.991 | 0.038 | 0.052 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.731     | 0.964    | 0.983     | 0.979  | 0.994 | 0.998 | 0.032 | 0.063 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.319     | 0.873    | 0.828     | 0.827  | 0.947 | 0.906 | 0.096 | 0.1   |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.75      | 0.963    | 0.98      | 0.983  | 0.991 | 0.997 | 0.034 | 0.047 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.09      | 0.857    | 0.845     | 0.664  | 0.969 | 0.817 | 0.047 | 0.054 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.36      | 0.775    | 0.636     | 0.771  | 0.866 | 0.699 | 0.135 | 0.027 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.093     | 0.792    | 0.569     | 0.679  | 0.942 | 0.577 | 0.047 | 0.026 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.007     | 0.52     | 0.429     | 0.041  | 0.873 | 0.28  | 0.054 | 0.024 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.549     | 0.731    | 0.638     | 0.835  | 0.807 | 0.719 | 0.178 | 0.042 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.26      | 0.751    | 0.569     | 0.681  | 0.819 | 0.579 | 0.131 | 0.078 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.882 | 0.72  |
| 1-10  | UK      | 219 | 0.911 | 0.739 |
| 11-30 | AU      | 717 | 0.862 | 0.754 |
| 11-30 | UK      | 429 | 0.815 | 0.769 |
| 31-80 | AU      | 602 | 0.875 | 0.742 |
| 31-80 | UK      | 385 | 0.807 | 0.707 |
| >80   | AU      | 320 | 0.86  | 0.662 |
| >80   | UK      | 110 | 0.824 | 0.559 |
