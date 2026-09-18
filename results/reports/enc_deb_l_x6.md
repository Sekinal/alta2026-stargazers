# Validation report: `enc_deb_l_x6`

generated 2026-09-18 03:21, code ae44cce-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.028 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.99 (valid 0.831, pool 0.867; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.932   | 0.973   | 0.793   | 0.791   | 0.862 | 0.009   | 0.812       |
| global+ukg    | 0.934   | 0.972   | 0.799   | 0.825   | 0.866 | 0.008   | 0.846       |
| variety+ukg   | 0.935   | 0.974   | 0.797   | 0.82    | 0.866 | 0.009   | 0.831       |
| segment+ukg   | 0.935   | 0.973   | 0.799   | 0.82    | 0.867 | 0.01    | 0.831       |
| prev+ukg      | 0.931   | 0.972   | 0.797   | 0.822   | 0.864 | 0.007   | 0.838       |
| prevtuned+ukg | 0.931   | 0.972   | 0.798   | 0.824   | 0.865 | 0.007   | 0.836       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8668**

## Expected test score for `segment+ukg`
**Point estimate 0.8668 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8413, 0.8923]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.935 | 0.936 | 0.012 | 0.916 | 0.954 | null        |
| sent-UK | 0.973 | 0.974 | 0.008 | 0.961 | 0.987 | null        |
| sarc-AU | 0.799 | 0.804 | 0.019 | 0.771 | 0.836 | null        |
| sarc-UK | 0.82  | 0.823 | 0.035 | 0.765 | 0.877 | null        |
| score   | 0.867 | 0.866 | 0.013 | 0.844 | 0.886 | 1.0         |

binding varieties: sent: AU binds 99.5%, sarc: UK binds 30.3%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.983   | 0.84    | 0.86    | 0.88  |
| 1    | 0.937   | 0.975   | 0.788   | 0.857   | 0.862 |
| 2    | 0.946   | 0.974   | 0.796   | 0.759   | 0.853 |
| 3    | 0.935   | 0.967   | 0.791   | 0.839   | 0.863 |
| 4    | 0.938   | 0.969   | 0.779   | 0.781   | 0.858 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.471     | 0.935    | 0.924     | 0.937  | 0.98  | 0.98  | 0.056 | 0.035 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.973    | 0.978     | 0.971  | 0.994 | 0.995 | 0.023 | 0.008 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.738     | 0.982    | 0.989     | 0.992  | 0.998 | 0.999 | 0.012 | 0.006 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.328     | 0.895    | 0.846     | 0.869  | 0.957 | 0.921 | 0.079 | 0.06  |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.747     | 0.969    | 0.985     | 0.983  | 0.994 | 0.998 | 0.018 | 0.007 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.108     | 0.92     | 0.886     | 0.832  | 0.976 | 0.911 | 0.032 | 0.018 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.33      | 0.799    | 0.688     | 0.764  | 0.902 | 0.768 | 0.13  | 0.092 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.081     | 0.82     | 0.656     | 0.683  | 0.942 | 0.693 | 0.041 | 0.021 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.024     | 0.624    | 0.538     | 0.192  | 0.915 | 0.531 | 0.049 | 0.032 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.494     | 0.768    | 0.692     | 0.813  | 0.855 | 0.783 | 0.174 | 0.125 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.817 | 0.003 | 0.002 | 0.011 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.227     | 0.788    | 0.656     | 0.685  | 0.874 | 0.713 | 0.113 | 0.079 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.898 | 0.761 |
| 1-10  | UK      | 219 | 0.936 | 0.744 |
| 11-30 | AU      | 717 | 0.893 | 0.777 |
| 11-30 | UK      | 429 | 0.882 | 0.787 |
| 31-80 | AU      | 602 | 0.893 | 0.785 |
| 31-80 | UK      | 385 | 0.932 | 0.804 |
| >80   | AU      | 320 | 0.878 | 0.72  |
| >80   | UK      | 110 | 0.893 | 0.73  |
