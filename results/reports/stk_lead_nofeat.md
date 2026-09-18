# Validation report: `stk_lead_nofeat`

generated 2026-09-18 02:11, code 1ddcc30-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.027 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.79 (valid 0.845, pool 0.866; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.933   | 0.975   | 0.795   | 0.802   | 0.864 | 0.011   | 0.835       |
| global+ukg    | 0.932   | 0.975   | 0.794   | 0.818   | 0.863 | 0.008   | 0.842       |
| variety+ukg   | 0.931   | 0.977   | 0.794   | 0.828   | 0.863 | 0.007   | 0.848       |
| segment+ukg   | 0.933   | 0.975   | 0.799   | 0.828   | 0.866 | 0.005   | 0.845       |
| prev+ukg      | 0.937   | 0.977   | 0.797   | 0.829   | 0.867 | 0.007   | 0.858       |
| prevtuned+ukg | 0.937   | 0.977   | 0.792   | 0.828   | 0.864 | 0.005   | 0.858       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8659**

## Expected test score for `segment+ukg`
**Point estimate 0.8659 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8420, 0.8898]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.933 | 0.934 | 0.012 | 0.913 | 0.954 | null        |
| sent-UK | 0.975 | 0.976 | 0.007 | 0.964 | 0.987 | null        |
| sarc-AU | 0.799 | 0.804 | 0.018 | 0.773 | 0.834 | null        |
| sarc-UK | 0.828 | 0.831 | 0.035 | 0.771 | 0.888 | null        |
| score   | 0.866 | 0.866 | 0.012 | 0.845 | 0.884 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 23.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.919   | 0.974   | 0.822   | 0.846   | 0.871 |
| 1    | 0.925   | 0.975   | 0.8     | 0.865   | 0.863 |
| 2    | 0.946   | 0.974   | 0.789   | 0.786   | 0.866 |
| 3    | 0.938   | 0.976   | 0.776   | 0.859   | 0.857 |
| 4    | 0.936   | 0.978   | 0.807   | 0.782   | 0.859 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.469     | 0.933    | 0.924     | 0.933  | 0.984 | 0.983 | 0.049 | 0.006 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.975    | 0.981     | 0.972  | 0.996 | 0.996 | 0.018 | 0.002 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.979    | 0.99      | 0.987  | 0.998 | 0.999 | 0.012 | 0.01  |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.327     | 0.893    | 0.845     | 0.866  | 0.964 | 0.934 | 0.068 | 0.015 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.751     | 0.974    | 0.985     | 0.989  | 0.995 | 0.998 | 0.014 | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.096     | 0.91     | 0.918     | 0.771  | 0.985 | 0.926 | 0.025 | 0.003 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.362     | 0.799    | 0.665     | 0.811  | 0.907 | 0.787 | 0.113 | 0.017 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.076     | 0.828    | 0.691     | 0.675  | 0.964 | 0.736 | 0.036 | 0.007 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.032     | 0.68     | 0.618     | 0.288  | 0.933 | 0.517 | 0.044 | 0.011 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.539     | 0.759    | 0.666     | 0.856  | 0.861 | 0.803 | 0.151 | 0.028 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.443 | 0.001 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.213     | 0.799    | 0.691     | 0.677  | 0.891 | 0.739 | 0.1   | 0.019 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.912 | 0.747 |
| 1-10  | UK      | 219 | 0.907 | 0.775 |
| 11-30 | AU      | 717 | 0.89  | 0.78  |
| 11-30 | UK      | 429 | 0.896 | 0.791 |
| 31-80 | AU      | 602 | 0.878 | 0.765 |
| 31-80 | UK      | 385 | 0.912 | 0.817 |
| >80   | AU      | 320 | 0.883 | 0.709 |
| >80   | UK      | 110 | 0.924 | 0.695 |
