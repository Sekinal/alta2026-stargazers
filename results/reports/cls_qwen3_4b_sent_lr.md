# Validation report: `cls_qwen3_4b_sent_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.016 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -2.48 (valid 0.794, pool 0.832; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.917   | 0.971   | 0.744   | 0.755   | 0.83  | 0.006   | 0.812       |
| global+ukg    | 0.914   | 0.971   | 0.747   | 0.749   | 0.831 | 0.006   | 0.788       |
| variety+ukg   | 0.914   | 0.972   | 0.747   | 0.748   | 0.831 | 0.008   | 0.793       |
| segment+ukg   | 0.916   | 0.969   | 0.772   | 0.748   | 0.832 | 0.009   | 0.794       |
| prev+ukg      | 0.916   | 0.97    | 0.767   | 0.755   | 0.836 | 0.009   | 0.79        |
| prevtuned+ukg | 0.916   | 0.97    | 0.766   | 0.753   | 0.835 | 0.011   | 0.804       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8320**

## Expected test score for `segment+ukg`
**Point estimate 0.8320 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7997, 0.8643]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.916 | 0.917 | 0.013 | 0.894 | 0.938 | null        |
| sent-UK | 0.969 | 0.969 | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.772 | 0.778 | 0.02  | 0.745 | 0.811 | null        |
| sarc-UK | 0.748 | 0.751 | 0.035 | 0.694 | 0.806 | null        |
| score   | 0.832 | 0.831 | 0.016 | 0.802 | 0.855 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 75.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.904   | 0.977   | 0.792   | 0.772   | 0.838 |
| 1    | 0.913   | 0.963   | 0.772   | 0.728   | 0.82  |
| 2    | 0.926   | 0.961   | 0.767   | 0.728   | 0.827 |
| 3    | 0.921   | 0.967   | 0.746   | 0.786   | 0.833 |
| 4    | 0.913   | 0.98    | 0.78    | 0.721   | 0.817 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.45      | 0.916    | 0.923     | 0.894  | 0.976 | 0.976 | 0.06  | 0.017 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.526     | 0.969    | 0.968     | 0.974  | 0.995 | 0.996 | 0.023 | 0.007 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.736     | 0.982    | 0.99      | 0.991  | 0.998 | 0.999 | 0.014 | 0.009 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.297     | 0.858    | 0.834     | 0.775  | 0.947 | 0.904 | 0.085 | 0.019 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.755     | 0.971    | 0.981     | 0.99   | 0.995 | 0.998 | 0.017 | 0.005 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.111     | 0.882    | 0.803     | 0.779  | 0.977 | 0.884 | 0.034 | 0.016 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.331     | 0.772    | 0.651     | 0.725  | 0.862 | 0.697 | 0.146 | 0.076 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.103     | 0.748    | 0.476     | 0.631  | 0.941 | 0.577 | 0.056 | 0.054 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.009     | 0.543    | 0.5       | 0.068  | 0.878 | 0.322 | 0.103 | 0.138 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.503     | 0.732    | 0.653     | 0.782  | 0.829 | 0.762 | 0.169 | 0.044 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.289     | 0.694    | 0.476     | 0.633  | 0.815 | 0.579 | 0.157 | 0.15  |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.858 | 0.71  |
| 1-10  | UK      | 219 | 0.896 | 0.656 |
| 11-30 | AU      | 717 | 0.865 | 0.747 |
| 11-30 | UK      | 429 | 0.835 | 0.678 |
| 31-80 | AU      | 602 | 0.844 | 0.738 |
| 31-80 | UK      | 385 | 0.902 | 0.713 |
| >80   | AU      | 320 | 0.845 | 0.706 |
| >80   | UK      | 110 | 0.865 | 0.683 |
