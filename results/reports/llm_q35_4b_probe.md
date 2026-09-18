# Validation report: `llm_q35_4b_probe`

generated 2026-09-18 01:03, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.034 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = 0.02 (valid 0.843, pool 0.842; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.919   | 0.975   | 0.78    | 0.759   | 0.839 | 0.015   | 0.819       |
| global+ukg    | 0.915   | 0.974   | 0.788   | 0.765   | 0.84  | 0.012   | 0.836       |
| variety+ukg   | 0.915   | 0.977   | 0.793   | 0.769   | 0.842 | 0.01    | 0.843       |
| segment+ukg   | 0.915   | 0.976   | 0.789   | 0.769   | 0.842 | 0.01    | 0.843       |
| prev+ukg      | 0.92    | 0.976   | 0.787   | 0.774   | 0.847 | 0.012   | 0.82        |
| prevtuned+ukg | 0.92    | 0.976   | 0.787   | 0.774   | 0.847 | 0.006   | 0.85        |

**Headline policy: `segment+ukg`** (default policy); **score 0.8422**

## Expected test score for `segment+ukg`
**Point estimate 0.8422 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8114, 0.8730]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.915 | 0.916 | 0.014 | 0.894 | 0.938 | null        |
| sent-UK | 0.976 | 0.976 | 0.008 | 0.964 | 0.987 | null        |
| sarc-AU | 0.789 | 0.793 | 0.02  | 0.76  | 0.828 | null        |
| sarc-UK | 0.769 | 0.77  | 0.035 | 0.713 | 0.827 | null        |
| score   | 0.842 | 0.84  | 0.016 | 0.812 | 0.864 | 1.0         |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 71.9%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.911   | 0.975   | 0.824   | 0.782   | 0.847 |
| 1    | 0.913   | 0.969   | 0.793   | 0.785   | 0.849 |
| 2    | 0.916   | 0.969   | 0.793   | 0.733   | 0.825 |
| 3    | 0.915   | 0.981   | 0.763   | 0.758   | 0.836 |
| 4    | 0.92    | 0.987   | 0.774   | 0.783   | 0.847 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.462     | 0.915    | 0.911     | 0.906  | 0.978 | 0.977 | 0.058 | 0.011 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.976    | 0.982     | 0.973  | 0.996 | 0.996 | 0.02  | 0.002 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.738     | 0.972    | 0.984     | 0.987  | 0.998 | 0.999 | 0.016 | 0.006 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.314     | 0.863    | 0.82      | 0.806  | 0.953 | 0.913 | 0.08  | 0.027 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.744     | 0.978    | 0.992     | 0.986  | 0.995 | 0.998 | 0.015 | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.108     | 0.91     | 0.863     | 0.817  | 0.981 | 0.909 | 0.028 | 0.008 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.324     | 0.789    | 0.68      | 0.742  | 0.892 | 0.76  | 0.122 | 0.023 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.098     | 0.769    | 0.519     | 0.655  | 0.953 | 0.646 | 0.042 | 0.005 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.037     | 0.668    | 0.525     | 0.288  | 0.923 | 0.454 | 0.047 | 0.011 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.478     | 0.756    | 0.687     | 0.781  | 0.839 | 0.778 | 0.162 | 0.028 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 1.0   | 0.5   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.275     | 0.722    | 0.519     | 0.657  | 0.847 | 0.648 | 0.118 | 0.017 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.873 | 0.734 |
| 1-10  | UK      | 219 | 0.893 | 0.725 |
| 11-30 | AU      | 717 | 0.859 | 0.774 |
| 11-30 | UK      | 429 | 0.876 | 0.709 |
| 31-80 | AU      | 602 | 0.868 | 0.765 |
| 31-80 | UK      | 385 | 0.94  | 0.729 |
| >80   | AU      | 320 | 0.832 | 0.713 |
| >80   | UK      | 110 | 0.964 | 0.528 |
