# Validation report: `cls_qwen3_4b_sarc_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.033 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.35 (valid 0.815, pool 0.835; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.909   | 0.959   | 0.766   | 0.762   | 0.835 | 0.012   | 0.818       |
| global+ukg    | 0.907   | 0.959   | 0.768   | 0.766   | 0.837 | 0.01    | 0.813       |
| variety+ukg   | 0.908   | 0.959   | 0.769   | 0.761   | 0.834 | 0.009   | 0.81        |
| segment+ukg   | 0.908   | 0.959   | 0.779   | 0.761   | 0.835 | 0.008   | 0.815       |
| prev+ukg      | 0.905   | 0.959   | 0.78    | 0.746   | 0.826 | 0.008   | 0.79        |
| prevtuned+ukg | 0.905   | 0.959   | 0.78    | 0.762   | 0.834 | 0.01    | 0.807       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8348**

## Expected test score for `segment+ukg`
**Point estimate 0.8348 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8046, 0.8649]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.908 | 0.909 | 0.014 | 0.886 | 0.932 | null        |
| sent-UK | 0.959 | 0.959 | 0.01  | 0.943 | 0.974 | null        |
| sarc-AU | 0.779 | 0.784 | 0.021 | 0.748 | 0.818 | null        |
| sarc-UK | 0.761 | 0.764 | 0.033 | 0.711 | 0.817 | null        |
| score   | 0.835 | 0.833 | 0.015 | 0.806 | 0.857 | 1.0         |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 69.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.908   | 0.971   | 0.826   | 0.773   | 0.84  |
| 1    | 0.907   | 0.955   | 0.771   | 0.754   | 0.831 |
| 2    | 0.917   | 0.961   | 0.779   | 0.762   | 0.839 |
| 3    | 0.908   | 0.951   | 0.751   | 0.779   | 0.829 |
| 4    | 0.902   | 0.956   | 0.765   | 0.739   | 0.82  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.458     | 0.908    | 0.908     | 0.894  | 0.973 | 0.973 | 0.063 | 0.006 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.519     | 0.959    | 0.963     | 0.958  | 0.992 | 0.993 | 0.03  | 0.007 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.738     | 0.96     | 0.977     | 0.981  | 0.995 | 0.998 | 0.025 | 0.006 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.307     | 0.856    | 0.818     | 0.788  | 0.946 | 0.904 | 0.084 | 0.017 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.753     | 0.952    | 0.973     | 0.979  | 0.992 | 0.997 | 0.027 | 0.008 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.098     | 0.869    | 0.83      | 0.71   | 0.967 | 0.858 | 0.037 | 0.017 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.31      | 0.779    | 0.677     | 0.707  | 0.867 | 0.696 | 0.141 | 0.072 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.104     | 0.761    | 0.494     | 0.663  | 0.942 | 0.573 | 0.057 | 0.057 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.025     | 0.633    | 0.556     | 0.205  | 0.88  | 0.402 | 0.092 | 0.121 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.463     | 0.744    | 0.681     | 0.75   | 0.83  | 0.757 | 0.168 | 0.05  |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.292     | 0.71     | 0.494     | 0.665  | 0.821 | 0.575 | 0.159 | 0.159 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.845 | 0.709 |
| 1-10  | UK      | 219 | 0.892 | 0.668 |
| 11-30 | AU      | 717 | 0.871 | 0.751 |
| 11-30 | UK      | 429 | 0.797 | 0.683 |
| 31-80 | AU      | 602 | 0.842 | 0.766 |
| 31-80 | UK      | 385 | 0.893 | 0.741 |
| >80   | AU      | 320 | 0.842 | 0.715 |
| >80   | UK      | 110 | 0.865 | 0.678 |
