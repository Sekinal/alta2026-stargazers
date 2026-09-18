# Validation report: `enc_deb_l_ls05_x3`

generated 2026-09-18 03:21, code ae44cce-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.024 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.95 (valid 0.837, pool 0.861; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.931   | 0.973   | 0.792   | 0.777   | 0.854 | 0.009   | 0.801       |
| global+ukg    | 0.93    | 0.972   | 0.8     | 0.813   | 0.865 | 0.009   | 0.811       |
| variety+ukg   | 0.929   | 0.973   | 0.795   | 0.812   | 0.862 | 0.009   | 0.838       |
| segment+ukg   | 0.929   | 0.969   | 0.793   | 0.812   | 0.861 | 0.007   | 0.837       |
| prev+ukg      | 0.93    | 0.971   | 0.799   | 0.82    | 0.864 | 0.006   | 0.834       |
| prevtuned+ukg | 0.93    | 0.971   | 0.797   | 0.817   | 0.863 | 0.007   | 0.836       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8611**

## Expected test score for `segment+ukg`
**Point estimate 0.8611 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8344, 0.8878]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.929 | 0.93  | 0.013 | 0.908 | 0.951 | null        |
| sent-UK | 0.969 | 0.969 | 0.008 | 0.956 | 0.982 | null        |
| sarc-AU | 0.793 | 0.798 | 0.02  | 0.764 | 0.83  | null        |
| sarc-UK | 0.812 | 0.815 | 0.035 | 0.756 | 0.87  | null        |
| score   | 0.861 | 0.859 | 0.014 | 0.836 | 0.88  | 1.0         |

binding varieties: sent: AU binds 99.4%, sarc: UK binds 31.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.908   | 0.98    | 0.825   | 0.843   | 0.866 |
| 1    | 0.932   | 0.972   | 0.784   | 0.834   | 0.858 |
| 2    | 0.941   | 0.97    | 0.784   | 0.754   | 0.847 |
| 3    | 0.935   | 0.958   | 0.782   | 0.836   | 0.858 |
| 4    | 0.93    | 0.964   | 0.787   | 0.785   | 0.858 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.467     | 0.929    | 0.922     | 0.926  | 0.98  | 0.98  | 0.057 | 0.038 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.517     | 0.969    | 0.975     | 0.965  | 0.994 | 0.995 | 0.023 | 0.009 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.738     | 0.982    | 0.989     | 0.992  | 0.998 | 0.999 | 0.012 | 0.005 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.321     | 0.884    | 0.84      | 0.844  | 0.956 | 0.919 | 0.082 | 0.064 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.742     | 0.968    | 0.988     | 0.98   | 0.994 | 0.998 | 0.018 | 0.006 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.109     | 0.89     | 0.824     | 0.786  | 0.974 | 0.908 | 0.033 | 0.021 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.33      | 0.793    | 0.68      | 0.756  | 0.9   | 0.767 | 0.129 | 0.088 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.085     | 0.812    | 0.626     | 0.687  | 0.928 | 0.682 | 0.041 | 0.019 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.03      | 0.655    | 0.562     | 0.247  | 0.899 | 0.518 | 0.048 | 0.032 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.491     | 0.759    | 0.684     | 0.8    | 0.853 | 0.782 | 0.172 | 0.119 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.813 | 0.003 | 0.002 | 0.021 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.239     | 0.778    | 0.626     | 0.69   | 0.871 | 0.708 | 0.111 | 0.068 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.895 | 0.755 |
| 1-10  | UK      | 219 | 0.903 | 0.757 |
| 11-30 | AU      | 717 | 0.878 | 0.775 |
| 11-30 | UK      | 429 | 0.85  | 0.768 |
| 31-80 | AU      | 602 | 0.877 | 0.772 |
| 31-80 | UK      | 385 | 0.908 | 0.794 |
| >80   | AU      | 320 | 0.878 | 0.702 |
| >80   | UK      | 110 | 0.847 | 0.678 |
