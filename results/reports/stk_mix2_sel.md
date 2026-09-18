# Validation report: `stk_mix2_sel`

generated 2026-09-18 00:59, code bbb1930-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.027 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -3.20 (valid 0.829, pool 0.872; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.933   | 0.976   | 0.795   | 0.808   | 0.864 | 0.01    | 0.811       |
| global+ukg    | 0.933   | 0.977   | 0.805   | 0.81    | 0.869 | 0.014   | 0.815       |
| variety+ukg   | 0.932   | 0.974   | 0.807   | 0.809   | 0.87  | 0.012   | 0.827       |
| segment+ukg   | 0.935   | 0.972   | 0.81    | 0.809   | 0.872 | 0.012   | 0.829       |
| prev+ukg      | 0.934   | 0.978   | 0.801   | 0.824   | 0.867 | 0.012   | 0.84        |
| prevtuned+ukg | 0.934   | 0.978   | 0.804   | 0.82    | 0.869 | 0.008   | 0.837       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8721**

## Expected test score for `segment+ukg`
**Point estimate 0.8721 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8438, 0.9004]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.935 | 0.936 | 0.012 | 0.916 | 0.956 | null        |
| sent-UK | 0.972 | 0.972 | 0.008 | 0.958 | 0.984 | null        |
| sarc-AU | 0.81  | 0.813 | 0.02  | 0.78  | 0.846 | null        |
| sarc-UK | 0.809 | 0.812 | 0.036 | 0.75  | 0.87  | null        |
| score   | 0.872 | 0.866 | 0.014 | 0.84  | 0.887 | 1.0         |

binding varieties: sent: AU binds 99.4%, sarc: UK binds 50.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.924   | 0.967   | 0.843   | 0.811   | 0.867 |
| 1    | 0.94    | 0.969   | 0.809   | 0.85    | 0.874 |
| 2    | 0.931   | 0.969   | 0.801   | 0.763   | 0.847 |
| 3    | 0.938   | 0.98    | 0.796   | 0.855   | 0.867 |
| 4    | 0.943   | 0.976   | 0.801   | 0.754   | 0.848 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.455     | 0.935    | 0.939     | 0.92   | 0.984 | 0.984 | 0.048 | 0.007 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.972    | 0.974     | 0.973  | 0.996 | 0.997 | 0.018 | 0.003 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.732     | 0.982    | 0.992     | 0.989  | 0.998 | 0.999 | 0.013 | 0.008 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.306     | 0.893    | 0.871     | 0.836  | 0.966 | 0.936 | 0.067 | 0.013 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.978    | 0.989     | 0.989  | 0.996 | 0.998 | 0.014 | 0.003 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.112     | 0.88     | 0.797     | 0.779  | 0.98  | 0.917 | 0.027 | 0.009 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.318     | 0.81     | 0.713     | 0.763  | 0.905 | 0.789 | 0.114 | 0.012 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.077     | 0.809    | 0.649     | 0.647  | 0.964 | 0.727 | 0.036 | 0.01  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.053     | 0.738    | 0.579     | 0.452  | 0.931 | 0.533 | 0.044 | 0.015 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.46      | 0.78     | 0.721     | 0.79   | 0.859 | 0.805 | 0.151 | 0.026 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.549 | 0.001 | 0.001 | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.217     | 0.776    | 0.649     | 0.649  | 0.89  | 0.729 | 0.101 | 0.02  |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.906 | 0.784 |
| 1-10  | UK      | 219 | 0.883 | 0.762 |
| 11-30 | AU      | 717 | 0.891 | 0.8   |
| 11-30 | UK      | 429 | 0.829 | 0.756 |
| 31-80 | AU      | 602 | 0.885 | 0.787 |
| 31-80 | UK      | 385 | 0.902 | 0.795 |
| >80   | AU      | 320 | 0.882 | 0.712 |
| >80   | UK      | 110 | 0.933 | 0.736 |
