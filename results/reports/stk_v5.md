# Validation report: `stk_v5`

generated 2026-09-18 12:41, code ebc1a23-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] test split: 757 rows bound to experiments/encoders/fake_test.csv (d9e22447974963d0)
  [OK  ] fold_hash matches current folds
  [OK  ] permutation canary passed: all members passed (enc_deb_l_x5, cls_e5me5_lr, llm_q35_9b_probe)
  [OK  ] per-fold score spread 0.029 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -1.61 (valid 0.854, pool 0.874; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.933   | 0.977   | 0.8     | 0.81    | 0.867 | 0.012   | 0.826       |
| global+ukg    | 0.937   | 0.976   | 0.805   | 0.824   | 0.871 | 0.012   | 0.835       |
| variety+ukg   | 0.937   | 0.975   | 0.805   | 0.832   | 0.871 | 0.007   | 0.856       |
| segment+ukg   | 0.939   | 0.975   | 0.809   | 0.832   | 0.874 | 0.008   | 0.854       |
| prev+ukg      | 0.936   | 0.977   | 0.806   | 0.84    | 0.871 | 0.009   | 0.87        |
| prevtuned+ukg | 0.936   | 0.977   | 0.804   | 0.834   | 0.87  | 0.009   | 0.86        |

**Headline policy: `segment+ukg`** (default policy); **score 0.8737**

## Expected test score for `segment+ukg`
**Point estimate 0.8737 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.8479, 0.8995]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.939 | 0.939 | 0.012 | 0.919 | 0.959 | null        |
| sent-UK | 0.975 | 0.975 | 0.008 | 0.964 | 0.987 | null        |
| sarc-AU | 0.809 | 0.813 | 0.02  | 0.779 | 0.846 | null        |
| sarc-UK | 0.832 | 0.834 | 0.036 | 0.771 | 0.892 | null        |
| score   | 0.874 | 0.872 | 0.013 | 0.849 | 0.893 | 1.0         |

binding varieties: sent: AU binds 99.3%, sarc: UK binds 31.1%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.927   | 0.983   | 0.846   | 0.837   | 0.882 |
| 1    | 0.937   | 0.975   | 0.816   | 0.887   | 0.876 |
| 2    | 0.954   | 0.972   | 0.8     | 0.801   | 0.877 |
| 3    | 0.938   | 0.973   | 0.787   | 0.835   | 0.862 |
| 4    | 0.938   | 0.972   | 0.794   | 0.793   | 0.865 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.479     | 0.939    | 0.921     | 0.95   | 0.985 | 0.984 | 0.048 | 0.011 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.518     | 0.975    | 0.98      | 0.972  | 0.996 | 0.997 | 0.018 | 0.005 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.733     | 0.983    | 0.992     | 0.99   | 0.999 | 0.999 | 0.012 | 0.008 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.342     | 0.902    | 0.84      | 0.9    | 0.966 | 0.937 | 0.068 | 0.032 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.974    | 0.987     | 0.986  | 0.996 | 0.998 | 0.015 | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.912    | 0.89      | 0.802  | 0.987 | 0.938 | 0.023 | 0.008 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.297     | 0.809    | 0.731     | 0.731  | 0.91  | 0.792 | 0.113 | 0.039 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.072     | 0.832    | 0.717     | 0.663  | 0.97  | 0.752 | 0.034 | 0.009 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.026     | 0.683    | 0.714     | 0.274  | 0.938 | 0.542 | 0.043 | 0.007 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.442     | 0.781    | 0.731     | 0.771  | 0.866 | 0.807 | 0.151 | 0.055 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.96  | 0.012 | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.201     | 0.804    | 0.717     | 0.665  | 0.905 | 0.754 | 0.095 | 0.028 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.912 | 0.752 |
| 1-10  | UK      | 219 | 0.93  | 0.74  |
| 11-30 | AU      | 717 | 0.896 | 0.802 |
| 11-30 | UK      | 429 | 0.885 | 0.801 |
| 31-80 | AU      | 602 | 0.898 | 0.796 |
| 31-80 | UK      | 385 | 0.915 | 0.839 |
| >80   | AU      | 320 | 0.894 | 0.732 |
| >80   | UK      | 110 | 0.847 | 0.695 |
