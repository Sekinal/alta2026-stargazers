# Validation report: `cls_qwen3_sent_lr`

generated 2026-09-17 23:55, code dc7268a-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [OK  ] per-fold score spread 0.025 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.903   | 0.963   | 0.732   | 0.721   | 0.812 | 0.011   | 0.788       |
| global+ukg    | 0.903   | 0.962   | 0.735   | 0.731   | 0.817 | 0.011   | 0.789       |
| variety+ukg   | 0.903   | 0.964   | 0.736   | 0.731   | 0.817 | 0.011   | 0.789       |
| segment+ukg   | 0.901   | 0.964   | 0.763   | 0.731   | 0.816 | 0.004   | 0.787       |
| prev+ukg      | 0.901   | 0.963   | 0.762   | 0.728   | 0.815 | 0.004   | 0.788       |
| prevtuned+ukg | 0.901   | 0.963   | 0.76    | 0.73    | 0.816 | 0.005   | 0.797       |

**Headline policy: `segment+ukg`** (default policy); **score 0.8163**

## Expected test score for `segment+ukg`
**Point estimate 0.8163 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7801, 0.8526]** = CV +- 1.645 x 1.2 x bootstrap SD.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.901 | 0.902 | 0.015 | 0.876 | 0.927 | null        |
| sent-UK | 0.964 | 0.964 | 0.009 | 0.948 | 0.977 | null        |
| sarc-AU | 0.763 | 0.769 | 0.021 | 0.735 | 0.804 | null        |
| sarc-UK | 0.731 | 0.732 | 0.038 | 0.667 | 0.791 | null        |
| score   | 0.816 | 0.815 | 0.018 | 0.782 | 0.843 | 0.98        |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 81.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.896   | 0.966   | 0.77    | 0.721   | 0.808 |
| 1    | 0.9     | 0.955   | 0.77    | 0.732   | 0.816 |
| 2    | 0.916   | 0.967   | 0.766   | 0.715   | 0.816 |
| 3    | 0.884   | 0.966   | 0.735   | 0.77    | 0.809 |
| 4    | 0.91    | 0.967   | 0.774   | 0.722   | 0.816 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.457     | 0.901    | 0.901     | 0.887  | 0.968 | 0.967 | 0.07  | 0.01  |
| sent-UK        | 3208 | 1676  | 0.522      | 0.516     | 0.964    | 0.972     | 0.959  | 0.992 | 0.993 | 0.029 | 0.011 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.743     | 0.957    | 0.972     | 0.982  | 0.995 | 0.998 | 0.021 | 0.012 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.304     | 0.845    | 0.807     | 0.769  | 0.931 | 0.874 | 0.097 | 0.019 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.744     | 0.964    | 0.984     | 0.979  | 0.991 | 0.997 | 0.023 | 0.004 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.103     | 0.867    | 0.805     | 0.725  | 0.967 | 0.85  | 0.039 | 0.022 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.311     | 0.763    | 0.655     | 0.686  | 0.84  | 0.64  | 0.158 | 0.095 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.095     | 0.731    | 0.462     | 0.566  | 0.925 | 0.498 | 0.066 | 0.071 |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.018     | 0.58     | 0.474     | 0.123  | 0.895 | 0.375 | 0.118 | 0.169 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.468     | 0.726    | 0.659     | 0.735  | 0.809 | 0.729 | 0.179 | 0.054 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.267     | 0.677    | 0.462     | 0.569  | 0.764 | 0.5   | 0.184 | 0.196 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.865 | 0.702 |
| 1-10  | UK      | 219 | 0.861 | 0.629 |
| 11-30 | AU      | 717 | 0.822 | 0.727 |
| 11-30 | UK      | 429 | 0.824 | 0.705 |
| 31-80 | AU      | 602 | 0.87  | 0.744 |
| 31-80 | UK      | 385 | 0.908 | 0.639 |
| >80   | AU      | 320 | 0.8   | 0.702 |
| >80   | UK      | 110 | 0.824 | 0.559 |
