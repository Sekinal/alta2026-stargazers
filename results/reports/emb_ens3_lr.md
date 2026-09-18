# Validation report: `emb_ens3_lr`

generated 2026-09-17 23:11, code 5fc610d-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [WARN] no fold_hash in meta (saved before provenance tracking); cannot prove fold match
  [WARN] no git commit recorded in meta
  [OK  ] per-fold score spread 0.021 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows; `valid_score` is the valid.csv subset of the same honest labels; `fold_sd` is the SD of the per-fold score.

| policy          | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|-----------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed           | 0.902   | 0.96    | 0.738   | 0.721   | 0.811 | 0.008   | 0.813       |
| fixed+ukg       | 0.902   | 0.96    | 0.738   | 0.721   | 0.811 | 0.008   | 0.813       |
| global+ukg      | 0.901   | 0.959   | 0.735   | 0.713   | 0.807 | 0.008   | 0.807       |
| variety+ukg     | 0.902   | 0.962   | 0.738   | 0.711   | 0.806 | 0.008   | 0.808       |
| segment+ukg     | 0.906   | 0.965   | 0.755   | 0.711   | 0.808 | 0.008   | 0.807       |
| prev+ukg        | 0.907   | 0.964   | 0.739   | 0.711   | 0.809 | 0.007   | 0.81        |
| prevtuned+ukg   | 0.907   | 0.964   | 0.748   | 0.712   | 0.81  | 0.003   | 0.809       |
| variety+ukg+neg | 0.898   | 0.958   | 0.738   | 0.711   | 0.804 | 0.008   | 0.803       |
| prev+ukg+neg    | 0.905   | 0.96    | 0.739   | 0.711   | 0.808 | 0.008   | 0.807       |

## Test-like bootstrap for best policy `fixed` (B=2000, valid-sized stratified replicates)
Expected leaderboard distribution from test-set sampling alone (p_above_ref = P(score > baseline 0.774)).

| metric  | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------------|
| sent-AU | 0.902 | 0.015 | 0.878 | 0.927 | null        |
| sent-UK | 0.96  | 0.01  | 0.943 | 0.974 | null        |
| sarc-AU | 0.737 | 0.022 | 0.698 | 0.772 | null        |
| sarc-UK | 0.723 | 0.037 | 0.661 | 0.784 | null        |
| score   | 0.807 | 0.016 | 0.779 | 0.833 | 0.97        |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 62.3%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.88    | 0.963   | 0.763   | 0.71    | 0.795 |
| 1    | 0.92    | 0.961   | 0.75    | 0.707   | 0.814 |
| 2    | 0.922   | 0.952   | 0.742   | 0.711   | 0.816 |
| 3    | 0.899   | 0.958   | 0.708   | 0.737   | 0.803 |
| 4    | 0.888   | 0.967   | 0.725   | 0.739   | 0.806 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.483     | 0.902    | 0.879     | 0.915  | 0.965 | 0.964 | 0.075 | 0.051 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.524     | 0.96     | 0.96      | 0.964  | 0.991 | 0.991 | 0.035 | 0.049 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.734     | 0.954    | 0.976     | 0.975  | 0.991 | 0.997 | 0.034 | 0.051 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.349     | 0.853    | 0.77      | 0.841  | 0.936 | 0.893 | 0.097 | 0.062 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.738     | 0.957    | 0.985     | 0.971  | 0.992 | 0.996 | 0.03  | 0.048 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.139     | 0.882    | 0.723     | 0.878  | 0.972 | 0.877 | 0.044 | 0.072 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.354     | 0.738    | 0.594     | 0.709  | 0.849 | 0.666 | 0.149 | 0.074 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.097     | 0.721    | 0.442     | 0.55   | 0.917 | 0.45  | 0.063 | 0.06  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.142     | 0.669    | 0.301     | 0.63   | 0.888 | 0.356 | 0.085 | 0.127 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.468     | 0.709    | 0.642     | 0.716  | 0.795 | 0.714 | 0.184 | 0.046 |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.001 | 0.005 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.271     | 0.663    | 0.442     | 0.552  | 0.736 | 0.451 | 0.177 | 0.16  |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.868 | 0.668 |
| 1-10  | UK      | 219 | 0.922 | 0.601 |
| 11-30 | AU      | 717 | 0.849 | 0.723 |
| 11-30 | UK      | 429 | 0.815 | 0.706 |
| 31-80 | AU      | 602 | 0.848 | 0.732 |
| 31-80 | UK      | 385 | 0.883 | 0.603 |
| >80   | AU      | 320 | 0.822 | 0.677 |
| >80   | UK      | 110 | 0.924 | 0.607 |
