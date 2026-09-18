# Validation report: `tfidf_lr`

generated 2026-09-17 23:11, code 5fc610d-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [WARN] no fold_hash in meta (saved before provenance tracking); cannot prove fold match
  [WARN] no git commit recorded in meta
  [WARN] per-fold score spread 0.076 (threshold 0.5)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows; `valid_score` is the valid.csv subset of the same honest labels; `fold_sd` is the SD of the per-fold score.

| policy          | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|-----------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed           | 0.837   | 0.917   | 0.681   | 0.584   | 0.711 | 0.028   | 0.731       |
| fixed+ukg       | 0.837   | 0.917   | 0.681   | 0.584   | 0.711 | 0.028   | 0.731       |
| global+ukg      | 0.835   | 0.916   | 0.671   | 0.66    | 0.747 | 0.011   | 0.746       |
| variety+ukg     | 0.835   | 0.916   | 0.698   | 0.66    | 0.747 | 0.012   | 0.751       |
| segment+ukg     | 0.837   | 0.913   | 0.698   | 0.66    | 0.748 | 0.012   | 0.746       |
| prev+ukg        | 0.837   | 0.915   | 0.678   | 0.626   | 0.731 | 0.01    | 0.731       |
| prevtuned+ukg   | 0.837   | 0.915   | 0.684   | 0.659   | 0.748 | 0.01    | 0.755       |
| variety+ukg+neg | 0.834   | 0.912   | 0.698   | 0.66    | 0.747 | 0.01    | 0.746       |
| prev+ukg+neg    | 0.835   | 0.914   | 0.678   | 0.626   | 0.731 | 0.009   | 0.73        |

## Test-like bootstrap for best policy `prevtuned+ukg` (B=2000, valid-sized stratified replicates)
Expected leaderboard distribution from test-set sampling alone (p_above_ref = P(score > baseline 0.774)).

| metric  | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------------|
| sent-AU | 0.837 | 0.019 | 0.805 | 0.865 | null        |
| sent-UK | 0.915 | 0.013 | 0.893 | 0.938 | null        |
| sarc-AU | 0.688 | 0.022 | 0.651 | 0.724 | null        |
| sarc-UK | 0.66  | 0.031 | 0.608 | 0.711 | null        |
| score   | 0.746 | 0.017 | 0.717 | 0.773 | 0.039       |

binding varieties: sent: AU binds 99.9%, sarc: UK binds 77.0%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.81    | 0.913   | 0.688   | 0.664   | 0.737 |
| 1    | 0.852   | 0.925   | 0.672   | 0.677   | 0.762 |
| 2    | 0.856   | 0.91    | 0.695   | 0.643   | 0.75  |
| 3    | 0.825   | 0.914   | 0.671   | 0.675   | 0.748 |
| 4    | 0.841   | 0.915   | 0.695   | 0.638   | 0.74  |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.465     | 0.837    | 0.825     | 0.825  | 0.917 | 0.917 | 0.12  | 0.077 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.522     | 0.915    | 0.919     | 0.919  | 0.974 | 0.978 | 0.068 | 0.073 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.735     | 0.869    | 0.93      | 0.93   | 0.96  | 0.985 | 0.083 | 0.097 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.319     | 0.776    | 0.695     | 0.695  | 0.86  | 0.762 | 0.14  | 0.061 |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.748     | 0.904    | 0.951     | 0.951  | 0.972 | 0.99  | 0.064 | 0.078 |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.114     | 0.738    | 0.538     | 0.534  | 0.871 | 0.576 | 0.075 | 0.064 |
| sarc-AU        | 3080 | 914   | 0.297      | 0.371     | 0.684    | 0.521     | 0.652  | 0.801 | 0.579 | 0.164 | 0.057 |
| sarc-UK        | 3208 | 249   | 0.078      | 0.149     | 0.659    | 0.301     | 0.578  | 0.885 | 0.313 | 0.064 | 0.04  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.086     | 0.616    | 0.261     | 0.329  | 0.847 | 0.303 | 0.063 | 0.091 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.524     | 0.626    | 0.544     | 0.68   | 0.692 | 0.593 | 0.219 | 0.04  |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.418     | 0.558    | 0.301     | 0.581  | 0.631 | 0.314 | 0.178 | 0.11  |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.749 | 0.599 |
| 1-10  | UK      | 219 | 0.806 | 0.483 |
| 11-30 | AU      | 717 | 0.755 | 0.637 |
| 11-30 | UK      | 429 | 0.657 | 0.543 |
| 31-80 | AU      | 602 | 0.806 | 0.629 |
| 31-80 | UK      | 385 | 0.733 | 0.561 |
| >80   | AU      | 320 | 0.771 | 0.615 |
| >80   | UK      | 110 | 0.569 | 0.453 |
