# Validation report: `cls_go_emo_lr`

generated 2026-09-18 00:25, code 8cf9b41-dirty, folds b9ba64c66508

## Integrity checks
```
  [OK  ] OOF covers all 6288 pool rows exactly once
  [OK  ] fold_hash matches current folds
  [WARN] folds not measured: fold_hash only proves the pool's folds, not the ones the model used; train through alta.validation.harness.cv_fit_predict and pass folds_used
  [WARN] no permutation canary in meta (alta.validation.harness.canary); leaks from supervised preprocessing / in-sample predictions are invisible without it
  [OK  ] per-fold score spread 0.022 (threshold 0.5)
  [OK  ] valid-vs-pool gap z = -0.64 (valid 0.785, pool 0.796; > 2 suggests valid rows leaked into training)
```

## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)
Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x the regret of the pool score. Policy differences within a run are mostly noise; the headline uses `segment+ukg` unless another policy wins by >= 0.005 with paired p >= 0.95.

| policy        | sent-AU | sent-UK | sarc-AU | sarc-UK | score | fold_sd | valid_score |
|---------------|---------|---------|---------|---------|-------|---------|-------------|
| fixed+ukg     | 0.886   | 0.948   | 0.695   | 0.709   | 0.791 | 0.009   | 0.796       |
| global+ukg    | 0.884   | 0.947   | 0.7     | 0.709   | 0.792 | 0.003   | 0.791       |
| variety+ukg   | 0.886   | 0.948   | 0.701   | 0.707   | 0.793 | 0.002   | 0.787       |
| segment+ukg   | 0.884   | 0.947   | 0.726   | 0.707   | 0.796 | 0.005   | 0.785       |
| prev+ukg      | 0.883   | 0.947   | 0.724   | 0.711   | 0.797 | 0.006   | 0.784       |
| prevtuned+ukg | 0.883   | 0.947   | 0.72    | 0.71    | 0.796 | 0.009   | 0.778       |

**Headline policy: `segment+ukg`** (default policy); **score 0.7955**

## Expected test score for `segment+ukg`
**Point estimate 0.7955 (pool CV; red-team: ~unbiased for the chosen system). 90% interval [0.7601, 0.8309]** = CV +- 1.645 x 1.2 x bootstrap SD. References: no-information floor 0.6416, organisers' baseline 0.774.

Test-sampling-only bootstrap (B=2000, valid-sized stratified replicates; its q05/q95 are too narrow on their own, and boot mean runs ~0.0015 pessimistic):

| metric  | cv    | mean  | sd    | q05   | q95   | p_above_ref |
|---------|-------|-------|-------|-------|-------|-------------|
| sent-AU | 0.884 | 0.885 | 0.016 | 0.857 | 0.911 | null        |
| sent-UK | 0.947 | 0.946 | 0.011 | 0.927 | 0.964 | null        |
| sarc-AU | 0.726 | 0.732 | 0.022 | 0.693 | 0.769 | null        |
| sarc-UK | 0.707 | 0.705 | 0.039 | 0.642 | 0.77  | null        |
| score   | 0.796 | 0.791 | 0.018 | 0.76  | 0.819 | 0.834       |

binding varieties: sent: AU binds 100.0%, sarc: UK binds 72.8%

## Per-fold scores

| fold | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|------|---------|---------|---------|---------|-------|
| 0    | 0.868   | 0.944   | 0.763   | 0.715   | 0.792 |
| 1    | 0.881   | 0.939   | 0.721   | 0.721   | 0.801 |
| 2    | 0.889   | 0.956   | 0.732   | 0.69    | 0.789 |
| 3    | 0.89    | 0.944   | 0.714   | 0.71    | 0.8   |
| 4    | 0.89    | 0.95    | 0.699   | 0.703   | 0.795 |

## Diagnostics by component and segment

| group          | n    | n_pos | prevalence | pred_rate | macro_f1 | precision | recall | auc   | ap    | brier | ece   |
|----------------|------|-------|------------|-----------|----------|-----------|--------|-------|-------|-------|-------|
| sent-AU        | 3080 | 1431  | 0.465      | 0.458     | 0.884    | 0.881     | 0.869  | 0.952 | 0.949 | 0.084 | 0.006 |
| sent-UK        | 3208 | 1676  | 0.522      | 0.53      | 0.947    | 0.942     | 0.956  | 0.987 | 0.988 | 0.042 | 0.015 |
| sent:AU-google | 1076 | 791   | 0.735      | 0.747     | 0.943    | 0.963     | 0.979  | 0.981 | 0.992 | 0.038 | 0.017 |
| sent:AU-reddit | 2004 | 640   | 0.319      | 0.303     | 0.82     | 0.773     | 0.733  | 0.911 | 0.835 | 0.109 | 0.02  |
| sent:UK-google | 2065 | 1545  | 0.748      | 0.766     | 0.936    | 0.958     | 0.98   | 0.987 | 0.995 | 0.036 | 0.02  |
| sent:UK-reddit | 1143 | 131   | 0.115      | 0.105     | 0.837    | 0.742     | 0.679  | 0.941 | 0.735 | 0.052 | 0.02  |
| sarc-AU        | 3080 | 914   | 0.297      | 0.29      | 0.726    | 0.62      | 0.606  | 0.792 | 0.549 | 0.18  | 0.11  |
| sarc-UK        | 3208 | 249   | 0.078      | 0.09      | 0.707    | 0.431     | 0.502  | 0.918 | 0.44  | 0.069 | 0.07  |
| sarc:AU-google | 1076 | 73    | 0.068      | 0.002     | 0.496    | 0.5       | 0.014  | 0.845 | 0.252 | 0.138 | 0.214 |
| sarc:AU-reddit | 2004 | 841   | 0.42       | 0.445     | 0.681    | 0.62      | 0.658  | 0.751 | 0.661 | 0.203 | 0.06  |
| sarc:UK-google | 2065 | 1     | 0.0        | 0.0       | 0.5      | 0.0       | 0.0    | 0.1   | 0.0   | 0.0   | 0.001 |
| sarc:UK-reddit | 1143 | 248   | 0.217      | 0.254     | 0.65     | 0.431     | 0.504  | 0.74  | 0.441 | 0.193 | 0.194 |

## Reddit slices by length (macro-F1)

| words | variety | n   | sent  | sarc  |
|-------|---------|-----|-------|-------|
| 1-10  | AU      | 365 | 0.795 | 0.666 |
| 1-10  | UK      | 219 | 0.861 | 0.68  |
| 11-30 | AU      | 717 | 0.812 | 0.68  |
| 11-30 | UK      | 429 | 0.757 | 0.672 |
| 31-80 | AU      | 602 | 0.825 | 0.706 |
| 31-80 | UK      | 385 | 0.884 | 0.586 |
| >80   | AU      | 320 | 0.841 | 0.647 |
| >80   | UK      | 110 | 0.761 | 0.505 |
