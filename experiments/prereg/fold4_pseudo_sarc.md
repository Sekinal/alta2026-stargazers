# Pre-registration: fold-4 confirmation of pseudo-label sarcasm students (written before any fold-4 result)

Date: 2026-09-18 16:40 CST. Chosen on fold 0 (holdout, train folds 1-4) among 4 recipes; fold 4 is untouched by this choice.

## Recipe R_deb (fixed)
- Sentiment: identical to stk_v5 (logit mean of enc_deb_l_x5, cls_e5me5_lr, llm_q35_9b_probe).
- Sarcasm: logit mean of three members:
  1. deberta sarcasm = arithmetic mean of probabilities of (enc_deb_l_x5, pseudo-sarc student x3),
  2. cls_e5me5_lr, 3. llm_q35_9b_probe.
- Pseudo-sarc student: deberta-v3-large, stage 1 = sarcasm-only soft-label BCE on 400k in-domain texts
  (same sample as fold0.parquet) labelled by the FOLD-4 checkpoints of enc_deb_l_s13/14/15 (trained on folds 0-3);
  stage 2 = the unchanged gold fine-tune on folds 0-3; seeds 13/14/15; predict fold 4.
- Policy: segment+ukg cross-fitted within the fold-4 holdout.

## Fold-0 evidence (selection-biased: best of 4 recipes)
stk_v5 0.881 -> R_deb 0.886 (p 0.903); sarc-AU .837 -> .847, sarc-UK .858 -> .870. Qwen pseudo student added nothing
once averaged (0.881, p 0.43) and is not part of the recipe.

## Success criterion (fold 4)
p_a_better(R_deb vs stk_v5) >= 0.8 on the fold-4 holdout (n_test = fold size) AND sarc-AU not lower than stk_v5's.
No changes to the recipe after seeing fold 4. If it fails, stk_v5 remains the lead.

## Clarifications (added 2026-09-18 ~17:00 CST, before any fold-4 result)
- "Arithmetic mean of probabilities of (enc_deb_l_x5, pseudo-sarc student x3)" means TWO EQUAL ARMS:
  0.5 x (enc_deb_l_x5 probability, itself the mean of 5 seeds) + 0.5 x (mean of the 3 student seeds).
  This is exactly what was computed on fold 0.
- The pseudo-sarc arm is ONE stage-1 student (seed 13) followed by three gold fine-tune seeds (13/14/15) started from
  it, exactly as screened on fold 0.

## Outcome (2026-09-18 ~19:05 CST): FAILED
Evaluated unchanged (experiments/encoders/eval_prereg_f4.py, which reproduced the fold-0 figures) on fold 4 (1,253 rows):
R_deb 0.8589 vs stk_v5 0.8650, p_a_better 0.051; sarc-AU .7912 vs .8035 (p 0.042); sarc-UK .7946 vs .8189 (p 0.072).
Both criteria fail. The fold-0 gain (0.886 vs 0.881, p 0.90) was selection on fold 0. stk_v5 remains the lead.
Descriptive only: the fold-4 student alone ties base deberta (0.859 vs 0.857); its UK-reddit sarcasm AUC is lower (.833 vs .864).
