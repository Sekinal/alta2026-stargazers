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
