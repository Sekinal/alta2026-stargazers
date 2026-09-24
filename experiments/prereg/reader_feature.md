# Pre-registration: LLM-reader sarcasm judgment as a stacker feature (written before the full annotation)

Date: 2026-09-24 ~01:30 UTC. Motivation: R6 U-stratum (150 UK Reddit rows) AUC lead .903 -> .929 with the two readers' vote
added (AU .842 -> .849). Selection caveat: this idea was chosen after seeing R6.

## Annotation
One Claude subagent reader per batch of 100, fixed guideline experiments/reannot/instructions.md, text + variety only,
shuffled, no gold/model output. Scope: all en-UK Reddit rows of the pool (1,143) and of the test file (286).
(AU only if UK passes.)

## Model (fixed)
For UK Reddit rows only, sarcasm: cross-fitted logistic regression on [lead logit (stk_v5r OOF), reader label (0/1)],
fitted on the other 4 folds' UK Reddit rows; all other rows/tasks keep the lead's probability. Test: average of the
5 fold models applied to the lead's test logit and the reader's test label.

## Success criterion
Official score with the cross-fitted segment+ukg policy, `validate compare` vs stk_v5r: sarc-UK p_a_better(new) >= 0.85
AND overall score higher on >= 4/5 folds. Otherwise report as negative.
