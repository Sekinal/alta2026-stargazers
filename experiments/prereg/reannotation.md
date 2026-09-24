# Pre-registration: blind LLM re-annotation of Reddit rows (written before any annotation)

Date: 2026-09-24 ~00:50 UTC. Purpose: test whether the ~0.87 plateau is label-limited.

## Sample (fixed seed 2026; experiments/reannot/build_sample.py)
Per variety (en-AU, en-UK), Reddit rows of the pool:
- U stratum: 150 rows uniformly at random.
- D stratum: 50 rows at random among rows where the lead's OOF probability (stk_v5r) contradicts gold with margin > 0.8
  for SARCASM (|p_sarc - y_sarc| > 0.8), excluding U rows.
Items are shuffled and shown with an opaque item id and the variety only: no gold, no model output, no subreddit.

## Annotators
Two independent Claude subagents (A1, A2), same fixed instructions (experiments/reannot/instructions.md), each labels
every item for sarcasm (0/1) and sentiment (0 negative / 1 positive), in batches of 100.

## Measures
- U stratum: Cohen's kappa for gold-A1, gold-A2, A1-A2, lead-gold, lead-A1, lead-A2 (lead labels = segment+ukg decisions),
  sarcasm and sentiment, per variety, bootstrap 95% CI.
- D stratum: share of items where the A1/A2 majority (both agree) sides with the lead vs with gold.
## Predictions / interpretation
- Label-limited plateau: A1-A2 kappa > gold-A kappa, and lead-A >= lead-gold; in D, annotators side with the lead on a
  clear majority (> 60%).
- Model-limited plateau: annotators side with gold in D (>= 50%), lead-A << gold-A.
## Caveat
LLM annotators may share the models' notion of sarcasm; this measures agreement with a strong independent reader,
not with humans. Results are reported whichever way they fall.
