# Annotated results (for the paper; not prose)

All scores: official metric, `segment+ukg` policy cross-fitted, 5-fold OOF on the 6,288 pool rows unless noted.
Tables in this folder contain no raw text.

## R1. Agreement with the single BESSTIE annotator vs a second human  (`kappa_vs_gold.csv`, `human_kappa_ci.csv`)
- Cohen's kappa vs gold (sarcasm on Reddit rows), 95% bootstrap CI:
  lead stk_v5: sent-AU .88 [.86,.89], sent-UK .95 [.94,.96], sarc-AU .56 [.53,.60], sarc-UK .61 [.55,.66].
- BESSTIE second-annotator kappa (n=50 per variety): sent-AU .61 [.37,.80], sent-UK .79 [.60,.95],
  sarc-AU .47 [.19,.72], sarc-UK .63 [-.03,1.00] (CIs: parametric bootstrap from the implied 2x2 at pool prevalence).
- Annotation: sentiment -> models clearly exceed human-human agreement (even label-free zero-shot LLMs do: .78-.80 AU,
  .92-.93 UK). Sarcasm-AU -> lead sits at the human estimate, well inside its CI. Sarcasm-UK -> the published human
  kappa is uninformative (~4 sarcastic items in 50); no claim possible without new annotation.
- Label-free zero-shot LLMs reach only .35-.41 on sarcasm: supervised models capture annotator-specific sarcasm.
- Consistent with the plateau: every method lands at 0.87-0.88 overall; sarcasm is at the agreement ceiling.

## R2. Cross-variety transfer (pending: transfer_matrix.csv, transfer_llm_probe.csv)
- Prior evidence (EDA): lexical sarcasm log-odds AU vs UK r = 0.00 (890 shared terms); probe weight cosine
  sarcasm 0.21 vs sentiment 0.37; adding the other variety hurts sarcasm at every training size
  (UK .683 -> .644 with AU data; embedding probes). Fine-tuned deberta: AU-only sarcasm training is WORSE for AU
  (.789 vs .798), i.e. for fine-tuning UK data helps AU (opposite of the probe result).

### R2d. Transfer with fine-tuned deberta-v3-large (3 seeds, standard folds; `transfer_matrix.csv`)
| train -> eval | sent F1 | sent AUC | sarc F1 (reddit) | sarc AUC (reddit) |
| AU -> AU | .917 +- .005 | .971 | .758 +- .003 | .840 |
| AU -> UK | .963 +- .002 | .992 | .638 +- .005 | .715 |
| UK -> AU | .916 +- .003 | .960 | .605 +- .021 | .678 |
| UK -> UK | .970 +- .003 | .992 | .778 +- .014 | .859 |
| both -> AU | .928 +- .004 | .977 | .765 +- .006 | .845 |
| both -> UK | .970 +- .001 | .994 | .776 +- .001 | .861 |
- Annotation: sentiment transfers fully (UK->AU .916 = AU->AU .917). Sarcasm does not (-0.12 to -0.17 F1, -0.14 to -0.16
  AUC), consistently across three model families (fine-tuned encoder R2d, LLM probe R2b, interpretable surrogate R2a).
  Yet joint training >= in-variety for a fine-tuned encoder (AU .765 vs .758; UK tie) - unlike frozen-embedding probes,
  where pooling hurt UK sarcasm (.683 -> .644): a fine-tuned model conditioned on the variety tag can hold two different
  sarcasm constructs; a single linear probe over fixed embeddings cannot.

### R2b. Transfer with LLM features (9B probe, standard folds; `transfer_llm_probe.csv`)
- Sarcasm (Reddit), AUC / cross-fitted macro-F1: AU->AU .851/.751, AU->UK .709/.601, UK->UK .878/.767, UK->AU .656/.627.
  Sentiment (all rows) AUC: AU->AU .979, AU->UK .995, UK->UK .995, UK->AU .970.
- Annotation: sarcasm does not transfer even with LLM representations (-.14 to -.20 AUC); sentiment transfers ~fully.

### R2c. Label-free LLM as an independent annotator (Qwen3.5-4B zero-shot, t=0.5; `zs_disagreement.csv`)
- AU Reddit sarcasm: agreement .709, kappa .385, AUC .781; predicts 34% sarcastic vs gold 42%; errors mostly
  gold-yes/model-no (18.6% of rows vs 10.5%), worst in AustralianPolitics (22.5%) and on replies (23.1% vs 17.0% top-level).
- UK Reddit sarcasm: agreement .621, kappa .269, AUC .786; predicts 53% vs gold 22%; errors mostly model-yes/gold-no
  (34.7% vs 3.1%), worst in GreatBritishMemes (39.6%; AUC .713); replies AUC .747 vs top-level .806.
- Annotation: identical ranking quality in both varieties (AUC ~.78) but the label-free decision boundary sits between
  the two annotators: stricter than AU (which also labels cynical political complaints sarcastic) and much looser than UK.
  Consistent with R2a: the AU and UK labels apply different thresholds / constructs of "sarcasm".

### R2a. Annotator surrogate: what each annotator labels sarcastic  (`surrogate_*.csv`, experiments/classical/surrogate.py)
- Reddit rows, interpretable L2-LR per variety, 5-fold AUC (fold SD):
  thread block (subreddit/top-level) AU .620 (.011) / UK .580 (.034); topic+polarity AU .629 / UK .623;
  pragmatic cues AU .604 / UK .574; ALL AU .692 (.016) / UK .637 (.054). Gold NEGATIVE sentiment alone: AU .745, UK .565.
  Reference: lead stk_v5 UK-reddit sarcasm AUC ~.90.
- Transfer (train one variety, AUC on the other): ALL AU->UK .466, UK->AU .462 (below chance);
  topic+polarity block anti-transfers (.393 / .402) because signs flip.
- Sign flips: VADER compound -.22 [-.35,-.11] AU vs +.33 [.17,.52] UK; laughter +.14 AU vs -.18 UK.
  Shared cues (same sign, CI excl. 0 in both): interjection-initial, question share, pos-lexicon rate (negative).
- Annotation: AU sarcasm ~ "political / negative" (AustralianPolitics +.30, negative tone, short texts; gold negativity
  alone beats the surrogate). UK sarcasm is mostly NOT captured by interpretable cues (.637 vs ~.90 for the lead); its best
  interpretable signal is polarity incongruity (positive overall tone with fewer explicit sentiment words).
  The two varieties' labels anti-transfer at the interpretable level: they encode different constructs.

## R3. Benchmark audit: Reddit provenance and thread overlap (pending: thread_gap.csv)
- 95% of Reddit rows recovered (AU 95.0%, UK 95.9%) from 7 subreddits, 2024-02-18 .. 2024-06-02 (latest comments per
  subreddit): AustralianPolitics 662, melbourne 636, AskAnAustralian 605, unitedkingdom 358, england 285,
  Britain 246, GreatBritishMemes 207.
- 82% of valid's matched rows share a thread with train (official split is by comment). 2,562/3,000 matched pool
  rows share a submission with another pool row (1,221 distinct submissions); 111 are direct replies to another row.
- AU sarcasm rate by subreddit: AustralianPolitics 61%, AskAnAustralian 35%, melbourne 31%.
- Text-only e5+me5 probe: 0.847 standard folds -> 0.838 link_id-grouped folds (sarc-UK .788 -> .764; all 5 folds).
- 9B probe: 0.853 standard -> 0.859 thread folds (sarc-UK .806 -> .797, UK-reddit AUC .878 -> .875): no thread inflation
  (`thread_folds_llm_probe.csv`).
- Logit-mean blend of e5+me5 probe and 9B probe: 0.862 standard -> 0.862 thread (-0.001): thread-neutral
  (`thread_gap_partial_e5_llm.csv`).
- deberta-v3-large x3 (`thread_folds.csv`): 0.866 standard vs 0.865 thread folds (paired p(std better) 0.57: tie);
  sent-AU .934 -> .927, sarc-AU .799 -> .804, sarc-UK .805 -> .808; fold SD .010 vs .015.
- Annotation: the strongest text-only members show no thread-memorisation advantage; only the frozen e5 probe loses
  ~0.008. The benchmark's comment-level split does not materially inflate CV for fine-tuned / LLM models.
- Full lead (3-member logit blend; deberta x3 seeds averaged by probability mean, as in the lead; `thread_gap.csv`):
  0.871 standard -> 0.869 thread (-0.001); thread components sent-AU .931, sent-UK .978, sarc-AU .808, sarc-UK .839;
  fold SD .008. Member gaps: deberta x3 0.866 -> 0.865 (-0.001), e5 -0.008, 9B probe +0.007.
  (Corrected: an earlier version averaged the thread seeds in logit space, mixing methods, and gave 0.868.)
- Annotation: the lead has essentially no thread-memorisation gap; only the frozen e5 probe shows one (-0.008).
  Headline stays the standard-fold 0.871 (official split is by comment); 0.869 is the thread-disjoint estimate.
- Method note: `validate report` fits thresholds over the STANDARD folds; thread-fold numbers must use the fold override
  in experiments/encoders/thread_compare.py / experiments/classical/thread_gap.py.

- Recovered context does not help: context e5 probe 0.839 vs 0.847 (std) / 0.827 vs 0.838 (thread); context 9B probe
  0.854 vs 0.853; context features in the stacker 0.871 vs 0.872.

## R4. Negative results under a strict protocol
- LoRA ladder (Qwen3.5-4B, fold 0, 2 seeds): LoRA+rsLoRA .853/.849; EVA+rsLoRA .818/.805 (p>=.995 worse);
  EVA+DoRA+rsLoRA+Riemannian .751. EVA init at a 32-batch cap: 115/127 of 248 layers unconverged.
- AstroCLIMB head recipe: .856/.856 (tie). Unsloth parity: .857 vs .853, 1.53x faster.
- Pretraining (deberta, fold 0, 3 seeds): /s sarcasm pre-FT .860 (sarc-AU .830 -> .810); DAPT 43M tokens .863 (tie,
  lower seed SD); both .853.
- Pseudo-label adaptation: fold-0 recipe 0.886 vs 0.881 (p .90, best of 4) FAILED its pre-registered fold-4
  confirmation (0.859 vs 0.865, p .05). See experiments/prereg/fold4_pseudo_sarc.md.
- Single-seed trap: Qwen sarcasm pre-SFT seed 0 .870, seed 1 .854.

## R5. Post-plateau screens (2026-09-23; fold 0, seeds 13-15 paired with enc_deb_l_s13..15; segment+ukg in-holdout)
| arm | 3-seed blend | seed mean +- SD | P(base better) |
| base deberta-v3-large | .8711 | .8639 +- .0086 | - |
| + weight EMA (.999) | .8716 | .8652 +- .0054 | .43 |
| + LLRD .9, lr 1.5e-5, EMA | .8754 | .8659 +- .0054 | .36 |
| NLI deberta-v3-large (MoritzLaurer mnli-fever-anli-ling-wanli) | .8548 | .8456 +- .0125 | .90 |
| ELECTRA-large | .8622 | .8467 +- .0059 | .90 |
| ModernBERT-large (lr 2e-5) | .8593 | .8533 +- .0032 | .82 |
- Inside the lead (fold 0, lead .881): adding any arm as a 4th logit-mean member .874-.883 (ties); swapping the lead's
  deberta for a 3-seed arm: base x3 .884, EMA x3 .879, LLRD+EMA x3 .877. EMA/LLRD lower seed variance ~35% but the
  standalone gain does not survive the ensemble.
- Transductive same-thread label features over the lead (nested cross-fit, experiments/classical/thread_labels.py):
  .8742 vs .8737, unstable coefficients -> noise. Cross-fitted kNN (e5+me5, K=30) as a 4th member: .869 vs .874 (lead wins 5/5 folds).
- Annotation: the ensemble is saturated at ~.874 OOF; backbone/recipe diversity and label-propagation signals add nothing.

## R6. Blind LLM re-annotation (pre-registered: experiments/prereg/reannotation.md; results/paper/reannotation.csv)
Two independent Claude subagents (same fixed guideline, text + variety only, shuffled, no gold/model output) labelled
399 Reddit pool rows: per variety 150 uniform (U) + ~50 disputed (D: lead stk_v5r OOF contradicts gold sarcasm, margin > .8).
U stratum, Cohen's kappa [95% bootstrap CI]:
| | gold-lead | gold-A1/A2 | lead-A1/A2 | A1-A2 | positive rate gold/lead/A1/A2 |
| sarc AU | .56 [.43,.70] | .21 / .28 | .30 / .34 | .90 | .43/.43/.15/.17 |
| sarc UK | .59 [.44,.72] | .62 / .62 | .45 / .45 | .97 | .29/.23/.27/.27 |
| sent AU | .74 | .60 / .65 | .60 / .65 | .91 | .29/.33/.26/.27 |
| sent UK | .85 | .81 / .83 | .67 / .69 | .98 | .17/.15/.22/.21 |
D stratum (annotators agree on 45/50 AU, 48/49 UK): they side with the LEAD on 31 vs gold 14 (AU), 31 vs 17 (UK).
Pre-registered criteria: A1-A2 > gold-A (yes, both); lead-A >= lead-gold (NO, both varieties); annotators side with the
lead on > 60% of D (yes: 69% AU, 65% UK). Mixed -> read per variety:
- AU sarcasm: the gold annotator labels 43% of AU Reddit comments sarcastic vs ~16% for a careful independent reader;
  the lead reproduces the annotator's construct (kappa .56 with gold vs .30-.34 with the readers). In confident
  disagreements the readers side with the model 2:1 -> a real, annotator-specific noise floor.
- UK sarcasm: a generic reader recovers gold as well as the lead (.62 vs .59, CIs overlap) while agreeing only .45 with
  the lead -> evidence (weak) of recoverable UK sarcasm signal the models miss; not a clean ceiling.
- Caveat: A1/A2 are the same LLM (their agreement is closer to test-retest than to two humans); LLM readers may share
  the models' notion of sarcasm.

## R7. Ensemble-size curve (10 deberta seeds: s13-17 + retrains r13-17; results/paper/ensemble_curve.csv)
Mean over up to 30 random n-subsets (probability mean), official score: deberta alone n=1 .8604, 2 .8618, 3 .8632,
5 .8638, 10 .8637; inside the lead (logit mean with e5+me5 and 9B probe) n=1 .8669, 3 .8700, 5 .8705, 10 .8695.
- Annotation: seed averaging saturates by ~3 seeds (+.003); model-family diversity adds +.006 on top. Retrained
  seeds r13-17 (.867 x5) vs originals s13-17 (.863 x5): seed-draw noise ~.004 even for 5-seed blends.

## R8. LLM-reader sarcasm label as a feature for UK Reddit (pre-registered: experiments/prereg/reader_feature.md)
One Claude reader labelled all 1,143 UK Reddit pool rows (and 286 test rows) blind; cross-fitted LR on [lead logit, reader label].
- Pre-registered criterion FAILED: sarc-UK .843 vs .841 (p .52); score higher on 1/5 folds (0 change elsewhere: sarcasm is
  bound by AU in most folds). stk_v5r stays the lead.
- Descriptive: the reader label is heavily weighted (coef 1.5-1.7 vs lead logit .8) and improves UK-reddit sarcasm ranking
  AUC .906 -> .920, AP .761 -> .794 (4/5 folds), but the gains fall away from the F1 operating point.
- Annotation: a strong generic reader carries UK-sarcasm information the fine-tuned models lack (consistent with R6),
  but under the official min-over-varieties metric and F1 decisions it does not move the score.

## R9. Where AU sarcasm is hard, and what each annotator calls sarcasm (pool OOF, stk_v5r; gold labels)
- Reddit: P(sarc | gold negative) AU .60 vs UK .24; P(sarc | positive) .05 / .02; 97% (AU) / 99% (UK) of sarcastic rows
  are negative. Lead sarcasm AUC overall AU .866 / UK .906, WITHIN gold-negative rows AU .773 / UK .909; gold negativity
  alone AUC AU .745 / UK .565. -> AU sarcasm = "which negative comments did the annotator call sarcastic", where the lead is
  weak (.77); UK sarcasm is separable within negatives (.91).
- R6 sample: AU annotator labels 58% of negative comments sarcastic vs 21% for the readers; 44 of 64 gold-sarcastic AU items
  are reader-non-sarcastic (UK off-diagonals 13 vs 11).
- Univariate AUC of zero-shot NLI cues (MoritzLaurer deberta-v3-large-zeroshot-v2.0, label-free) for gold sarcasm WITHIN
  negative Reddit rows, AU / UK: mocking .68/.45, contempt .65/.36, angry .61/.32, cynical .60/.40, politics .58/.45;
  joking .65/.79, praise .58/.73, ironic praise .62/.71, joy (emotion model) .57/.73.
- Annotation: among negative comments the AU annotator's sarcasm tracks derision (mocking, contempt, anger, cynicism); the UK
  annotator's tracks playful irony with a positive surface, and hostility is evidence AGAINST sarcasm in UK. The cue signs
  flip, which explains the zero/negative cross-variety transfer (R2) mechanistically.
- Exploratory (10 feature sets tried, selection-biased): within AU negatives, lead + zs_nli .785 and lead + zs_affect .784 vs
  lead .771 (cross-fitted LR); other ling/ctx sets add nothing or hurt. Realistic version (all AU Reddit rows, gold
  sentiment unknown): AU-reddit AUC .866 -> .858-.863, score .869 -> .870 (noise): the cues only separate sarcasm once
  negativity is known; otherwise they re-encode negativity the lead already has.

## R10. The AU sarcasm construct also drives AU sentiment errors (stk_v5r OOF, segment+ukg decisions, Reddit rows)
- AU: 184 sentiment errors, 431 sarcasm errors, 75 on both; P(sarc error | sent error) .41 vs base .22 (UK .39 vs .12).
- 55/184 (30%) of AU sentiment errors are one pattern: gold sarcastic+negative, predicted sincere+positive (UK 4/36, 11%).
  Reading the most confident ones: sincere-looking recommendations / factual comments labelled sarcastic AND negative.
- Enforcing "predicted sarcastic -> negative" at decision time changes nothing (score .8693 -> .8691-.8696): the lead is
  already consistent; the coupled errors are items it reads as sincere.
- Annotation: both AU components (the binding ones on test) share one source of error, the AU annotator's broad sarcasm
  construct, which propagates to sentiment because sarcastic implies negative in 97% of AU gold rows.

## R11. Learning curves (results/paper/learning_curve_{deberta,probes}.csv; paper/fig_learning_curve.pdf)
Nested segment-stratified training subsets, full 5-fold OOF, official score (segment+ukg):
| share of training rows | 10% | 25% | 50% | 75% | 100% |
| DeBERTa (2 seeds; 100% = 10 seeds) | - | .828 | .858 | .859 | .860 |
| 9B probe (3 draws) | .833 | .846 | .853 | .854 | .853 |
| e5+me5 probe (3 draws) | .791 | .824 | .837 | .843 | .847 |
DeBERTa components (2 seeds; 100% = s13/s14): Reddit sarcasm AUC AU .821/.833/.845/.845, UK .778/.845/.856/.867;
AU Reddit sentiment AUC .929/.945/.946/.950.
- Annotation: the two strong members plateau from ~50% of the data (2.5k rows); the binding AU sarcasm component is flat
  from 75% in both families, while UK sarcasm (248 Reddit positives) is still rising in both. AU sarcasm is construct-limited
  (R6, R9), UK sarcasm data-limited. At 25% the frozen 9B probe beats fine-tuned DeBERTa.

## R12. DeBERTa architecture ablations (5-fold OOF, seeds 13+14 averaged, vs enc_deb_l_s13+s14 = .865; `validate compare`)
| ablation | score | sarc-UK | p(base better) | folds base wins |
| no [variety] [source] prefix (per-variety heads kept) | .857 | .811 | .84 | 4/5 |
| two single-task models (sentonly + sarconly) | .865 | .810 | .63 | 3/5 |
| one sarcasm head shared by both varieties | .863 | .803 | .74 (sarc-UK .78) | 4/5 |
- Single-seed reads were misleading (no-prefix looked like a tie at one seed: .860 vs .862).
- Annotation: the segment prefix is the one design choice with evidence (+.008); multi-task sharing is neutral despite the
  label coupling (R10); separate per-variety sarcasm heads mainly protect UK sarcasm (.824 vs .803), consistent with the
  two constructs (R9).

## R13. Test-set ablation: DeBERTa x5 alone (enc_deb_l_rx5, segment+ukg; test submission 3, 2026-09-24)
Test: sent-AU .9155, sent-UK .9750, sarc-AU .7927, sarc-UK .8195, score .8541 (OOF .867).
Ensemble (stk_v5r) test .8668 (OOF .869). Test gain of the two probes: +.0127; OOF paired expectation +.005
(90% band -.010 to +.023; validate compare stk_v5r enc_deb_l_rx5) -> consistent.
- The ensemble's test score tracks its OOF estimate (-.002); DeBERTa alone drops -.013, mostly on the binding AU
  components (sent-AU .931 -> .916, sarc-AU .803 -> .793). Matches the valid-subset diagnostic (DeBERTa alone .819 vs
  ensemble .851 on valid.csv rows).
- Submission 2 was an accidental duplicate of submission 1 (identical file, identical score).

## Corrections after internal review (2026-10-25; three reviewer sub-agents + a provenance audit)
These supersede the readings above where they conflict. The paper (paper/main.tex) uses the corrected versions.
- **R6 disputed (D) stratum is a direction artefact, not evidence that the model is right.** Split by direction, the
  readers side with whichever label says "not sarcastic": AU gold=1/lead=0: 28 lead : 4 gold; AU gold=0/lead=1: 3 lead :
  10 gold; UK gold=1/lead=0: 26 : 16; UK gold=0/lead=1: 5 : 1. U-stratum control (lead decision != gold): AU 18 lead : 13
  gold, UK 8 : 16. The D result reflects the readers' stricter threshold. The "2:1, real noise floor" reading is withdrawn.
- **R6 reader guideline is narrower than BESSTIE's** (requires non-literal meaning; plain complaints = 0; BESSTIE: "irony
  or mockery to express contempt or ridicule"). Disagreement with readers measures distance from that reading, not error.
  The readers are two runs of one LLM (Claude Opus 5.5), not independent. Pre-registered criterion 2 (lead-A >= lead-gold)
  failed in both varieties; now reported in the paper.
- **R6 unconditional rates** used as headline: AU gold .43 vs readers .16; UK .29 vs .27. Among reader-negative comments:
  AU .52 vs .22, UK .29 vs .32. Lead-gold AU kappa .56 vs BESSTIE second human .47 [.19,.72].
- **R9 cue analysis had a collider** (filtered on gold-negative). Re-run (experiments/classical/cue_flip.py) on comments a
  label-free sentiment model (SiEBERT) calls negative; results/paper/cue_flip.csv (auc = filtered, auc_all = unfiltered)
  and cue_flip_subreddit.csv. The flip survives: AU derision .66-.73 vs UK .38-.49; playful UK .68-.77 vs AU .50-.61;
  unfiltered AU derision .76-.80, UK .40-.51; within every subreddit (AU derision .62-.74; UK contempt/anger .33-.47).
  Cues are 8 interpretable picks from 18 NLI hypotheses + an emotion model: exploratory.
- **R10 retracted.** The "30% of AU sentiment errors" pattern is a base-rate artefact: AU sentiment error rate is 8.1% on
  sarcastic vs 8.0% overall, and the sarcastic share of errors (60%) equals the base rate (60%).
- **R11 learning curves now use matched seeds (13/14) at every share**: DeBERTa .858/.859/.862 at 50/75/100% (the old
  100% point used 10 seeds); paper/fig_gains.pdf. DeBERTa sarcasm AUC AU .833/.845/.845, UK .845/.856/.867.
- **R12**: no-prefix result is "suggestive" (P .84), not evidence; the shared-head hint is dropped.
- **R2 transfer**: BESSTIE already reported weak cross-variety sarcasm transfer; ours is "consistent with" it.
