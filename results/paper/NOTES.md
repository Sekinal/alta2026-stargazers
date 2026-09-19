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
