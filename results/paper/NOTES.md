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
