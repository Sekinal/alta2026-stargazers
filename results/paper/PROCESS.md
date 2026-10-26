# Development process record (for the paper's process / AI-assistance section)

Compiled 2026-09-24 from the two Claude Code session transcripts of this project (mined by read-only agents) and the git
history. Facts only; paraphrases rather than quotes of the author's chat messages.

## Who did what
- **Author (Irving Ernesto Quezada Ramírez):** set goals and constraints, provided compute, uploaded submissions, made
  go/no-go decisions, suggested several methods (below), wrote/approved final text decisions.
- **Claude Code agents:** wrote all code, ran all experiments and analyses, drafted the paper.
  - Phase 1 (2026-09-17/19): one Claude Opus 5 orchestrator ("team lead") + 16 subagents (EDA x3, red team x3,
    modelling x3, data x4 incl. one sub-subagent, feature engineering x2, context recovery x1); ~135 inter-agent messages.
  - Phase 2 (2026-09-23/24): one Claude Opus 5.5 session + subagents used as blind re-annotators (8 on the 399-item
    sample, 15 on all UK Reddit rows) and 4 read-only transcript miners for this record.

## Author constraints and interventions (phase 1)
- Tooling: polars (never pandas), duckdb allowed, lets-plot for plots; multiple agents; git from the start.
- "Generalization first" as the overriding criterion -> selection charter (README): second untouched holdout, >= 3 seeds,
  worst-slice and perturbation checks, no selection on valid.csv or the dev leaderboard.
- Asked for a thorough validation stack and adversarial red-teaming of it.
- Compute: laptop RTX 4070 8 GB; A100 80 GB spot instance (pre-empted); 2x RTX 3090 on-demand box.
- LLMs: latest releases, Qwen3.5-4B named as the Pareto choice; local models only (no API LLMs as members).
- Suggested: gradient-boosted trees (XGBoost/LightGBM/CatBoost, CPU), TF-IDF, SVMs, embeddings as GBDT features, xRFM,
  PiSSA (later dropped, never run), Unsloth (used for LoRA speed), the EVA/DoRA/rsLoRA/Riemannian-LoRA ladder (from a
  ChatGPT-generated recommendation the author pasted), recipes from the author's own earlier project AstroCLIMB
  (bf16 LoRA r16 with a separate linear head; a pseudo-label "adaptation" stage).
- Protocol decision: single holdout (fold 0) screening for neural models, 5-fold for non-neural models.
- Asked for more (pre-)training data -> external corpora collection; later asked about literature.
- Set an unrealistic score goal (0.97); the agent declined to game it and explained the label-noise ceiling.

## Agent-originated decisions (phase 1)
- "Do not use BESSTIE" (the public benchmark the task data come from) in every agent prompt; the author did not object.
- The final ensemble, selected under the charter: DeBERTa-v3-large x5 (multi-task, per-variety sarcasm heads, variety/
  source prefix — head design and prefix suggested by the orchestrator's prompt), e5-large-v2 + multilingual-e5-large-
  instruct LR probe, Qwen3.5-9B NF4 hidden-state probe (the probe design is the agent's own), weight-free logit mean,
  cross-fitted per-segment thresholds.
- Where the design came from (corrected 2026-10-26): the author's Kaggle competition experience (validation-first
  selection, gradient-boosted trees and embedding features, the AstroCLIMB Kaggle system whose head and pseudo-label
  recipes were adapted) and a targeted search for methods to push performance before the system was fixed: the PiSSA
  paper (fetched at the author's request), a literature-backed ChatGPT recommendation of LoRA variants with paper links
  (EVA, DoRA, rsLoRA, Riemannian preconditioning, FARSS, Super-Tuning, HD-PiSSA), and xRFM. On 2026-09-19 the
  orchestrator said it had not itself searched prior work on BESSTIE or sarcasm detection; that search followed. Other
  method knowledge came from the models' prior knowledge.

## Phase 2 (2026-09-23/24)
- The GPU box holding all checkpoints was lost; the author provided a 1x RTX 3090 box; the lead was retrained
  (enc_deb_l_r13..17). Author constraint: no LLM larger than 4B for new work; focus on the dev side.
- Screens (EMA, LLRD, NLI-DeBERTa, ELECTRA, ModernBERT), thread-label features, kNN: no gain.
- A test-cache bug (LLM cache keyed by split, not test file) was found and fixed before any test prediction was made.
- Test submissions (uploaded by the author): stk_v5r 0.8668; an accidental duplicate (deleted by the organiser);
  DeBERTa x5 ablation 0.8541.
- Paper analyses: learning curves, ensemble-size curve, architecture ablations, blind LLM re-annotation (pre-registered),
  reader feature (pre-registered, failed), construct analysis (cue flip), coupled errors (later retracted, see below).
- Author decisions: no paid human annotation study; paper figure/table choices; release plan (private HF repo).
- Release: text scan with positive control; one tracked EDA table containing competition text found, removed and the
  git history rewritten.

## Mistakes found and corrected (both phases)
Red-team findings: test predictions not bound to their test file; submit not running integrity checks; unvalidated
source/variety values; a realistic feature-selection leak passing all checks (-> permutation canary); plain KFold not
detected. Others: raw competition text committed early (removed); a gitignore rule hiding experiments/data/; a polars
row-order bug in pseudo-label sampling; DAPT starting from a random MLM head; EVA initialisation not converging;
mixed logit/probability averaging in a thread-gap table; a canary value reused across seeds; an overstated claim of
leaderboard contamination (retracted); the LLM test-cache collision (phase 2); relative path given for an upload (phase 2,
cost one submission, restored by the organiser).

## External data (phase 1; none in the final system)
Sarcasm: Reddit "/s" AU/UK (Arctic Shift), SARC 2.0, EPIC, iSarcasmEval, SemEval-2018 Task 3, FigLang 2020, Wallace
et al. 2014, MUStARD, News Headlines. Sentiment: GoEmotions, TweetEval, DynaSent, SST-2, Rotten Tomatoes, IMDB, Yelp and
Amazon polarity, Sentiment140 (+ QUT election data and Google Local, excluded). In-domain unlabelled (~236M tokens):
Reddit AU/UK (Arctic Shift), Google Local 2018, Booking.com, NHS GP reviews, Inside Airbnb. All deduplicated against the
labelled pool (and scripted against the test file). Used only in pre-training / pseudo-labelling screens that failed.
Thread context (audit only): Arctic Shift day dumps (95% of Reddit rows matched).

## Phase 2b (2026-09-25 .. 2026-10-25): paper
- Verified bibliography (paper/build_bib.py): 95 entries, 82 retrieved and title/author-checked against Crossref, the
  ACL Anthology or official arXiv BibTeX; 13 entered by hand (software, model cards, web pages, papers without DOI/arXiv).
- Literature: after the system was fixed the agent read the BESSTIE paper and related work; none of it changed the
  submitted system.
- Internal review (three Claude Opus 5.5 reviewer sub-agents + one provenance-audit sub-agent) found two base-rate
  artefacts (R10 coupled errors; R6 disputed-item direction), a collider in the cue analysis, mismatched learning-curve
  seeds and an unreported failed pre-registered criterion. All corrected (results/paper/NOTES.md, last section).
- Pre-registration sequence for R6/R8: the documents were written before annotation started, but committed together with
  the results (commits b1b89dd/0cb9b59, 181280b/2ddd699, minutes apart); the order rests on the session record.
- Phase-1 session transcripts were later removed by the tool's retention policy; this record relies on extracts made
  while they existed. Readers in R6/R8 were Claude Opus 5.5 sub-agents.
- Acknowledgements: Lium GPU-credit grant; Claude Code with Claude Opus 5 and Opus 5.5. The author must review the paper
  before submission (responsibility statement in the acknowledgements).

## Everything explored but not used in any reported result (moved out of the paper, 2026-10-26)
- Encoders: RoBERTa-large (Liu et al. 2019); Twitter-RoBERTa models (Barbieri et al. 2020, TweetEval; Loureiro et al.
  2022, TimeLMs; Loureiro et al. 2023, Tweet Insights), incl. cardiffnlp/twitter-roberta-large-2022-154m.
- Embedding models: intfloat/e5-base-v2; BAAI/bge-large-en-v1.5 (Xiao et al. 2024, C-Pack); thenlper/gte-large
  (Li et al. 2023); mixedbread-ai/mxbai-embed-large-v1; sentence-transformers/all-mpnet-base-v2 (Reimers & Gurevych
  2019); Qwen3-Embedding 0.6B/4B (Zhang et al. 2025).
- LLMs: Qwen3.5-4B probes and AstroCLIMB-recipe LoRA runs (its plain LoRA ladder IS reported in Table 3); Qwen3-8B 4-bit (Qwen3 technical report 2025); Qwen2.5-1.5B for surprisal (2024).
- Zero-shot feature models: facebook/bart-large-mnli (Lewis et al. 2020; Williams et al. 2018); SiEBERT as a stacker
  feature (SiEBERT IS used in the reported cue analysis as the sentiment filter); nlptown star
  rating; cardiffnlp irony, Elron irony, w11wo iSarcasm classifiers; SamLowe go_emotions; unitary toxicity; a formality
  classifier; cardiffnlp tweet-topic (Antypas et al. 2022).
- Classical: TF-IDF + LR / SVM, NB-SVM (Wang & Manning 2012), feature augmentation (Daume III 2007), XGBoost
  (Chen & Guestrin 2016), LightGBM (Ke et al. 2017), CatBoost (Prokhorenkova et al. 2018), xRFM (Beaglehole et al. 2025).
- Data-exploration tools: VADER (Hutto & Gilbert 2014), NRC Emotion Lexicon (Mohammad & Turney 2013), weighted
  log-odds (Monroe et al. 2008), confident learning (Northcutt et al. 2021), UMAP (McInnes et al. 2018).
- Corpora collected but not used in any reported run (verified against experiments/encoders/pretrain_sarc.py):
  SARC (Khodak et al. 2018), MUStARD (Castro et al. 2019), News
  Headlines sarcasm (Misra & Arora 2023); sentiment: GoEmotions (Demszky et al. 2020), TweetEval sentiment, DynaSent
  (Potts et al. 2021), SST-2 (Socher et al. 2013), Rotten Tomatoes (Pang & Lee 2005), IMDB (Maas et al. 2011), Yelp and
  Amazon polarity (Zhang et al. 2015), Sentiment140 (Go et al. 2009); QUT AusVotes2025 (collected, excluded).
Full BibTeX for all of these is in paper/references.bib (verified; see paper/bib_report.txt).
