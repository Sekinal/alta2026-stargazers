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
- No literature review guided the design: the orchestrator stated so when asked (2026-09-19); the first literature
  search happened after the final system was chosen. Method knowledge came from the models' prior knowledge, the
  author's suggestions and the pasted recommendation.

## Phase 2 (2026-09-23/24)
- The GPU box holding all checkpoints was lost; the author provided a 1x RTX 3090 box; the lead was retrained
  (enc_deb_l_r13..17). Author constraint: no LLM larger than 4B for new work; focus on the dev side.
- Screens (EMA, LLRD, NLI-DeBERTa, ELECTRA, ModernBERT), thread-label features, kNN: no gain.
- A test-cache bug (LLM cache keyed by split, not test file) was found and fixed before any test prediction was made.
- Test submissions (uploaded by the author): stk_v5r 0.8668; an accidental duplicate (deleted by the organiser);
  DeBERTa x5 ablation 0.8541.
- Paper analyses: learning curves, ensemble-size curve, architecture ablations, blind LLM re-annotation (pre-registered),
  reader feature (pre-registered, failed), construct analysis (cue flip), coupled errors.
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
