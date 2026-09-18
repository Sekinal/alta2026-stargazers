# EDA: semantic geometry, label noise and difficulty

Numbers are on the pool of 6,288 rows (train + valid), 5-fold out-of-fold on the shared folds, threshold 0.5 unless stated.
Run in order with `uv run python eda/semantic/0X_*.py`:

| script | what it does |
|---|---|
| `01_embed.py` | embeddings from bge-large, bge-base, all-mpnet, all-MiniLM (cached in `cache/`, git-ignored) |
| `02_probes.py` | logistic-regression probes on the embeddings, 5 ways of splitting the training data |
| `03_umap.py` | 2-D UMAP maps |
| `04_noise.py` | label-noise checks (confident learning, kNN agreement, near-duplicates) |
| `05_difficulty.py` | calibration, thresholds, hard examples, slices |
| `06_transfer.py` | learning curves and cross-variety transfer |

Saved predictions: `preds/emb_{bge_large,bge_base,mpnet,minilm}_lr.parquet` and `preds/emb_ens3_lr.parquet`.

Terms:
- **Probe:** logistic regression trained on frozen sentence embeddings.
- **ens3:** average of the bge-large, bge-base and mpnet probes.
- **Neighbours:** a row's 10 nearest neighbours in bge-large space.
- **Confident learning (CL):** flags rows whose OOF probability for the other class clears that class's per-segment threshold.

## Findings

### 1. The frozen-embedding baseline is strong

| model | rows | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|---|---|---:|---:|---:|---:|---:|
| bge-large, one probe per variety × source | pool | .893 | .958 | .728 | .702 | .798 |
| **ens3** | pool | .902 | .960 | .738 | .721 | **.811** |
| **ens3** | valid only | .922 | .953 | .749 | .704 | **.813** |
| organisers' baseline (distilbert) | valid only | .911 | .953 | .714 | .637 | .774 |

- Only 28 texts exceed 512 tokens; those were truncated keeping the start and end.
- The score varies by 0.008–0.010 (SD) across folds, so gaps under about 0.01 are noise.
- The coordinator re-scored the saved predictions with `alta.common.evaluate_oof` and got the same numbers.

### 2. Sarcasm doesn't transfer between varieties; sentiment does
Plot: `learning_curves.png`.

- Training only on the other variety costs about 1 point of sentiment F1 but 10–11 points of sarcasm F1: UK sarcasm drops from .683 to .577, AU from .746 to .631.
- Adding the other variety's data hurts sarcasm at every training fraction:

  | target | 25% own data | 100% own data |
  |---|---:|---:|
  | UK, own only | .657 | .683 |
  | UK, own + AU | .606 | .644 |
  | AU, own only | .729 | .747 |
  | AU, own + UK | .686 | .732 |

- For sentiment, adding the other variety is neutral or slightly positive (AU +0.002 to +0.008).
- UK sarcasm plateaus from 50% of its own data (.680 at 50%, .683 at 100%).
- Cosine between the AU and UK probe weight vectors: sarcasm 0.21, sentiment 0.37. Between the sarcasm and sentiment vectors: AU −0.42, UK −0.21.

### 3. In embedding space, source is the main split
Plots: `umap_bge_large_source.png`, `umap_bge_large_variety.png` (and the `umap_mpnet_*` equivalents).

- Google reviews and Reddit posts form separate regions.
- In Google, AU and UK are fully mixed: for an AU review, 36% of its neighbours are AU, against 34% by chance.
- In Reddit they separate by topic. The same-variety neighbour share is 79% for AU (chance 64%) and 66% for UK (chance 36%).

### 4. Sarcasm clusters for AU but is scattered for UK
Plots: `umap_bge_large_joint.png`, `umap_bge_large_sarcasm.png`, `knn_agreement.png`.

- For a sarcastic AU post, 54% of its neighbours are sarcastic (base rate 42%).
- For a sarcastic UK post it's 18%, below the 21.7% base rate.
- UK Google has exactly 1 sarcastic row out of 2,065, so UK sarcasm is effectively a Reddit-only task.

### 5. AU sarcasm labels largely mean "negative Reddit post"

| Reddit | P(sarcastic given negative) | P(negative given sarcastic) |
|---|---:|---:|
| AU | 0.595 | 0.966 |
| UK | 0.242 | 0.988 |

- The rule "Reddit post with negative gold sentiment ⇒ sarcastic" gets AU sarcasm F1 **.772**, beating every AU probe (.747). For UK it gets only .620.
- With predicted sentiment instead of gold: AU .733, UK .601.
- Blend 0.75·p_sarc + 0.25·reddit·(1 − p_sent): AU .754 vs .738 (threshold tuned in-sample, so optimistic); UK .713 vs .721, no gain.

### 6. Label noise is modest and concentrated in AU-reddit sarcasm

Confident-learning flags per segment:

| segment | n | sentiment flags | sarcasm flags |
|---|---:|---:|---:|
| AU-google | 1,076 | 6 (0.6%) | 83 (7.7%) |
| AU-reddit | 2,004 | 73 (3.6%) | 312 (15.6%) |
| UK-google | 2,065 | 9 (0.4%) | 0 |
| UK-reddit | 1,143 | 12 (1.0%) | 217 (19.0%) |

The probe is weak, so most sarcasm flags are model errors rather than label errors. A random sample was read by hand:
- **UK-reddit, 30 flags:** about 1 clear mislabel and about 5 ambiguous. In the other 24 the label looks right and the model is wrong (angry posts that aren't sarcastic; obvious deadpan sarcasm it missed).
- **AU-reddit, 20 flags:** about 6 clear mislabels, all labelled sarcastic on sincere critical posts, and about 1 ambiguous.

Estimated real noise:
- Sarcasm: about 5% of AU-reddit rows (mostly sincere posts labelled sarcastic, about 7% of AU sarcastic labels); about 1–3% of UK-reddit rows.
- Sentiment: about 0.5% for Google (mostly mixed reviews), 1–3% for Reddit.
- Near-duplicates: 20 pairs with cosine ≥ 0.95 (14 byte-identical), none with conflicting labels.

**Ceiling:** label noise alone would cap sarcasm macro-F1 at about 0.95–0.97, so noise is not the limit. The limit is ambiguity from missing thread context.

Suspected mislabels are listed by uid only (look the text up with `load_pool()`). "Same-label neighbours" is the share of the 10 nearest neighbours with the same gold label; "verdict" is from reading the text.

**Sentiment:**

| uid | segment | gold | P(other) | same-label neighbours | verdict |
|---|---|---:|---:|---:|---|
| t02721 | UK-google | 0 | 1.00 | 0.0 | mislabel |
| t03624 | UK-google | 1 | 0.99 | 0.4 | mislabel |
| t05291 | UK-reddit | 1 | 0.98 | 0.0 | ambiguous |
| t04714 | UK-reddit | 0 | 0.97 | 0.5 | label OK (sarcastic, so negative) |
| t05426 | UK-reddit | 1 | 0.97 | 0.0 | ambiguous |
| t02346 | AU-reddit | 1 | 0.97 | 0.1 | ambiguous |
| t01100 | AU-reddit | 1 | 0.97 | 0.1 | mislabel |
| t03659 | UK-google | 1 | 0.96 | 0.1 | mixed, leans mislabel |
| t03227 | UK-google | 1 | 0.96 | 0.0 | mislabel |
| t00153 | AU-google | 1 | 0.96 | 0.0 | mixed review |
| t01963 | AU-reddit | 0 | 0.96 | 0.0 | mislabel |
| t00181 | AU-google | 1 | 0.96 | 0.1 | label OK |
| t02235 | AU-reddit | 1 | 0.95 | 0.0 | label OK |
| t01293 | AU-reddit | 0 | 0.95 | 0.2 | leans mislabel |

**Sarcasm:**

| uid | segment | gold | P(other) | same-label neighbours | verdict |
|---|---|---:|---:|---:|---|
| t00172 | AU-google | 1 | 0.97 | 0.0 | ambiguous |
| t02005 | AU-reddit | 1 | 0.96 | 0.2 | mislabel |
| t00663 | AU-google | 1 | 0.96 | 0.0 | mislabel |
| v00157 | AU-reddit | 0 | 0.95 | 0.4 | label OK |
| t00268 | AU-google | 0 | 0.95 | 0.5 | ambiguous (joking) |
| t00932 | AU-google | 0 | 0.94 | 0.5 | label OK (model reads negative as sarcastic) |
| t01597 | AU-reddit | 0 | 0.94 | 0.2 | label OK |
| t05148 | UK-reddit | 1 | 0.93 | 0.0 | label OK (scare quotes) |
| t02366 | AU-reddit | 0 | 0.93 | 0.3 | ambiguous |
| t01282 | AU-reddit | 1 | 0.92 | 0.6 | mislabel |
| t01046 | AU-reddit | 0 | 0.92 | 0.5 | ambiguous, leans sarcastic |
| t01579 | AU-reddit | 0 | 0.92 | 0.5 | label OK |
| t02463 | AU-reddit | 0 | 0.92 | 0.2 | label OK |
| t05139 | UK-reddit | 1 | 0.91 | 0.0 | label OK (model missed it) |
| v00077 | AU-google | 0 | 0.91 | 0.5 | label OK |
| t05164 | UK-reddit | 1 | 0.91 | 0.0 | ambiguous |

Other UK-reddit rows judged arguably mislabelled: t04713 (gold 1, P(not sarcastic) 0.79, looks sincere); t05272 and t05451 (gold 0, P(sarcastic) 0.91 and 0.86, look sarcastic).

### 7. Calibration and thresholds
Plots: `calibration.png`, `threshold_curves.png`, `pr_sarcasm_reddit.png`.

**Calibration:**
- The class-balanced probe over-predicts sarcasm. Mean P(sarc) vs base rate: AU 0.371 vs 0.297; UK 0.137 vs 0.078.
- Brier score: sarcasm AU .149, UK .063; sentiment AU .075, UK .035.
- UK is worse calibrated: around p ≈ 0.45 only 21% of UK rows are sarcastic, against 39% of AU rows.
- Sentiment has an S-shaped reliability curve (under-confident at both ends) but ranks well.

**F1-optimal thresholds:**
- Over all OOF rows: sarcasm AU 0.46 (F1 .742 vs .738 at 0.5), UK 0.48 (.722 vs .721); sentiment AU 0.57 (.905 vs .902), UK 0.51 (.961 vs .960).
- Per-fold optimum:

  | task | variety | mean | SD | range |
  |---|---|---:|---:|---|
  | sarcasm | AU | 0.51 | 0.046 | 0.46–0.58 |
  | sarcasm | UK | 0.51 | 0.049 | 0.44–0.56 |
  | sentiment | AU | 0.57 | 0.049 | 0.51–0.62 |
  | sentiment | UK | 0.54 | 0.063 | 0.47–0.62 |

- The F1 curve is flat within about ±0.1 of the optimum. Across folds, the at-0.5 F1 has SD sarc-AU .022, sarc-UK .015, sent-AU .019, sent-UK .006.
- Once the model is class-balanced, AU and UK optimal thresholds don't differ.

**Nested threshold tuning** (tune on 4 folds, apply to the 5th): tuning gives no gain.

| thresholds | score |
|---|---|
| fixed 0.5 | 0.807 ± 0.008 |
| one global tuned threshold | 0.807 ± 0.010 |
| per-variety tuned thresholds | 0.805 ± 0.007 |

**Sarcasm precision-recall on Reddit:** AU precision stays at or above 0.65 up to recall 0.7. UK precision is below 0.55 from recall 0.2 onwards: the UK probe has no high-confidence sarcasm region.

**Share predicted sarcastic by gold class (Reddit):**

| variety | neg, not sarcastic | neg, sarcastic | pos, not sarcastic |
|---|---|---|---|
| AU | 50% (n=552) | 72% (n=812) | 10% (n=611) |
| UK | 17% (n=767) | 56% (n=245) | 34% (n=128) |

### 8. Hard UK-reddit examples and slices
Plot: `sarcasm_by_length.png`.

- **False negatives** (gold 1, p < 0.16): t05148, t05139, t05164, v00706, t05143, t05351, t04586, t04832, t04570, v00698, t05480, t05022, v00730, t04697, t05431. Mostly deadpan or mock outrage with no lexical cue. Some gold positives are hard to read as sarcasm at all (t05143, t04586).
- **False positives** (gold 0, p > 0.75): t05272, t04748, t05330, t04549, v00751, t05451, t05455, t05324, t04829, t04856, t04547, t05106, t05059, t05202, t04550. Mostly angry political one-liners; t05272 and t05451 look genuinely sarcastic.

Reddit sarcasm F1 at 0.5 (AU / UK) by length:

| words | F1 AU / UK | n AU / UK |
|---|---|---|
| ≤ 10 | .668 / .601 | 365 / 219 |
| 11–30 | .723 / .706 | 717 / 429 |
| 31–80 | .732 / .603 | 602 / 385 |
| > 80 | .677 / .607 | 320 / 110 (only 8% of these UK rows are sarcastic) |

- Scare quotes: sarcasm rate AU .57 vs .40, UK .29 vs .21, yet F1 is worse (AU .62 vs .72, UK .59 vs .67).
- Lines starting with ">" (78 AU / 32 UK rows): sarcasm rate AU .58 vs .41, UK .31 vs .21.
- Questions: sarcasm rate AU .57 vs .39; F1 unchanged.
- Short context-free replies are the least knowable without the parent comment.

## Implications for modelling

1. **ens3 is the bar:** 0.811 OOF, 0.813 on valid.csv, against the baseline's 0.774, with no fine-tuning. Keep it as an ensemble or stacking member. A fine-tuned model has to beat 0.81 to be worth its cost.
2. **Don't pool varieties for sarcasm.** Pooling costs UK about 4 points, and cross-variety transfer costs about 10. Use per-variety sarcasm models, or a shared encoder with variety-conditioned sarcasm heads. Sentiment can be shared. Choose checkpoints on min(sarc-AU, sarc-UK).
3. **Split by source too.** Per-segment probes are best for the bge models (.798 vs .790 per-variety). Hard-code UK-google sarcasm to 0 and train and evaluate UK sarcasm on UK-reddit only.
4. **Use sentiment as a feature for AU sarcasm only.** AU labels encode "negative Reddit post ⇒ often sarcastic" (P = .60 vs .24 for UK). Use multi-task training, stack p_sent into the sarcasm model, or the blend (about +0.016 AU). It doesn't help UK.
5. **Skip per-variety threshold tuning for class-balanced models** (nested CV −0.002). If tuning, use OOF across all 5 folds; the per-fold optimum moves by about ±0.05.
6. **Treat gains under about 0.01 as noise.** Fold SD is 0.008–0.010 on the score and 0.015–0.02 on sarcasm.
7. **Handle AU-reddit sarcasm noise softly.** Noise there is about 5%, mostly over-labelled positives. Use label smoothing (about 0.05–0.1) or down-weight CL-flagged rows. Don't drop flagged rows: most flags are model errors, and 80–95% of UK flags have correct labels.
8. **Ceiling:** noise alone would allow about 0.95 sarcasm F1, but missing parent-thread context makes some UK-reddit sarcasm unknowable. A realistic sarc-UK target with a strong fine-tuned model or LLM is about 0.75–0.80, up from .72. LLMs that know UK politics (Tories, NHS) are the most likely source of gains on deadpan examples.
