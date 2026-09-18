# Feature store contract (feature-engineering agents -> classical/stacking agent)

- One parquet per feature set: `experiments/features/store/<set>__pool.parquet` with column `uid`
  (all 6288 pool uids, from `alta.common.load_pool()`) plus numeric feature columns prefixed `<set>__`.
- Features are LABEL-FREE: computed from text/source/variety only, or from models trained on OTHER
  data. Nothing may read `sentiment`/`sarcasm`. (Label-using features belong to the stacker's per-fold
  harness, not here.)
- Every set has a builder: `uv run python experiments/features/build_<set>.py --input CSV --out PARQUET`
  that works on the pool AND on a test CSV (`alta.common.load_test`, uids x00000..), deterministic.
- A sidecar `<set>.json`: feature list with one-line descriptions, model ids/versions, runtime, and
  the quick evaluation below.
- Quick evaluation, reported in the sidecar and to the classical agent: (1) LR on the set alone
  (5-fold OOF via alta.validation.harness.cv_fit_predict, segment+ukg score + per-fold);
  (2) incremental value: LR on [cls_e5me5_lr OOF logits + segment + this set] vs without the set,
  per-fold deltas and `validate compare`-style p. Worst slices: UK-reddit sarc, AU-reddit sent.
