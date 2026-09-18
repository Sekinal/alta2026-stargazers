# Runbook: when the test CSV is released

Everything below runs from the project root. `R` = run on the remote via
`scripts/remote_run.sh NAME "..."` (GPU jobs need `CUDA_VISIBLE_DEVICES=<free gpu>`).
Check `scripts/remote_run.sh --status` first.

## 0. Put the file in place (laptop and remote)
```
cp ~/Downloads/<test>.csv data/test.csv
uv run python -c "from alta.common import load_test; t=load_test('data/test.csv'); print(t.height, t.columns)"
scripts/remote_push.sh   # also syncs data/
```
`load_test` rejects unknown source/variety values, CR-only files and empty files, and warns on blank lines.

## 1. Clean every external corpus of test texts (remote) BEFORE any final training
```
R dedupe_test "PYTHONPATH=. uv run --with pyahocorasick python experiments/data/dedupe_sentiment.py --against data/test.csv data/external/*/*.parquet"
R dedupe_test_manifest "uv run python experiments/data/dedupe_indomain.py --in data/external/indomain --against data/test.csv"
# the first command checks every text column (text, parent_text, context, title, selftext);
# this second, independent implementation is a cross-check on the sarcasm sets:
R dedupe_test_sarc "uv run --with pyahocorasick python experiments/data/dedupe.py --test data/test.csv data/external/sarcasm/*.parquet"
```
Any model trained on external data before this step must be retrained if its corpus lost rows.

## 2. Test predictions per member (each saves with `save_preds(..., test_csv=data/test.csv)`)
- Encoders (remote GPU): `experiments/encoders/predict_test.py --run RUN --test data/test.csv`
  (averages the 5 fold checkpoints in models/RUN/).
- LLM (remote GPU): `experiments/llm/predict_test.py --run RUN --test data/test.csv`.
- Classical: first embed on the remote
  `R cls_test_emb "CUDA_VISIBLE_DEVICES=<gpu> uv run python experiments/classical/embed.py --test data/test.csv e5_large me5_sarc"`,
  wait for the 2-minute pull, then `uv run python experiments/classical/predict_test.py --run RUN --test data/test.csv`
  (runs stack members recursively).
- Feature sets (only if a backup stack uses them): `bash experiments/features/build_zs_all.sh data/test.csv`.

## 3. Rebuild the chosen blend so it carries the members' test predictions
```
uv run python -m alta.validate blend <LEAD> <members...> --method logit --force
uv run python -m alta.validate check <LEAD>        # must show the test split bound to data/test.csv
```

## 4. Build, check and inspect the submission
```
uv run python -m alta.validate submit <LEAD> --policy segment+ukg --test data/test.csv
```
This refuses runs that fail integrity checks, checks the format and row order against the test file, runs the official
scorer, and prints the shift diagnostics (segment shares vs valid, and segment+ukg rates vs pool prevalence).
If shift WARNs appear or the test has < 400 rows, stay on `segment+ukg` (not a prevalence policy).
Upload `submissions/test_<LEAD>_segment+ukg/answer.zip` (answer.csv at the zip root).

## Rules that still apply
- Never use BESSTIE or anything derived from it; never tune on the test file.
- Up to 3 test-phase submissions per team: the first is the lead; later ones only for principled alternatives.
