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

## 1b. Recover Reddit context for the test rows (only if a context-aware member is in the lead)
```
R ctx_test "uv run --with orjson python experiments/data/context_match.py match --input data/test.csv --out data/external/context/test_context.parquet"
# needs internet (~1 min). If coverage is low, fetch the missing days: context_match.py fetch --subs ... --start ... --end ...,
# then index --src data/external/raw/context_days --out data/external/context/index/days, then match again.
```
Keyed by test uid (x00000...). Check the match rate against the pool's (AU ~92%, UK ~56%+); a much lower rate means
the test came from a different crawl, in which case the context-aware members fall back to their no-context prompts.

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

## Current lead: `stk_v5` (0.874 OOF, segment+ukg), verified end to end on the fake test
Weight-free logit mean of `enc_deb_l_x5` (deberta-v3-large, seeds 13-17), `cls_e5me5_lr`, `llm_q35_9b_probe`.
(Replaced `stk_v4` / `enc_deb_l_x3`: +0.002, positive on 5/5 folds, lower variance.)
```
# encoders (remote GPU): each seed averages its 5 fold checkpoints
for s in 13 14 15 16 17; do R enc_test_$s "CUDA_VISIBLE_DEVICES=0 uv run python experiments/encoders/predict_test.py --run enc_deb_l_s$s --test data/test.csv"; done
# LLM probe (remote GPU)
R llm_test "CUDA_VISIBLE_DEVICES=1 uv run python experiments/llm/predict_test.py --run llm_q35_9b_probe --test data/test.csv"
# classical: embed on the remote, then predict (laptop or remote)
R cls_test_emb "CUDA_VISIBLE_DEVICES=0 uv run python experiments/classical/embed.py --test data/test.csv e5_large me5_sarc"
uv run python experiments/classical/predict_test.py --run cls_e5me5_lr --test data/test.csv
# blends: ONLY via validate blend (it writes the metadata + test binding)
uv run python -m alta.validate blend enc_deb_l_x5 enc_deb_l_s13 enc_deb_l_s14 enc_deb_l_s15 enc_deb_l_s16 enc_deb_l_s17 --method mean --force
uv run python -m alta.validate blend stk_v5 enc_deb_l_x5 cls_e5me5_lr llm_q35_9b_probe --method logit --force
uv run python -m alta.validate check stk_v5
uv run python -m alta.validate submit stk_v5 --policy segment+ukg --test data/test.csv
```
Expect the organisers' dev board (valid.csv) to show ~0.856 for this lead: noise from valid's 30 UK sarcastic rows.

## Rules that still apply
- Never use BESSTIE or anything derived from it; never tune on the test file.
- Up to 3 test-phase submissions per team: the first is the lead; later ones only for principled alternatives.
