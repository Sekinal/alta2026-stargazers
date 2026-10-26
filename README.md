# Team Stargazers (CodaBench: thermostatic) — ALTA 2026 Shared Task (sentiment & sarcasm in en-AU / en-UK)

**Release with predictions, cached features and the 25 model checkpoints:** https://huggingface.co/Thermostatic/alta2026-thermostatic. This GitHub repository holds the code, results and paper; large artefacts (`preds/`, caches, `models/`) are only on Hugging Face.

Code and paper for our system in the
[ALTA 2026 shared task](https://www.alta.asn.au/events/sharedtask2026/) on sentiment and sarcasm classification in
Australian and British English (BESSTIE en-AU / en-UK subsets).

Paper: *Two Varieties, Two Sarcasms: What Limits a Well-Validated Ensemble on Australian and British
English* — Irving Ernesto Quezada Ramírez (`paper/main.tex`, `paper/main.pdf`).

## Results

Official score = ½ [min(sent-AU, sent-UK) + min(sarc-AU, sarc-UK)], macro-F1 per component.

| System | sent-AU | sent-UK | sarc-AU | sarc-UK | score |
|---|---|---|---|---|---|
| Ensemble `stk_v5r`, pooled 5-fold OOF (dev estimate) | .934 | .976 | .805 | .841 | .869 |
| **Ensemble `stk_v5r`, official test** (2nd on the leaderboard; one of six runs not significantly different from the top run) | .932 | .976 | .801 | .837 | **.867** |
| DeBERTa-v3-large ×5 alone, official test (ablation) | .916 | .975 | .793 | .820 | .854 |
| Organisers' baseline, official test | .866 | .931 | .680 | .696 | .773 |

The ensemble is a weight-free logit mean of three members, with decision thresholds cross-fitted per task × segment:

1. `enc_deb_l_rx5` — DeBERTa-v3-large, multi-task (one sentiment head, one sarcasm head per variety), input prefixed with
   `[variety] [source]`, 4 epochs, lr 1e-5, batch 16; 5 seeds (13–17) × 5 CV folds = 25 checkpoints (`models/`).
2. `cls_e5me5_lr` — logistic regression over frozen e5-large-v2 + multilingual-e5-large-instruct embeddings.
3. `llm_q35_9b_probe` — logistic-regression probe over Qwen3.5-9B (NF4) layer-24 hidden states under zero-shot prompts.

Everything else we tried (other backbones, EMA/LLRD, LoRA-tuned LLMs, pseudo-labels, thread context, kNN, LLM-reader
features, ...) and the analyses behind the paper are logged in `results/paper/NOTES.md` (R1–R13).

## What is (and is not) in this repository

**Included:** all code (`alta/`, `experiments/`, `scripts/`, `tests/`, `eda/` code), run predictions (`preds/`: uid →
probabilities, OOF and test, with JSON metadata), cached label-free features (`experiments/*/cache`, `experiments/features/store`:
embeddings, LLM hidden states and margins, zero-shot features), validation reports and the experiment ledger (`results/`),
the paper and its tables (`paper/`, `results/paper/`), pre-registrations (`experiments/prereg/`), blind re-annotation
labels without text (`experiments/reannot/labels*`), and the 25 DeBERTa checkpoints (`models/enc_deb_l_r{13..17}/fold{0..4}/`).

**Not included (not redistributable):** the competition data (train/valid/test CSVs, gold labels), any raw text from it
(EDA examples, submission `answer.csv` files, perturbed/fake test CSVs), recovered Reddit context, and third-party external
corpora. Every published file was scanned for competition texts before upload. Get the data from the organisers
(CodaBench competition 17681) and place it as described below.

## Reproducing

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), one 24 GB GPU for the neural members (we used an RTX 3090).

```bash
uv sync
# data from the organisers
cp train.csv valid.csv data/            # pool = train + valid
cp <test>.csv data/test.csv
uv run python -m alta.common            # builds data/pool.parquet and the fixed 5 folds (fold hash b9ba64c66508)
```

### Test predictions from the released checkpoints (no training)
```bash
for s in 13 14 15 16 17; do
  uv run python -m experiments.encoders.predict_test --run enc_deb_l_r$s --test data/test.csv   # averages fold0..4
done
uv run python experiments/llm/predict_test.py --run llm_q35_9b_probe --test data/test.csv        # scores test with Qwen3.5-9B
uv run python experiments/classical/embed.py --test data/test.csv e5_large me5_sarc
uv run python experiments/classical/predict_test.py --run cls_e5me5_lr --test data/test.csv
uv run python -m alta.validate blend enc_deb_l_rx5 enc_deb_l_r13 enc_deb_l_r14 enc_deb_l_r15 enc_deb_l_r16 enc_deb_l_r17 --method mean --force
uv run python -m alta.validate blend stk_v5r enc_deb_l_rx5 cls_e5me5_lr llm_q35_9b_probe --method logit --force
uv run python -m alta.validate submit stk_v5r --policy segment+ukg --test data/test.csv
```

### Retraining from scratch
`bash experiments/encoders/queue_retrain.sh data/test.csv` retrains the 25 DeBERTa fold models (~2.5 h on one 3090)
and scores the LLM probe and embedding member on the test file. `RUNBOOK_TEST.md` has the full test-day procedure.
Checkpoints load with `experiments.encoders.common.load_fold(path)` (bf16 state dict in `weights.pt` + config + tokenizer).

### Paper analyses
| Analysis | Command |
|---|---|
| Learning curves (DeBERTa) | `bash experiments/encoders/queue_paper2.sh` (GPU) |
| Learning curves (probes) | `uv run python experiments/classical/learning_curve.py` |
| Ensemble-size curve | `uv run python experiments/classical/ensemble_curve.py` |
| Figure | `uv run python experiments/classical/plot_paper.py` |
| Blind re-annotation | `experiments/reannot/build_sample.py` → annotate → `experiments/reannot/analyze.py --ann-dir experiments/reannot/labels` |
| Thread-label features | `uv run python experiments/classical/thread_labels.py --base stk_v5r --run thr` |
| Model comparisons | `uv run python -m alta.validate compare RUN_A RUN_B` |

The blind re-annotation labels were produced by Claude subagents from the texts alone; the item → uid mapping is rebuilt
deterministically by `build_sample.py` from the (non-redistributable) pool.

## Validation protocol (short)
Pooled 5-fold CV over train+valid (stratified by segment and labels, identical texts grouped); decision thresholds
cross-fitted; candidates compared by paired test-sized bootstrap (p < 0.8 = no evidence); permutation canaries for leaks;
fold-0 screens confirmed on an untouched fold under written pre-registration. Details in the paper and `README` of the
project (`PROJECT_README.md`).

## License
Code: MIT. Model checkpoints and predictions are derived from the BESSTIE data used in the shared task and are released
for research reproducibility; check the BESSTIE terms before other uses.

## Citation
```bibtex
@inproceedings{quezada-2026-thermostatic,
  title     = {Two Varieties, Two Sarcasms: What Limits a Well-Validated Ensemble on Australian and British English},
  author    = {Quezada Ram{\'i}rez, Irving Ernesto},
  booktitle = {Proceedings of the 24th Annual Workshop of the Australasian Language Technology Association (ALTA 2026)},
  year      = {2026},
  note      = {Shared task system description}
}
```
