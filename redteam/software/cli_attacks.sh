#!/usr/bin/env bash
# Attack 5: every CLI subcommand with bad arguments; prints exit code + last stderr line.
# Run from project root: bash redteam/software/cli_attacks.sh
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
V="uv run python -m alta.validate"
t() {
  local desc="$1"; shift
  local out; out=$("$@" 2>&1); local rc=$?
  printf '%-58s rc=%-3s %s\n' "$desc" "$rc" "$(echo "$out" | grep -v -i warning | tail -1 | cut -c1-120)"
}
# a run whose checks FAIL: OOF truncated
uv run python - <<'EOF'
import polars as pl, json, shutil
p = pl.read_parquet("preds/emb_ens3_lr.parquet")
p.head(100).write_parquet("preds/rt_sw_broken.parquet")
shutil.copy("preds/emb_ens3_lr.json", "preds/rt_sw_broken.json")
EOF
t "check missing run"                  $V check nope
t "check FAILing run"                  $V check rt_sw_broken
t "report FAILing run --no-write"      $V report rt_sw_broken --no-write
t "report bad policy"                  $V report emb_minilm_lr --policies bogus --no-write
t "report empty policies"              $V report emb_minilm_lr --policies '' --no-write
t "report --B 1"                       $V report emb_minilm_lr --policies fixed --B 1 --no-write
t "report --B 0"                       $V report emb_minilm_lr --policies fixed --B 0 --no-write
t "compare missing run"                $V compare emb_minilm_lr nope
t "compare FAILing run (no checks)"    $V compare emb_minilm_lr rt_sw_broken --B 10
t "compare bad policy"                 $V compare emb_minilm_lr tfidf_lr --policy xx
t "blend duplicate member"             $V blend rt_sw_b emb_minilm_lr emb_minilm_lr
t "blend FAILing member"               $V blend rt_sw_b2 emb_minilm_lr rt_sw_broken
t "check blend of FAILing member"      $V check rt_sw_b2
t "stack FAILing member"               $V stack rt_sw_s emb_minilm_lr rt_sw_broken
t "submit no policy"                   $V submit emb_minilm_lr --dev
t "submit bad policy"                  $V submit emb_minilm_lr --policy nope --dev
t "submit FAILing run --dev"           $V submit rt_sw_broken --policy fixed --dev
t "submit --test missing file"         $V submit emb_minilm_lr --policy fixed --test /nope.csv
t "submit --test (run has no test)"    $V submit emb_minilm_lr --policy fixed --test redteam/software/fake/test.csv
t "check-submission missing zip"       $V check-submission /nope.zip --test data/valid.csv
t "check-submission not a zip"         $V check-submission data/valid.csv --test data/valid.csv
t "check-submission wrong test file"   $V check-submission "submissions/dev_emb_ens3_lr_fixed+ukg/answer.zip" --test redteam/software/fake/test.csv
t "check-submission truth w/o labels"  $V check-submission "submissions/dev_emb_ens3_lr_fixed+ukg/answer.zip" --test data/valid.csv --truth redteam/software/fake/test.csv
t "board"                              $V board
t "no subcommand"                      $V
rm -f preds/rt_sw_broken.* preds/rt_sw_b.* preds/rt_sw_b2.* preds/rt_sw_s.*
