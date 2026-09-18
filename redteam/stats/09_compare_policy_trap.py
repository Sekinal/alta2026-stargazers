"""Attack 6a: `compare` defaults to policy fixed+ukg for both runs; a calibrated run (stack,
unweighted LR) is then judged at the wrong operating point and loses a comparison it wins."""

from _lib import *  # noqa: F403
from alta.common import PREDS
from alta.validation import ensemble, report

MEMBERS = ["emb_bge_large_lr", "emb_bge_base_lr", "emb_mpnet_lr"]

ensemble.stack(MEMBERS, "rt_stats_stack3")
for spec in ("fixed+ukg", "variety+ukg"):
    report.compare("rt_stats_stack3", "emb_ens3_lr", spec, B=1000)
for ext in ("parquet", "json"):
    (PREDS / f"rt_stats_stack3.{ext}").unlink(missing_ok=True)
