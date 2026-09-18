"""Re-run the whole lexical EDA.  From the project root:  uv run python eda/lexical/eda_lexical.py

Steps (each is also runnable on its own from eda/lexical/):
  01_logodds.py          Monroe log-odds per label/cell + AU<->UK cue transfer        (~2 min)
  02_style.py            stylistic markers, lifts, VADER surface polarity  [needs --with vaderSentiment]
  03_dialect_topics.py   dialect markers, AU/UK separability, reddit NMF topics
  04_tfidf_baseline.py   TF-IDF + LR OOF baseline, writes preds/tfidf_lr.parquet     (~10 min)
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ["uv", "run", "python", "01_logodds.py"],
    ["uv", "run", "--with", "vaderSentiment", "python", "02_style.py"],
    ["uv", "run", "python", "03_dialect_topics.py"],
    ["uv", "run", "python", "04_tfidf_baseline.py"],
]
only = set(sys.argv[1:])  # optional: step prefixes, e.g. "01 04"
for cmd in STEPS:
    if only and cmd[-1][:2] not in only:
        continue
    print(f"\n######## {cmd[-1]}", flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)
