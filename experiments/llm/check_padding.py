"""Debug: batched (left-padded) vs one-at-a-time scores must agree (linear-attention layers)."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "experiments/llm")
sys.path.insert(0, ".")
from llm_common import Scorer, build_prompts, load_model
from alta.common import load_pool

model_name = sys.argv[1]
tok, model = load_model(model_name)
sc = Scorer(tok, model, (len(model.model.layers) - 1,))
df = load_pool().filter(pl.col("source") == "reddit").sample(12, seed=3)
prompts = build_prompts(tok, df, "sarc_v1")
pb, mb, _ = sc(prompts, "sarc")
m1 = np.array([sc([p], "sarc")[1][0] for p in prompts])
print("batched ", np.round(mb, 3))
print("single  ", np.round(m1, 3))
print("max abs diff", np.abs(mb - m1).max())
