"""Print a compact summary of feature-set sidecars: uv run python experiments/features/summ.py SET [SET ...]"""
import json
import sys
from pathlib import Path

for s in sys.argv[1:]:
    e = json.loads((Path(__file__).parent / "store" / f"{s}.json").read_text())
    v = e["evaluation"]
    a = v["alone"]
    print(f"== {s} ({e['n_features']} feats)  alone score {a['score']} sarc-UK {a['sarc-UK']} sent-AU {a['sent-AU']} "
          f"| UKr-sarc AUC {a['slices']['UKreddit_sarc']['auc']} AUr-sent AUC {a['slices']['AUreddit_sent']['auc']} "
          f"| canary {a['canary']['passed']} ({a['canary']['mean']} vs {a['canary']['floor']})")
    for c, inc in v["increment"].items():
        sd = inc["slice_delta"]
        print(f"  {c}: dScore {inc['delta']['score']:+.4f} p={inc['p_set_better']['score']} folds {inc['per_fold_delta']} "
              f"dSarcUK {inc['delta']['sarc-UK']:+.4f} dSentAU {inc['delta']['sent-AU']:+.4f} dLL {inc['mean_component_logloss_delta']:+.5f}")
        print(f"      UKr-sarc dAUC {sd['UKreddit_sarc']['auc']:+.4f} dLL {sd['UKreddit_sarc']['logloss']:+.4f} folds {inc['per_fold_auc_delta']['UKreddit_sarc']}"
              f" | AUr-sent dAUC {sd['AUreddit_sent']['auc']:+.4f} dLL {sd['AUreddit_sent']['logloss']:+.4f} folds {inc['per_fold_auc_delta']['AUreddit_sent']}")
