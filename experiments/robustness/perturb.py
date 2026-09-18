"""Shared perturbation-robustness suite: perturbed holdouts, a storage convention, and the report.

Question answered: how much does a model's score drop when the test text differs in form but not
in meaning (casing, punctuation, typos, dialect spelling/slang, google tokeniser artefacts,
whitespace, a missing segment prefix)? Labels stay the same, so any drop means brittleness.

Protocol (so every agent's numbers are comparable)
---------------------------------------------------
* Holdout = the pool rows of ONE project fold (default fold 0, 1,264 rows), seed 2026.
  The perturbation of a row depends only on (seed, kind, uid), not on the fold or the row order.
* MANDATORY: predict each perturbed CSV with the FOLD-k MODEL ONLY, i.e. the model trained on
  folds != k that produced your OOF for these rows (single-holdout neural runs: the model trained
  on the other folds). Fold-averaged predictions (predict_test.py style) are INVALID: 4 of the 5
  models trained on these very rows, so they are contaminated, not merely optimistic. The suite
  refuses them: save_rob_preds only accepts protocol="fold_model", `snapshot` requires the explicit
  --fold-model attestation, and the report marks anything else INVALID and leaves it out.
* Always also predict `clean` (the unperturbed holdout) with the same code path.
* Decision thresholds: segment+ukg fitted on the CLEAN holdout predictions and applied unchanged
  to every kind (a deployed model keeps its thresholds). The drop therefore isolates the text
  change. A second read-out fits thresholds on the run's OOF of the other folds (from preds/RUN)
  when available.

Kinds (KINDS below; all deterministic)
  clean                  identity (the reference; always predict it)
  lowercase              text.lower()
  strip_punct            drop all punctuation except in-word apostrophes; collapse spaces
  typo_3pct              ~3% of letters: swap with next letter / drop / double / keyboard neighbour;
                         never touches a "/s" marker
  au_to_uk, uk_to_au     AU <-> UK slang and currency (arvo<->afternoon, footpath<->pavement,
                         servo<->petrol station, $<->£, ...); the two varieties share spelling
  to_us, from_us         Commonwealth <-> US spelling (colour/color, -ise/-ize, centre/center, mum/mom)
  google_artefacts_on    split contractions and parentheses the way the google texts are
                         ("don't" -> "do n't", "can't" -> "ca n't", "(x" -> "( x"), on ALL rows
  google_artefacts_off   undo them ("do n't" -> "don't", "( x" -> "(x", "x ?" -> "x?")
  whitespace_noise       ~10% of spaces doubled, ~2% become newlines, leading/trailing spaces
  drop_prefix            text unchanged; FLAG for models that prepend a [variety][source] prefix:
                         predict this CSV with the prefix switched off in your own code

Usage
  uv run python experiments/robustness/perturb.py make [--fold 0] [--seed 2026] [--out DIR]
      -> DIR/<kind>.csv (load_test format: source, variety, text, empty labels) + DIR/manifest.json
         (default DIR = experiments/robustness/holdout_f<fold>_s<seed>)
  # predict every DIR/<kind>.csv with your fold-k model, then save, either in code:
  #   from experiments.robustness.perturb import save_rob_preds
  #   save_rob_preds(RUN, kind, frame_with_uid_p_sent_p_sarc, DIR, protocol="fold_model")
  # (uids may be the CSV's x00000.. uids from alta.common.load_test or the pool uids), or, if your
  # predict_test wrote the rows into preds/RUN.parquet (split == 'test'):
  # (only if those test rows came from the fold-k model, NOT from predict_test's 5-fold average):
  uv run python experiments/robustness/perturb.py snapshot --run RUN --kind KIND --fold-model [--dir DIR]
  uv run python experiments/robustness/perturb.py report --run RUN [--dir DIR]
      -> table + experiments/robustness/results/RUN__<holdout>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import PREDS, frame_fingerprint, load_pool, load_test  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 2026
FOLD = 0
KINDS = ("clean", "lowercase", "strip_punct", "typo_3pct", "au_to_uk", "uk_to_au", "to_us", "from_us",
         "google_artefacts_on", "google_artefacts_off", "whitespace_noise", "drop_prefix")

# ------------------------------------------------------------------ swap lists
# (AU form, UK form). One-way pairs are marked: only the AU -> UK direction is safe for them.
AU_UK = [("arvo", "afternoon"), ("footpath", "pavement"), ("servo", "petrol station"),
         ("bottle-o", "off-licence"), ("bottle shop", "off-licence"), ("brekkie", "breakfast"),
         ("sunnies", "sunglasses"), ("mozzie", "mosquito"), ("mozzies", "mosquitoes"), ("heaps", "loads"),
         ("doona", "duvet"), ("esky", "cool box"), ("thongs", "flip-flops"), ("bogan", "chav"),
         ("bogans", "chavs"), ("maccas", "mcdonald's"), ("dunny", "loo"), ("truck", "lorry"),
         ("trucks", "lorries"), ("bucks", "quid"),
         ("dollars", "pounds"), ("mate's rates", "mates' rates"), ("capsicum", "pepper"),
         ("rockmelon", "cantaloupe")]
AU_UK_ONEWAY = [("ute", "pickup"), ("rego", "registration"), ("g'day", "hello"), ("chook", "chicken"),
                ("barbie", "barbecue"), ("stubby", "bottle"), ("togs", "swimming costume"),
                ("fair dinkum", "genuinely"), ("bottlo", "off-licence"), ("avo", "avocado")]
# UK -> AU only where the UK word is unambiguous in the other direction
UK_AU_SKIP = {"afternoon", "breakfast", "loads", "pepper", "pounds"}

# (Commonwealth, US)
SPELL = [("colour", "color"), ("colours", "colors"), ("coloured", "colored"), ("favour", "favor"),
         ("favourite", "favorite"), ("favourites", "favorites"), ("flavour", "flavor"), ("flavours", "flavors"),
         ("behaviour", "behavior"), ("honour", "honor"), ("neighbour", "neighbor"), ("neighbours", "neighbors"),
         ("harbour", "harbor"), ("humour", "humor"), ("rumour", "rumor"), ("rumours", "rumors"),
         ("centre", "center"), ("centres", "centers"), ("theatre", "theater"), ("metre", "meter"),
         ("metres", "meters"), ("litre", "liter"), ("litres", "liters"), ("fibre", "fiber"),
         ("mum", "mom"), ("mums", "moms"), ("mummy", "mommy"), ("programme", "program"), ("tyre", "tire"),
         ("tyres", "tires"), ("travelled", "traveled"), ("travelling", "traveling"), ("cancelled", "canceled"),
         ("jewellery", "jewelry"), ("grey", "gray"), ("cheque", "check"), ("aluminium", "aluminum"),
         ("defence", "defense"), ("offence", "offense"), ("licence", "license"), ("catalogue", "catalog"),
         ("mould", "mold"), ("pyjamas", "pajamas"), ("maths", "math"), ("arse", "ass"), ("analyse", "analyze"),
         ("analysed", "analyzed"), ("paediatric", "pediatric"), ("anaemic", "anemic"), ("manoeuvre", "maneuver"),
         ("sceptical", "skeptical"), ("plough", "plow"), ("cosy", "cozy"), ("storey", "story")]
ISE_STEMS = ["real", "recogn", "organ", "apolog", "critic", "emphas", "priorit", "summar", "character",
             "special", "minim", "maxim", "final", "util", "custom", "author", "legal", "privat", "subsid",
             "sympath", "normal", "stabil", "capital", "mobil", "penal", "scrutin", "standard", "categor",
             "memor", "patron", "sanit", "jeopard", "fantas", "hospital", "idol", "civil", "global", "modern",
             "optim", "visual", "central", "harmon", "social", "vandal", "terror", "weapon", "monet", "polar"]
ISE_SUFFIXES = [("ise", "ize"), ("ised", "ized"), ("ises", "izes"), ("ising", "izing"), ("isation", "ization"),
                ("isations", "izations"), ("iser", "izer"), ("isers", "izers")]
for _s in ISE_STEMS:
    SPELL += [(_s + a, _s + b) for a, b in ISE_SUFFIXES]

KEYBOARD = {c: n for c, n in zip("qwertyuiopasdfghjklzxcvbnm",
                                 ["wa", "qes", "wrd", "etf", "ryg", "tuh", "yij", "uok", "ipl", "ol", "qsz", "awdxz",
                                  "sefxc", "drgcv", "fthvb", "gyjbn", "hukmn", "jilm", "kop", "asx", "zsdc", "xdfv",
                                  "cfgb", "vghn", "bhjm", "njk"])}


def _case_like(src: str, rep: str) -> str:
    if src.isupper() and len(src) > 1:
        return rep.upper()
    if src[:1].isupper():
        return rep[:1].upper() + rep[1:]
    return rep


def _swap(text: str, pairs: list[tuple[str, str]]) -> str:
    """Whole-word / whole-phrase replacement, longest first, case pattern kept."""
    table = {a.lower(): b for a, b in pairs if a.lower() != b.lower()}
    if not table:
        return text
    alt = "|".join(re.escape(k) for k in sorted(table, key=len, reverse=True))
    return re.sub(rf"(?<![\w'-])(?:{alt})(?![\w'-])", lambda m: _case_like(m.group(0), table[m.group(0).lower()]),
                  text, flags=re.I)


def _typos(text: str, rng: np.random.Generator, rate: float = 0.03) -> str:
    protected = np.zeros(len(text), bool)
    for m in re.finditer(r"(?:^|\s)/s\b", text):
        protected[m.start():m.end()] = True
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isalpha() and not protected[i] and rng.random() < rate:
            op = int(rng.integers(4))
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if op == 0 and nxt.isalpha() and not protected[i + 1]:
                out += [nxt, ch]
                i += 2
                continue
            if op == 1:
                i += 1
                continue
            if op == 2:
                out += [ch, ch]
                i += 1
                continue
            nb = KEYBOARD.get(ch.lower())
            if nb:
                r = nb[int(rng.integers(len(nb)))]
                out.append(r.upper() if ch.isupper() else r)
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _artefacts_on(text: str) -> str:
    t = re.sub(r"\b([Cc])an't\b", r"\1a n't", text)
    t = re.sub(r"\b([Ww])on't\b", r"\1o n't", t)
    t = re.sub(r"(\w)n't\b", r"\1 n't", t)
    return re.sub(r"\((?=\w)", "( ", t)


def _artefacts_off(text: str) -> str:
    t = re.sub(r"(\w) (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1\2", text, flags=re.I)
    t = re.sub(r"([(\[]) ", r"\1", t)
    t = re.sub(r" ([)\]])", r"\1", t)
    return re.sub(r"(\w) ([?!.,;:])(?=\s|$)", r"\1\2", t)


def _whitespace(text: str, rng: np.random.Generator) -> str:
    parts = text.split(" ")
    out = [parts[0]]
    for p in parts[1:]:
        r = rng.random()
        out.append(("\n" if r < 0.02 else "  " if r < 0.12 else " ") + p)
    lead = " " * int(rng.integers(0, 3))
    trail = " " * int(rng.integers(0, 3))
    return lead + "".join(out) + trail


def perturb_one(text: str, kind: str, rng: np.random.Generator) -> str:
    if kind in ("clean", "drop_prefix"):
        return text
    if kind == "lowercase":
        return text.lower()
    if kind == "strip_punct":
        t = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", " ", text)
        t = re.sub(r"[^\w\s']", " ", t)
        return re.sub(r"[ \t]+", " ", t).strip()
    if kind == "typo_3pct":
        return _typos(text, rng)
    if kind == "au_to_uk":
        return _swap(text.replace("$", "£"), AU_UK + AU_UK_ONEWAY)
    if kind == "uk_to_au":
        return _swap(text.replace("£", "$"), [(b, a) for a, b in AU_UK if b not in UK_AU_SKIP])
    if kind == "to_us":
        return _swap(text, SPELL)
    if kind == "from_us":
        return _swap(text, [(b, a) for a, b in SPELL if b not in ("check", "story", "tire", "tires", "mold",
                                                                    "program", "meter", "meters")])
    if kind == "google_artefacts_on":
        return _artefacts_on(text)
    if kind == "google_artefacts_off":
        return _artefacts_off(text)
    if kind == "whitespace_noise":
        return _whitespace(text, rng)
    raise ValueError(f"unknown kind {kind!r}; kinds: {KINDS}")


def perturb(texts: list[str], kind: str, seed: int = SEED, keys: list[str] | None = None) -> list[str]:
    """Perturb texts; row i uses an RNG seeded by (seed, kind, keys[i]) (keys default: the index).

    Pass uids as keys so a row's perturbation does not depend on which rows are perturbed with it.
    """
    keys = [str(i) for i in range(len(texts))] if keys is None else keys
    out = []
    for t, k in zip(texts, keys):
        rng = np.random.default_rng([seed, zlib.crc32(f"{kind}|{k}".encode())])
        out.append(perturb_one(t, kind, rng))
    return out


# ------------------------------------------------------------------ holdouts
def holdout_dir(fold: int = FOLD, seed: int = SEED) -> Path:
    return HERE / f"holdout_f{fold}_s{seed}"


def make_perturbed_holdout(fold: int, kind: str, seed: int = SEED) -> pl.DataFrame:
    """Holdout rows of `fold` (pool uids, pool order) with perturbed text; labels dropped."""
    pool = load_pool()
    h = pool.filter(pl.col("fold") == fold)
    t = perturb(h["text"].to_list(), kind, seed, h["uid"].to_list())
    return h.select("uid", pl.col("source").cast(pl.String), pl.col("variety").cast(pl.String)).with_columns(
        pl.Series("text", t))


def make(fold: int, seed: int, out: Path, kinds=KINDS) -> None:
    out.mkdir(parents=True, exist_ok=True)
    man = {"fold": fold, "seed": seed, "kinds": {}, "protocol": __doc__.split("Kinds")[0].strip()}
    clean = make_perturbed_holdout(fold, "clean", seed)
    for kind in kinds:
        h = make_perturbed_holdout(fold, kind, seed)
        path = out / f"{kind}.csv"
        h.select("source", "variety", "text", pl.lit(None, pl.Int64).alias("sentiment"),
                 pl.lit(None, pl.Int64).alias("sarcasm")).write_csv(path)
        test = load_test(path)  # round trip through the official reader
        if test.height != h.height or test["text"].to_list() != h["text"].to_list():
            raise RuntimeError(f"{kind}: CSV round trip changed the text")
        changed = float(np.mean([a != b for a, b in zip(h["text"].to_list(), clean["text"].to_list())]))
        man["kinds"][kind] = {"csv": path.name, "sha": frame_fingerprint(test), "n": h.height,
                              "share_changed": round(changed, 4)}
        print(f"{kind:22s} {h.height} rows, {changed:6.1%} changed -> {path}")
    man["uids"] = clean["uid"].to_list()  # CSV row i (test uid x{i:05d}) = pool uid uids[i]
    (out / "manifest.json").write_text(json.dumps(man, indent=1))


def _manifest(d: Path) -> dict:
    p = d / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `perturb.py make` first")
    return json.loads(p.read_text())


# ------------------------------------------------------------------ prediction storage
def rob_path(run: str, kind: str, d: Path) -> Path:
    return d / "preds" / run / f"{kind}.parquet"


def save_rob_preds(run: str, kind: str, preds: pl.DataFrame, d: Path | str | None = None,
                   protocol: str = "fold_model", meta: dict | None = None) -> Path:
    """preds: uid (test x-uids of the kind's CSV OR pool uids), p_sent, p_sarc."""
    d = Path(d) if d else holdout_dir()
    man = _manifest(d)
    if protocol != "fold_model":
        raise ValueError("only protocol='fold_model' is valid: holdout rows must be scored by the model that "
                         "held them out (fold-averaged predictions are contaminated by training on these rows)")
    uids = man["uids"]
    xmap = {f"x{i:05d}": u for i, u in enumerate(uids)}
    df = preds.select("uid", pl.col("p_sent").cast(pl.Float64), pl.col("p_sarc").cast(pl.Float64))
    if df["uid"].str.starts_with("x").all():
        df = df.with_columns(pl.col("uid").replace_strict(xmap))
    if sorted(df["uid"].to_list()) != sorted(uids):
        raise ValueError(f"{run}/{kind}: uids do not match the holdout ({df.height} vs {len(uids)})")
    for c in ("p_sent", "p_sarc"):
        x = df[c].to_numpy()
        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError(f"{run}/{kind}.{c}: probabilities must be finite and in [0, 1]")
    path = rob_path(run, kind, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    path.with_suffix(".json").write_text(json.dumps({"run": run, "kind": kind, "protocol": protocol,
                                                     "fold": man["fold"], "seed": man["seed"],
                                                     "sha": man["kinds"][kind]["sha"], **(meta or {})}, indent=1))
    return path


def snapshot(run: str, kind: str, d: Path, fold_model: bool) -> Path:
    """Copy preds/RUN.parquet test rows (must have been predicted on DIR/<kind>.csv by the fold-k model)."""
    if not fold_model:
        raise SystemExit("refused: pass --fold-model to attest that these test rows come from the fold-k model "
                         "only. predict_test.py-style 5-fold averages are INVALID for robustness.")
    meta = json.loads((PREDS / f"{run}.json").read_text())
    if "average of the 5 fold models" in str(meta.get("test_protocol", "")):
        raise SystemExit(f"refused: {run}'s test rows are a 5-fold average (test_protocol in its meta): INVALID here")
    man = _manifest(d)
    if meta.get("test_sha") != man["kinds"][kind]["sha"]:
        raise ValueError(f"{run}: its test rows were made for sha {meta.get('test_sha')}, "
                         f"not {kind}.csv ({man['kinds'][kind]['sha']}); run its predict_test on {d / (kind + '.csv')}")
    p = pl.read_parquet(PREDS / f"{run}.parquet").filter(pl.col("split") == "test")
    return save_rob_preds(run, kind, p, d, "fold_model", {"source": "snapshot of preds test rows (attested fold-k)"})


# ------------------------------------------------------------------ report
def _eval_data(pool_h: pl.DataFrame, p: pl.DataFrame):
    from alta.validation.data import _frame_to_eval
    df = pool_h.join(p, on="uid", how="left", maintain_order="left")
    if df["p_sent"].null_count():
        raise ValueError("predictions do not cover the holdout")
    return _frame_to_eval(df, labelled=True, meta={})


def robustness_report(run_base: str, runs: dict[str, str] | None = None, d: Path | str | None = None,
                      spec: str = "segment+ukg", write: bool = True) -> list[dict]:
    """run_base: run name whose robustness preds live in DIR/preds/<run>/; runs: optional
    {kind: other run name} to take a kind's predictions from a different stored run."""
    from sklearn.metrics import roc_auc_score
    from alta.validation import policies
    from alta.validation.metrics import components
    d = Path(d) if d else holdout_dir()
    man = _manifest(d)
    pool = load_pool()
    hold = pool.filter(pl.col("uid").is_in(man["uids"]))
    src = {k: run_base for k in KINDS if rob_path(run_base, k, d).exists()}
    src.update(runs or {})
    if "clean" not in src:
        raise FileNotFoundError(f"no clean predictions for {run_base} in {d / 'preds'}: predict clean.csv first")
    protos = {k: json.loads(rob_path(r, k, d).with_suffix(".json").read_text()).get("protocol") for k, r in src.items()}
    invalid = sorted(k for k, p in protos.items() if p != "fold_model")
    if invalid:
        print(f"INVALID (not fold-k-model predictions, excluded): {invalid}")
    if "clean" in invalid:
        raise ValueError("clean predictions are INVALID (not from the fold-k model): nothing can be reported")
    src = {k: r for k, r in src.items() if k not in invalid}
    ev = {k: _eval_data(hold, pl.read_parquet(rob_path(r, k, d))) for k, r in src.items()}
    thr_clean = policies.fit(spec, ev["clean"])
    thr_oof = None
    if (PREDS / f"{run_base}.parquet").exists():  # deployment read-out: thresholds from the other folds' OOF
        from alta.validation.data import load_oof
        o = load_oof(run_base, pool)
        thr_oof = policies.fit(spec, o.subset(o.fold != man["fold"]))
    rows = []
    base = {}
    for k in [k for k in KINDS if k in ev]:
        e = ev[k]
        c = components(e, policies.apply(thr_clean, e))
        uk = (e.variety == 1) & (e.source == 1)
        au = (e.variety == 0) & (e.source == 1)
        r = {"kind": k, "protocol": protos[k], "changed": man["kinds"][k]["share_changed"], **c,
             "auc_UKr_sarc": float(roc_auc_score(e.y["sarc"][uk], e.p["sarc"][uk])),
             "auc_AUr_sent": float(roc_auc_score(e.y["sent"][au], e.p["sent"][au])),
             "mean_abs_dp": float(np.mean([np.abs(e.p[t] - ev["clean"].p[t]).mean() for t in ("sent", "sarc")]))}
        if thr_oof is not None:
            r["score_oofthr"] = components(e, policies.apply(thr_oof, e))["score"]
        if k == "clean":
            base = r
        for m in ("score", "sent-AU", "sarc-UK", "auc_UKr_sarc", "auc_AUr_sent") + (("score_oofthr",) if thr_oof else ()):
            r[f"d_{m}"] = r[m] - base[m]
        rows.append(r)
    _print(run_base, man, rows)
    if write:
        out = HERE / "results" / f"{run_base}__{d.name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"run": run_base, "holdout": d.name, "fold": man["fold"], "seed": man["seed"],
                                   "policy": spec, "thresholds": "fitted on clean holdout predictions",
                                   "invalid_excluded": invalid,
                                   "rows": rows}, indent=1))
    return rows


def _print(run: str, man: dict, rows: list[dict]) -> None:
    print(f"# robustness: {run}, holdout fold {man['fold']} seed {man['seed']}, thresholds fitted on clean")
    cols = ["kind", "protocol", "changed", "score", "d_score", "sent-AU", "d_sent-AU", "sarc-UK", "d_sarc-UK",
            "d_auc_UKr_sarc", "d_auc_AUr_sent", "mean_abs_dp"] + (["d_score_oofthr"] if "d_score_oofthr" in rows[0] else [])
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:+.4f}" if c.startswith("d_") else f"{r[c]:.4f}" if isinstance(r[c], float)
                         else str(r[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("make")
    s.add_argument("--fold", type=int, default=FOLD)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("snapshot")
    s.add_argument("--run", required=True)
    s.add_argument("--kind", required=True, choices=KINDS)
    s.add_argument("--dir", type=Path, default=None)
    s.add_argument("--fold-model", action="store_true", help="attest: rows come from the fold-k model only")
    s = sub.add_parser("report")
    s.add_argument("--run", required=True)
    s.add_argument("--dir", type=Path, default=None)
    s.add_argument("--policy", default="segment+ukg")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.fold, a.seed, a.out or holdout_dir(a.fold, a.seed))
    elif a.cmd == "snapshot":
        print(snapshot(a.run, a.kind, a.dir or holdout_dir(), a.fold_model))
    else:
        robustness_report(a.run, d=a.dir, spec=a.policy)


if __name__ == "__main__":
    main()
