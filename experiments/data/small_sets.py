"""Download and normalise the small, public sarcasm/irony corpora into data/external/sarcasm/.

  uv run --with pyahocorasick python experiments/data/small_sets.py [name ...]

Every builder downloads to data/external/raw/<name>/, maps labels to sarcasm 0/1 and writes
<name>.parquet via ext_common.write (pool dedupe + MANIFEST entry). None of these is BESSTIE or
derived from it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import RAW, fetch, finalize, write  # noqa: E402

GH = "https://raw.githubusercontent.com"
HF = "https://huggingface.co/datasets"


def isarcasmeval() -> None:
    """SemEval-2022 Task 6 (English): tweets labelled sarcastic by their own authors."""
    base = f"{GH}/iabufarha/iSarcasmEval/main"
    d = RAW / "isarcasmeval"
    tr = pl.read_csv(fetch(f"{base}/train/train.En.csv", d / "train.En.csv"), infer_schema_length=0)
    te = pl.read_csv(fetch(f"{base}/test/task_A_En_test.csv", d / "task_A_En_test.csv"), infer_schema_length=0)
    tr = tr.rename({c: c.strip() for c in tr.columns})
    te = te.rename({c: c.strip() for c in te.columns})
    main = pl.concat([
        tr.select(pl.col("tweet").alias("text"), pl.col("sarcastic").cast(pl.Int8).alias("sarcasm")),
        te.select(pl.col("text"), pl.col("sarcastic").cast(pl.Int8).alias("sarcasm")),
    ])
    common = dict(source="twitter", variety_guess="unknown", license="MIT (repo)", label_quality="gold",
                  year=2021)
    write(finalize(main, dataset="isarcasmeval_en", **common), "isarcasmeval_en",
          url="https://github.com/iabufarha/iSarcasmEval", license="MIT",
          notes="Author-labelled intended sarcasm (Abu Farha et al. 2022), train + task A test; ~25% positive.")
    # the authors' own non-sarcastic rephrasings of their sarcastic tweets: minimal-pair negatives
    reph = tr.filter(pl.col("sarcastic") == "1", pl.col("rephrase").is_not_null()).select(
        pl.col("rephrase").alias("text"), pl.lit(0).alias("sarcasm"))
    write(finalize(reph, dataset="isarcasmeval_en_rephrase", **common), "isarcasmeval_en_rephrase",
          url="https://github.com/iabufarha/iSarcasmEval", license="MIT",
          notes="Author-written non-sarcastic rephrasings of the sarcastic iSarcasmEval tweets (all label 0). "
                "Pair with isarcasmeval_en positives for contrastive/minimal-pair training.")


def semeval2018() -> None:
    """SemEval-2018 Task 3 subtask A: ironic vs not (tweets; #irony/#sarcasm/#not hashtags removed)."""
    base = f"{GH}/Cyvhee/SemEval2018-Task3/master/datasets"
    d = RAW / "semeval2018"
    parts = [fetch(f"{base}/train/SemEval2018-T3-train-taskA_emoji.txt", d / "train.txt"),
             fetch(f"{base}/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt", d / "gold_test.txt")]
    rows = []
    for p in parts:
        for line in p.read_text(encoding="utf-8").splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 3 and f[1] in ("0", "1"):
                rows.append({"text": "\t".join(f[2:]), "sarcasm": int(f[1])})
    df = finalize(pl.DataFrame(rows), dataset="semeval2018_irony", source="twitter", variety_guess="en-US",
                  license="CC BY-NC-SA 4.0", label_quality="gold", year=2014)
    write(df, "semeval2018_irony", url="https://github.com/Cyvhee/SemEval2018-Task3", license="CC BY-NC-SA 4.0",
          notes="Van Hee et al. 2018; hashtag-collected then manually annotated (irony incl. sarcasm). "
                "Organisers ask for academic use only.")


def news_headlines() -> None:
    """Misra & Arora: TheOnion (sarcastic) vs HuffPost headlines. Label = outlet."""
    d = RAW / "news_headlines"
    p = fetch(f"{GH}/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json",
              d / "Sarcasm_Headlines_Dataset.json")
    txt = p.read_text().strip()
    recs = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    df = pl.DataFrame({"text": [r["headline"] for r in recs], "sarcasm": [int(r["is_sarcastic"]) for r in recs]})
    df = finalize(df, dataset="news_headlines", source="news", variety_guess="en-US", license="CC BY 4.0",
                  label_quality="weak", year=2019)
    write(df, "news_headlines", url="https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection",
          license="CC BY 4.0 (per author's Kaggle release)",
          notes="Label = outlet (The Onion satire vs HuffPost); headline register, far from reddit. Low priority.")


def figlang2020() -> None:
    """FigLang 2020 shared task: reddit (from SARC) and twitter responses with conversation context."""
    base = f"{GH}/EducationalTestingService/sarcasm/master"
    d = RAW / "figlang2020"
    for plat in ("reddit", "twitter"):
        frames = []
        for split in ("training", "testing"):
            p = fetch(f"{base}/{plat}/sarcasm_detection_shared_task_{plat}_{split}.jsonl", d / f"{plat}_{split}.jsonl")
            recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            recs = [r for r in recs if r.get("label") in ("SARCASM", "NOT_SARCASM")]
            if not recs:
                print(f"figlang {plat} {split}: no labels, skipped", flush=True)
                continue
            frames.append(pl.DataFrame({
                "text": [r["response"] for r in recs],
                "sarcasm": [int(r["label"] == "SARCASM") for r in recs],
                "parent_text": [(r.get("context") or [None])[-1] for r in recs],
            }))
        df = pl.concat(frames)
        if plat == "twitter":  # anonymised handles
            df = df.with_columns(pl.col("text", "parent_text").str.replace_all(r"@USER\s*", "").str.strip_chars())
        name = f"figlang2020_{plat}"
        df = finalize(df, dataset=name, source=plat, variety_guess="unknown", license="research (no licence file)",
                      label_quality="self-labelled")
        write(df, name, url="https://github.com/EducationalTestingService/sarcasm",
              license="no explicit licence; released for research (Ghosh et al. 2020)",
              notes=("Reddit part is re-sampled from SARC (/s, balanced) -> overlaps sarc_*; "
                     if plat == "reddit" else "Twitter part labelled by #sarcasm/#sarcastic hashtags (removed); ")
              + "parent_text = last context turn.")


def mustard() -> None:
    d = RAW / "mustard"
    p = fetch(f"{GH}/soujanyaporia/MUStARD/master/data/sarcasm_data.json", d / "sarcasm_data.json")
    recs = json.loads(p.read_text())
    df = pl.DataFrame({
        "text": [r["utterance"] for r in recs.values()],
        "sarcasm": [int(bool(r["sarcasm"])) for r in recs.values()],
        "parent_text": [(r.get("context") or [None])[-1] for r in recs.values()],
    })
    df = finalize(df, dataset="mustard", source="other", variety_guess="en-US", license="MIT",
                  label_quality="gold")
    write(df, "mustard", url="https://github.com/soujanyaporia/MUStARD", license="MIT",
          notes="TV-show utterances (Friends, Big Bang Theory, Golden Girls, Sarcasmaholics); text-only use.")


def reddit_irony() -> None:
    """Wallace et al. 2014 (ACL): reddit comments annotated for irony by 3 annotators."""
    d = RAW / "reddit_irony"
    p = fetch(f"{GH}/bwallace/ACL-2014-irony/master/irony-labeled.csv", d / "irony-labeled.csv")
    df = pl.read_csv(p, infer_schema_length=0)
    df = df.select(pl.col("comment_text").alias("text"),
                   (pl.col("label").cast(pl.Int32) == 1).cast(pl.Int8).alias("sarcasm"))
    df = finalize(df, dataset="reddit_irony_wallace2014", source="reddit", variety_guess="en-US",
                  license="no explicit licence (research release)", label_quality="gold", year=2013)
    write(df, "reddit_irony_wallace2014", url="https://github.com/bwallace/ACL-2014-irony",
          license="no explicit licence (public research release)",
          notes="r/politics, r/conservative, r/progressive, r/atheism, r/christianity, r/news, r/technology.")


EPIC_VARIETY = {"gb": "en-UK", "au": "en-AU", "us": "en-US", "ie": "other", "in": "other"}


def epic() -> None:
    """EPIC (Frenda et al. 2023): perspectivist irony annotations of reddit/twitter posts from 5 English varieties."""
    d = RAW / "epic"
    url = "https://huggingface.co/api/datasets/Multilingual-Perspectivist-NLU/EPIC/parquet/default/train/0.parquet"
    try:
        df = pl.read_parquet(fetch(url, d / "epic.parquet"))
        src_url = "https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/EPIC"
    except Exception as e:  # fall back to the CreativeLang mirror
        print("EPIC official failed:", e, flush=True)
        df = pl.read_parquet(fetch(f"{HF}/CreativeLang/EPIC_Irony/resolve/main/data/"
                                   "train-00000-of-00001-364b3dda3ec09218.parquet", d / "epic_cl.parquet"))
        src_url = "https://huggingface.co/datasets/CreativeLang/EPIC_Irony"
    df = df.with_columns((pl.col("label").cast(pl.String).str.to_lowercase().is_in(["iro", "1", "ironic", "irony"]))
                         .cast(pl.Float64).alias("y"))
    g = df.group_by("id_original").agg(
        pl.col("text").first(), pl.col("parent_text").first(), pl.col("source").first(),
        pl.col("subreddit").first(), pl.col("Language_variety").first(), pl.col("y").mean().alias("p"),
        pl.len().alias("n"))
    g = g.filter(pl.col("p") != 0.5).with_columns((pl.col("p") > 0.5).cast(pl.Int8).alias("sarcasm"))
    g = g.with_columns(
        pl.col("Language_variety").str.to_lowercase().replace_strict(EPIC_VARIETY, default="unknown")
        .alias("variety_guess"),
        pl.col("source").str.to_lowercase(),
    )
    out = finalize(g, dataset="epic_irony", license="CC BY-NC-SA 4.0", label_quality="gold")
    write(out, "epic_irony", url=src_url, license="CC BY-NC-SA 4.0",
          notes="Majority vote over ~5 annotators per text (ties dropped); variety_guess from the corpus' "
                "Language_variety (text origin: AU/UK/US/IE/IN reddit+twitter). Irony, not strictly sarcasm.")


BUILDERS = {f.__name__: f for f in (isarcasmeval, semeval2018, news_headlines, figlang2020, mustard,
                                    reddit_irony, epic)}

if __name__ == "__main__":
    names = sys.argv[1:] or list(BUILDERS)
    for n in names:
        try:
            BUILDERS[n]()
        except Exception as e:
            print(f"FAILED {n}: {e!r}", flush=True)
