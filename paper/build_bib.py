"""Build paper/references.bib from authoritative sources (no hand-typed metadata for papers).

For each entry: search Crossref by title, require a near-exact title match, then take the official BibTeX
(ACL Anthology for 10.18653 DOIs, DOI content negotiation otherwise). Entries without a DOI are looked up in the arXiv
API (title match required) and emitted as arXiv preprints, with the venue added when known. Non-paper resources
(model cards, software, data portals) are listed in MANUAL with their URLs. Unmatched entries are reported, never guessed.

  uv run python paper/build_bib.py            -> paper/references.bib, paper/bib_report.txt
"""

from __future__ import annotations

import difflib
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "alta2026-bib/0.1 (mailto:research@irvingernesto.com)"}

# key, exact title, first-author family name, year (for checking), source preference ("doi" or "arxiv"), venue note
REFS = [
    # task, data, core models
    ("srirag-etal-2025-besstie", "BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English", "Srirag", 2025, "doi", ""),
    ("he2023debertav", "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing", "He", 2021, "arxiv", "ICLR 2023"),
    ("wang2022e5", "Text Embeddings by Weakly-Supervised Contrastive Pre-training", "Wang", 2022, "arxiv", ""),
    ("wang2024me5", "Multilingual E5 Text Embeddings: A Technical Report", "Wang", 2024, "arxiv", ""),
    ("dettmers2023qlora", "QLoRA: Efficient Finetuning of Quantized LLMs", "Dettmers", 2023, "arxiv", "NeurIPS 2023"),
    ("alain2017probes", "Understanding intermediate layers using linear classifier probes", "Alain", 2016, "arxiv", "ICLR 2017 workshop"),
    ("laurer2023zeroshot", "Building Efficient Universal Classifiers with Natural Language Inference", "Laurer", 2023, "arxiv", ""),
    ("cohen1960kappa", "A Coefficient of Agreement for Nominal Scales", "Cohen", 1960, "doi", ""),
    ("faisal-etal-2024-dialectbench", "DIALECTBENCH: An NLP Benchmark for Dialects, Varieties, and Closely-Related Languages", "Faisal", 2024, "doi", ""),
    # sarcasm background
    ("riloff-etal-2013-sarcasm", "Sarcasm as Contrast between a Positive Sentiment and Negative Situation", "Riloff", 2013, "doi", ""),
    ("joshi2017survey", "Automatic Sarcasm Detection: A Survey", "Joshi", 2017, "doi", ""),
    ("khodak-etal-2018-large", "A Large Self-Annotated Corpus for Sarcasm", "Khodak", 2018, "arxiv", "LREC 2018"),
    ("van-hee-etal-2018-semeval", "SemEval-2018 Task 3: Irony Detection in English Tweets", "Van Hee", 2018, "doi", ""),
    ("abu-farha-etal-2022-semeval", "SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic", "Abu Farha", 2022, "doi", ""),
    ("frenda-etal-2023-epic", "EPIC: Multi-Perspective Annotation of a Corpus of Irony", "Frenda", 2023, "doi", ""),
    ("ghosh-etal-2020-report", "A Report on the 2020 Sarcasm Detection Shared Task", "Ghosh", 2020, "doi", ""),
    ("wallace-etal-2014-humans", "Humans Require Context to Infer Ironic Intent (so Computers Probably do, too)", "Wallace", 2014, "doi", ""),
    ("castro-etal-2019-towards", "Towards Multimodal Sarcasm Detection (An _Obviously_ Perfect Paper)", "Castro", 2019, "doi", ""),
    # fine-tuning methods tried
    ("hu2022lora", "LoRA: Low-Rank Adaptation of Large Language Models", "Hu", 2021, "arxiv", "ICLR 2022"),
    ("kalajdzievski2023rslora", "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA", "Kalajdzievski", 2023, "arxiv", ""),
    ("liu2024dora", "DoRA: Weight-Decomposed Low-Rank Adaptation", "Liu", 2024, "arxiv", "ICML 2024"),
    ("paischer2024eva", "Parameter Efficient Fine-tuning via Explained Variance Adaptation", "Paischer", 2025, "arxiv", ""),
    ("zhang2024riemannian", "Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models", "Zhang", 2024, "arxiv", "ICML 2024"),
    ("gururangan-etal-2020-dont", "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks", "Gururangan", 2020, "doi", ""),
    ("howard-ruder-2018-universal", "Universal Language Model Fine-tuning for Text Classification", "Howard", 2018, "doi", ""),
    ("clark2020electra", "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators", "Clark", 2020, "arxiv", "ICLR 2020"),
    ("warner2024modernbert", "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference", "Warner", 2024, "arxiv", ""),
    ("liu2019roberta", "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "Liu", 2019, "arxiv", ""),
    ("barbieri-etal-2020-tweeteval", "TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification", "Barbieri", 2020, "doi", ""),
    ("lee2013pseudo", "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks", "Lee", 2013, "manual", ""),
    ("beaglehole2025xrfm", "xRFM: Accurate, scalable, and interpretable feature learning models for tabular data", "Beaglehole", 2025, "arxiv", ""),
    ("ke2017lightgbm", "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", "Ke", 2017, "manual", ""),
    ("chen2016xgboost", "XGBoost: A Scalable Tree Boosting System", "Chen", 2016, "doi", ""),
    ("prokhorenkova2018catboost", "CatBoost: unbiased boosting with categorical features", "Prokhorenkova", 2017, "arxiv", "NeurIPS 2018"),
    ("wang-manning-2012-baselines", "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification", "Wang", 2012, "manual", ""),
    ("xiao2023cpack", "C-Pack: Packed Resources For General Chinese Embeddings", "Xiao", 2023, "arxiv", ""),
    # analysis methods
    ("monroe2008fightin", "Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict", "Monroe", 2008, "doi", ""),
    ("northcutt2021confident", "Confident Learning: Estimating Uncertainty in Dataset Labels", "Northcutt", 2021, "doi", ""),
    ("hutto2014vader", "VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text", "Hutto", 2014, "doi", ""),
    ("mohammad2013emolex", "Crowdsourcing a Word-Emotion Association Lexicon", "Mohammad", 2013, "doi", ""),
    ("mcinnes2018umap", "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction", "McInnes", 2018, "arxiv", ""),
    ("daume-iii-2007-frustratingly", "Frustratingly Easy Domain Adaptation", "Daum", 2007, "manual", ""),
    ("dodge2020finetuning", "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping", "Dodge", 2020, "arxiv", ""),
    ("mosbach2021stability", "On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines", "Mosbach", 2020, "arxiv", "ICLR 2021"),
    ("efron1979bootstrap", "Bootstrap Methods: Another Look at the Jackknife", "Efron", 1979, "doi", ""),
    ("nosek2018preregistration", "The preregistration revolution", "Nosek", 2018, "doi", ""),
    ("hartmann2023siebert", "More than a Feeling: Accuracy and Application of Sentiment Analysis", "Hartmann", 2023, "doi", ""),
    # external data used in failed screens
    ("demszky-etal-2020-goemotions", "GoEmotions: A Dataset of Fine-Grained Emotions", "Demszky", 2020, "doi", ""),
    ("potts-etal-2021-dynasent", "DynaSent: A Dynamic Benchmark for Sentiment Analysis", "Potts", 2021, "doi", ""),
    ("socher-etal-2013-recursive", "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank", "Socher", 2013, "manual", ""),
    ("pang-lee-2005-seeing", "Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales", "Pang", 2005, "manual", ""),
    ("maas-etal-2011-learning", "Learning Word Vectors for Sentiment Analysis", "Maas", 2011, "manual", ""),
    ("zhang2015character", "Character-level Convolutional Networks for Text Classification", "Zhang", 2015, "arxiv", "NeurIPS 2015"),
    ("chang-etal-2020-convokit", "ConvoKit: A Toolkit for the Analysis of Conversations", "Chang", 2020, "doi", ""),
    ("he2017translation", "Translation-based Recommendation", "He", 2017, "doi", ""),
]

# resources without a paper (URLs verified to exist when this file was written); plus classic papers without DOIs whose
# ACL Anthology / proceedings pages are fetched below
MANUAL_BIB = {
    "qwen35": """@misc{qwen35,
  title        = {{Qwen3.5-9B}},
  author       = {{Qwen Team}},
  year         = {2026},
  howpublished = {\\url{https://huggingface.co/Qwen/Qwen3.5-9B}},
  note         = {Model card}
}""",
    "unsloth": """@misc{unsloth,
  title        = {{Unsloth}},
  author       = {Han, Daniel and Han, Michael and {Unsloth team}},
  year         = {2023},
  howpublished = {\\url{https://github.com/unslothai/unsloth}}
}""",
    "arcticshift": """@misc{arcticshift,
  title        = {{Arctic Shift}: Reddit data archive and API},
  author       = {{Arctic Shift contributors}},
  year         = {2024},
  howpublished = {\\url{https://github.com/ArthurHeitmann/arctic_shift}}
}""",
    "claudecode": """@misc{claudecode,
  title        = {{Claude Code}},
  author       = {{Anthropic}},
  year         = {2026},
  howpublished = {\\url{https://www.anthropic.com/claude-code}},
  note         = {Agentic coding tool; models Claude Opus 5 and Claude Opus 5.5}
}""",
}
ACL_IDS = {"wang-manning-2012-baselines": "P12-2018", "pang-lee-2005-seeing": "P05-1015", "daume-iii-2007-frustratingly": "P07-1033",
           "socher-etal-2013-recursive": "D13-1170", "maas-etal-2011-learning": "P11-1015"}
URL_ONLY = {  # classic papers with neither DOI nor anthology id: title/author verified via proceedings page
    "lee2013pseudo": """@inproceedings{lee2013pseudo,
  title     = {Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks},
  author    = {Lee, Dong-Hyun},
  booktitle = {ICML 2013 Workshop on Challenges in Representation Learning},
  year      = {2013}
}""",
    "ke2017lightgbm": """@inproceedings{ke2017lightgbm,
  title     = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle = {Advances in Neural Information Processing Systems 30},
  year      = {2017}
}""",
}


def get(url: str, accept: str | None = None) -> str:
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            time.sleep(4 + 4 * attempt)
    raise RuntimeError(f"failed: {url}")


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def rekey(bib: str, key: str) -> str:
    bib = re.sub(r"^(@\w+\{)[^,]+,", lambda m: m.group(1) + key + ",", bib.strip(), count=1)
    # DOI-resolver BibTeX writes months as bare words (June, Sept) that are not BibTeX macros
    return re.sub(r"month\s*=\s*([A-Za-z]+)(?=\s*[,}])", lambda m: f"month={m.group(1)[:3].lower()}", bib)


def crossref(title: str, first: str, year: int):
    import json
    q = urllib.parse.quote(f"{title} {first}")  # the author term ranks the published version above preprints
    d = json.loads(get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=8"))
    best = None
    for it in d["message"]["items"]:
        t = (it.get("title") or [""])[0]
        if it.get("subtitle"):
            t = f"{t}: {it['subtitle'][0]}"
        fam = " ".join(a.get("family", "") for a in it.get("author", [])[:1])
        yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        s = sim(t, title)
        preprint = it["DOI"].startswith(("10.31219/", "10.48550/", "10.31234/", "10.1101/")) or it.get("type") == "posted-content"
        if preprint:
            s -= 0.05  # a published version with the same title wins
        if s >= 0.85 and first.lower()[:4] in fam.lower() and (yr is None or abs(yr - year) <= 1):
            if best is None or s > best[0]:
                best = (s, it["DOI"], t, yr)
    return best


_last_arxiv = [0.0]


def arxiv(title: str, first: str):
    wait = 3.5 - (time.time() - _last_arxiv[0])
    if wait > 0:
        time.sleep(wait)
    _last_arxiv[0] = time.time()
    q = urllib.parse.quote(f'ti:"{title}"')
    xml = get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=5")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in ET.fromstring(xml).findall("a:entry", ns):
        t = " ".join(e.find("a:title", ns).text.split())
        authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
        if sim(t, title) >= 0.9 and authors and first.lower()[:4] in authors[0].lower():
            aid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
            aid = re.sub(r"v\d+$", "", aid)
            year = e.find("a:published", ns).text[:4]
            return t, authors, year, aid
    return None


# arXiv ids for preprint-only / non-DOI venues; each is accepted only if arXiv's official BibTeX has the expected title
ARXIV_IDS = {"he2023debertav": "2111.09543", "wang2022e5": "2212.03533", "wang2024me5": "2402.05672",
             "dettmers2023qlora": "2305.14314", "alain2017probes": "1610.01644", "laurer2023zeroshot": "2312.17543",
             "khodak-etal-2018-large": "1704.05579", "hu2022lora": "2106.09685", "kalajdzievski2023rslora": "2312.03732",
             "liu2024dora": "2402.09353", "paischer2024eva": "2410.07170", "zhang2024riemannian": "2402.02347",
             "clark2020electra": "2003.10555", "warner2024modernbert": "2412.13663", "liu2019roberta": "1907.11692",
             "prokhorenkova2018catboost": "1706.09516", "xiao2023cpack": "2309.07597", "mcinnes2018umap": "1802.03426",
             "dodge2020finetuning": "2002.06305", "mosbach2021stability": "2006.04884", "zhang2015character": "1509.01626",
             "beaglehole2025xrfm": "2508.10053"}


def arxiv_official(key: str, title: str, first: str, venue: str) -> str | None:
    """Official arXiv BibTeX (arxiv.org/bibtex/<id>), accepted only if its title and first author match."""
    wait = 6.0 - (time.time() - _last_arxiv[0])
    if wait > 0:
        time.sleep(wait)
    _last_arxiv[0] = time.time()
    local = HERE / "arxiv_bib" / f"{ARXIV_IDS[key]}.bib"  # pre-fetched official BibTeX (arXiv throttles scripts)
    bib = local.read_text() if local.exists() and local.stat().st_size > 50 else get(f"https://arxiv.org/bibtex/{ARXIV_IDS[key]}")
    t = " ".join(re.search(r"title=\{(.+?)\},\s*\n", bib, re.S).group(1).split())
    au = re.search(r"author=\{(.+?)\},", bib, re.S).group(1)
    if sim(t, title) < 0.9 or first.lower()[:4] not in au.split(" and ")[0].lower():
        return None
    bib = rekey(bib, key)
    if venue:
        bib = bib.rstrip().rstrip("}").rstrip().rstrip(",") + f",\n      note={{{venue}}}\n}}"
    return bib


def arxiv_bib(key: str, t: str, authors: list[str], year: str, aid: str, venue: str) -> str:
    def fmt(n: str) -> str:
        parts = n.split()
        return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else n
    t_b = re.sub(r"([A-Z]{2,}[a-zA-Z0-9]*)", r"{\1}", t)
    note = f",\n  note      = {{{venue}}}" if venue else ""
    return (f"@article{{{key},\n  title     = {{{t_b}}},\n  author    = {{{' and '.join(fmt(a) for a in authors)}}},\n"
            f"  journal   = {{arXiv preprint arXiv:{aid}}},\n  year      = {{{year}}}{note}\n}}")


def resolve(key: str, title: str, first: str, year: int, src: str, venue: str) -> tuple[str | None, str]:
    """(bibtex or None, one-line status) for one entry."""
    if key in ACL_IDS:
        bib = rekey(get(f"https://aclanthology.org/{ACL_IDS[key]}.bib"), key)
        ok = sim(re.search(r"title\s*=\s*[\"{](.+?)[\"}],?\n", bib, re.S).group(1), title) >= 0.85
        return (bib if ok else None), f"{'OK ' if ok else 'CHK'} {key}: ACL {ACL_IDS[key]}"
    if key in URL_ONLY:
        return URL_ONLY[key], f"MAN {key}: manual (no DOI / arXiv)"
    if src == "doi":
        hit = crossref(title, first, year)
        if hit:
            doi = hit[1]
            if doi.startswith("10.18653/v1/"):
                aid = doi.split("/v1/")[1]
                if re.fullmatch(r"[a-z]\d{2}-\d{4}", aid):
                    aid = aid.upper()
                bib = get(f"https://aclanthology.org/{aid}.bib")
            else:
                bib = get(f"https://doi.org/{doi}", accept="application/x-bibtex")
            return rekey(bib, key), f"OK  {key}: doi {doi} (title sim {hit[0]:.2f})"
    if key in ARXIV_IDS:
        bib = arxiv_official(key, title, first, venue)
        if bib:
            return bib, f"OK  {key}: arXiv {ARXIV_IDS[key]} (official bibtex)"
        return None, f"CHK {key}: arXiv {ARXIV_IDS[key]} title/author mismatch"
    return None, f"MISS {key}: no verified match for {title!r}"


def main() -> None:
    import json
    cache_f = HERE / "bib_cache.json"  # verified entries persist across runs; only misses are retried
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    out, report = [], []
    for key, title, first, year, src, venue in REFS:
        if key not in cache:
            time.sleep(1.0)
            try:
                bib, line = resolve(key, title, first, year, src, venue)
            except Exception as e:  # noqa: BLE001
                bib, line = None, f"ERR {key}: {type(e).__name__}: {str(e)[:90]}"
            print(line, flush=True)
            if bib and line.startswith(("OK", "MAN")):
                cache[key] = {"bib": bib.strip(), "report": line}
                cache_f.write_text(json.dumps(cache, indent=1))
            else:
                report.append(line)
                continue
        out.append(cache[key]["bib"]); report.append(cache[key]["report"])
    out += list(MANUAL_BIB.values())
    (HERE / "references.bib").write_text("% Generated by paper/build_bib.py from Crossref / ACL Anthology / arXiv "
                                         "(see bib_report.txt)\n\n" + "\n\n".join(out) + "\n")
    (HERE / "bib_report.txt").write_text("\n".join(report) + "\n")
    bad = [r for r in report if not r.startswith(("OK", "MAN"))]
    print(f"{len(report) - len(bad)} verified/manual, {len(bad)} problems")
    for r in bad:
        print(r)


if __name__ == "__main__":
    sys.exit(main())
