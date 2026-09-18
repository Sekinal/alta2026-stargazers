"""Attack 2b: near-duplicate / same-thread leakage across folds.

Scans all pool pairs for (a) high embedding cosine (bge_large, mpnet), (b) normalised-text
containment (one text inside another, >= 25 chars) and (c) shared word 8-grams (quoted text),
and reports how many of those pairs straddle folds and how often their labels agree.
"""

import re

from _lib import *  # noqa: F403


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def cos_pairs(E, thr):
    out = []
    for i0 in range(0, len(E), 1024):
        S = E[i0:i0 + 1024] @ E.T
        ii, jj = np.nonzero(S >= thr)
        keep = jj > ii + i0
        out += list(zip((ii[keep] + i0).tolist(), jj[keep].tolist(), S[ii[keep], jj[keep]].tolist()))
    return out


def summarise(name, pairs, d):
    if not pairs:
        print(name, "no pairs")
        return
    i = np.array([p[0] for p in pairs]); j = np.array([p[1] for p in pairs])
    cross = d.fold[i] != d.fold[j]
    rows = {"pairs": len(pairs), "cross_fold": int(cross.sum()),
            "rows_involved(cross)": len(set(i[cross].tolist()) | set(j[cross].tolist()))}
    for t in TASKS:
        rows[f"{t}_agree(cross)"] = float((d.y[t][i[cross]] == d.y[t][j[cross]]).mean()) if cross.any() else np.nan
    rows["same_segment(cross)"] = float((d.segment()[i[cross]] == d.segment()[j[cross]]).mean()) if cross.any() else np.nan
    print(name, rows)
    return i[cross], j[cross]


def main():
    d = base_eval()
    texts = load_pool()["text"].to_list()
    nt = [norm(t) for t in texts]
    for m in ("bge_large", "mpnet"):
        E = emb(m).astype(np.float32)
        for thr in (0.98, 0.95, 0.92, 0.90):
            summarise(f"{m} cos>={thr}", cos_pairs(E, thr), d)
    # normalised exact duplicates
    groups = {}
    for k, s in enumerate(nt):
        groups.setdefault(s, []).append(k)
    pairs = [(g[a], g[b], 1.0) for g in groups.values() if len(g) > 1
             for a in range(len(g)) for b in range(a + 1, len(g))]
    summarise("normalised-exact", pairs, d)
    # containment (short text inside a longer one), texts >= 25 chars normalised
    idx = sorted([k for k, s in enumerate(nt) if len(s) >= 25], key=lambda k: len(nt[k]))
    pairs = []
    for a_pos, a in enumerate(idx):
        sa = nt[a]
        for b in idx[a_pos + 1:]:
            if len(nt[b]) > len(sa) and sa in nt[b]:
                pairs.append((min(a, b), max(a, b), 1.0))
    ci = summarise("containment", pairs, d)
    for a, b, _ in pairs[:8]:
        print("  ", d.fold[a], d.fold[b], repr(texts[a][:70]), "<in>", repr(texts[b][:70]))
    # shared word 8-grams
    grams = {}
    for k, s in enumerate(nt):
        w = s.split()
        for g in {" ".join(w[x:x + 8]) for x in range(len(w) - 7)}:
            grams.setdefault(g, set()).add(k)
    pairs = set()
    for ks in grams.values():
        if 1 < len(ks) <= 20:
            ks = sorted(ks)
            pairs |= {(ks[a], ks[b], 1.0) for a in range(len(ks)) for b in range(a + 1, len(ks))}
    pairs = [p for p in pairs if nt[p[0]] != nt[p[1]]]
    summarise("shared 8-gram (non-identical)", sorted(pairs), d)
    for a, b, _ in sorted(pairs)[:10]:
        print("  ", d.fold[a], d.fold[b], d.y["sarc"][a], d.y["sarc"][b], repr(texts[a][:80]), "|", repr(texts[b][:80]))


if __name__ == "__main__":
    main()
