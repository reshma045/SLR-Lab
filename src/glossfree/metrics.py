"""Translation metrics for Japanese output (character level, no word segmentation needed).

BLEU-4: corpus-level, character n-grams 1-4, brevity penalty, +1 smoothing for n > 1 (scale 0-100).
chrF:   character n-gram F-score (n = 1..6, beta = 2), averaged over sentences (scale 0-100).
WER:    edit distance between token sequences / reference length (used for sign sequences).
"""
from __future__ import annotations

import math
from collections import Counter

PUNCT = set("。、？！?!., ")


def _chars(s: str) -> list[str]:
    return [c for c in s if c not in PUNCT]


def _ngrams(t: list[str], n: int) -> Counter:
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def bleu4(hyps: list[str], refs: list[str]) -> float:
    match, total = [0] * 4, [0] * 4
    hl = rl = 0
    for h, r in zip(hyps, refs):
        hc, rc = _chars(h), _chars(r)
        hl += len(hc)
        rl += len(rc)
        for n in range(1, 5):
            hn, rn = _ngrams(hc, n), _ngrams(rc, n)
            match[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            total[n - 1] += max(0, len(hc) - n + 1)
    if hl == 0:
        return 0.0
    logp = 0.0
    for n in range(4):
        m, t = (match[n], total[n]) if n == 0 else (match[n] + 1, total[n] + 1)
        if m == 0 or t == 0:
            return 0.0
        logp += math.log(m / t) / 4
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    return 100 * bp * math.exp(logp)


def chrf(hyps: list[str], refs: list[str], n_max: int = 6, beta: float = 2.0) -> float:
    scores = []
    for h, r in zip(hyps, refs):
        hc, rc = _chars(h), _chars(r)
        ps, rs = [], []
        for n in range(1, n_max + 1):
            hn, rn = _ngrams(hc, n), _ngrams(rc, n)
            if not hn and not rn:
                continue
            m = sum(min(c, rn[g]) for g, c in hn.items())
            ps.append(m / max(1, sum(hn.values())))
            rs.append(m / max(1, sum(rn.values())))
        p, rec = (sum(ps) / len(ps), sum(rs) / len(rs)) if ps else (0.0, 0.0)
        b2 = beta * beta
        scores.append(0.0 if p + rec == 0 else (1 + b2) * p * rec / (b2 * p + rec))
    return 100 * sum(scores) / max(1, len(scores))


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def wer(hyps: list[list[str]], refs: list[list[str]]) -> float:
    errs = sum(edit_distance(h, r) for h, r in zip(hyps, refs))
    return errs / max(1, sum(len(r) for r in refs))


def exact(hyps: list[str], refs: list[str]) -> float:
    return sum(_chars(h) == _chars(r) for h, r in zip(hyps, refs)) / max(1, len(refs))
