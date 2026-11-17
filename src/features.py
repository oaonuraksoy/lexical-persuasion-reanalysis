"""Dictionary-based lexical and structural features.

Lexical rates are counts per 100 alphanumeric word tokens. A token matches a
category if it equals a listed word or a listed word followed by "s"/"es"
(e.g. "risk" also matches "risks"). Category word lists are pairwise disjoint
(checked by ``check_disjoint``).
"""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np

LEXICONS: dict[str, frozenset[str]] = {
    # analytical / evidential cues
    "connectives": frozenset("because therefore consequently thus hence since whereas although furthermore moreover".split()),
    "evidential": frozenset("evidence research data study studies findings analysis report reported statistics survey".split()),
    "hedges": frozenset("may might could often suggests likely tends however potential partially sometimes uncertain nuanced typically generally".split()),
    # affective / relational cues
    "collective": frozenset("we our us ourselves everyone society people citizens humanity community communities together".split()),
    "empathy": frozenset("feel feeling hope fear pain care trust compassion empathy family children dignity".split()),
    "loss": frozenset("risk threat danger jeopardize harm crisis destroy catastrophic vulnerable exploit".split()),
    "certainty": frozenset("must should undoubtedly clearly always never essential crucial proven inevitable".split()),
    "authority": frozenset("university journal institute researchers scientists according federal clinical official regulatory".split()),
    "commercial": frozenset("buy purchase price pricing discount sale shop shipping affordable guarantee coupon retail checkout".split()),
}
# first-person plural pronouns only (a subset of "collective"); sensitivity analysis
PRONOUNS = frozenset("we our us ourselves".split())

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
CITE_RE = re.compile(r"(?:\b10\.\d{4,9}/\S+|https?://\S+|www\.\S+|\[\d+(?:[,–-]\s*\d+)*\])", re.I)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
VOWELS = re.compile(r"[aeiouy]+")


def check_disjoint() -> None:
    for (a, la), (b, lb) in combinations(LEXICONS.items(), 2):
        overlap = la & lb
        assert not overlap, f"lexicons {a} and {b} overlap: {overlap}"


def _match(word: str, lex: frozenset[str]) -> bool:
    if word in lex:
        return True
    if word.endswith("es") and word[:-2] in lex:
        return True
    return word.endswith("s") and word[:-1] in lex


def _syllables(word: str) -> int:
    w = word.lower()
    n = len(VOWELS.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(n, 1)


def text_features(text: str) -> dict[str, float]:
    text = text or ""
    words = WORD_RE.findall(text)
    lw = [w.lower() for w in words]
    n = len(lw)
    out: dict[str, float] = {"n_words": float(n)}
    per100 = 100.0 / n if n else np.nan
    for name, lex in LEXICONS.items():
        out[name] = sum(_match(w, lex) for w in lw) * per100
    out["pronouns"] = sum(w in PRONOUNS for w in lw) * per100
    out["numeric"] = len(NUM_RE.findall(text)) * per100
    out["citations"] = len(CITE_RE.findall(text)) * per100
    out["questions"] = text.count("?") * per100
    sents = [s for s in SENT_RE.split(text.strip()) if WORD_RE.search(s)]
    lens = [len(WORD_RE.findall(s)) for s in sents]
    out["sent_sd"] = float(np.std(lens, ddof=1)) if len(lens) > 1 else 0.0
    out["guiraud"] = len(set(lw)) / np.sqrt(n) if n else np.nan
    if n and sents:
        syl = sum(_syllables(w) for w in words if w.isalpha()) or 1
        out["flesch"] = 206.835 - 1.015 * (n / len(sents)) - 84.6 * (syl / n)
    else:
        out["flesch"] = np.nan
    return out


AFF_REL = ["collective", "empathy", "certainty", "loss"]
ANALYTIC = ["connectives", "evidential", "numeric", "hedges"]
