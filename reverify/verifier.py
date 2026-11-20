"""Deterministic post-hoc verifier (Architecture B).

Checks every candidate relation against the source text and ontology and drops those that
fail. Each drop is recorded with a reason so the effect of every rule can be measured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from reverify.ontology import Ontology, load_ontology

# Optional hedge/negation cue filter (off by default; measured as a separate ablation).
# Case-insensitive cues, plus "may" matched in lower case only so the month "May" is not a hedge
# (FAILURES.md F-007).
HEDGE_CUES = re.compile(
    r"(?i:\b(plans? to|planned|considering|consider(s|ed)? (buying|acquiring|purchasing)|might|could|"
    r"rumou?red|reportedly|in talks|talks (to|with|about)|proposed|bid for|offer(ed)? to|"
    r"expected to|would|intends? to|seeking to|denied|not|never|no longer|failed to|collapsed|"
    r"called off|abandoned|rejected)\b)|\bmay\b")


def norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Only a leading "the" is dropped. Stripping "a"/"an" anywhere merged "Company A" and
    # "Company B" (FAILURES.md F-002).
    return re.sub(r"^the ", "", s)


def contains(hay: str, needle: str) -> bool:
    """Token-boundary containment on normalized strings. Plain substring matching let a hallucinated
    "Ma" pass as grounded in "Marta ..." (FAILURES.md F-006)."""
    return bool(needle) and f" {needle} " in f" {hay} "


@dataclass
class VerifyResult:
    kept: list = field(default_factory=list)
    dropped: list = field(default_factory=list)  # (relation, reason)


def verify(rels: list[dict], sents: list[str], onto: Ontology | None = None,
           hedge_filter: bool = False, repair_evidence: bool = True) -> VerifyResult:
    onto = onto or load_ontology()
    text_n = norm(" ".join(sents))
    sents_n = [norm(s) for s in sents]
    res, seen = VerifyResult(), set()
    for r in rels:
        s, rel, o = str(r.get("s", "")).strip(), str(r.get("r", "")).strip(), str(r.get("o", "")).strip()
        if rel not in onto.relations:
            res.dropped.append((r, "invalid_relation")); continue
        if not s or not o:
            res.dropped.append((r, "missing_field")); continue
        if norm(s) == norm(o):
            res.dropped.append((r, "self_relation")); continue
        if not contains(text_n, norm(s)) or not contains(text_n, norm(o)):
            res.dropped.append((r, "entity_not_in_text")); continue
        ev = r.get("ev", [])
        ev = [i for i in ev if isinstance(i, int) and 0 <= i < len(sents)] if isinstance(ev, list) else []
        # evidence must mention at least one of the two arguments
        if ev and not any(contains(sents_n[i], norm(s)) or contains(sents_n[i], norm(o)) for i in ev):
            ev = []
        if not ev and repair_evidence:
            ev = [i for i, sn in enumerate(sents_n) if contains(sn, norm(s)) and contains(sn, norm(o))]
        if not ev:
            res.dropped.append((r, "no_valid_evidence")); continue
        if hedge_filter and all(HEDGE_CUES.search(sents[i]) for i in ev):
            res.dropped.append((r, "hedged_evidence")); continue
        if onto.is_symmetric(rel) and norm(o) < norm(s):
            s, o = o, s
        key = (norm(s), rel, norm(o))
        if key in seen:
            res.dropped.append((r, "duplicate")); continue
        seen.add(key)
        res.kept.append({"s": s, "r": rel, "o": o, "ev": sorted(set(ev))})
    return res
