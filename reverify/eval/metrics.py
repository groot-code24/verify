"""Scoring program shared by every system (RE-Verify, baselines, competitors).

Inputs: gold docs in unified format; predictions {doc_id: [{"s","r","o","ev"}...]} AFTER whatever
post-processing the system applies. The scorer never looks at system identity.

Matching (match="lenient", the primary mode since protocol v1.1): a predicted argument resolves to
every gold entity with a mention whose normalized form equals the prediction. If none does, it resolves
to the single entity (unique, else no match) whose mention is contained in the prediction with at most
3 extra tokens ("Kestrel electric scooter" -> "Kestrel"), or, for predictions of >= 2 tokens, whose
mention contains the prediction at token boundaries ("Ardent Labs" inside "Ardent Labs Inc").
Single-token predictions never match by containment, so "Labs" or "Inc" cannot score.
match="strict" = exact normalized equality only. Both are reported. A prediction is a TP if any resolved
(subject, relation, object) is an unmatched explicit gold triple (both orders for symmetric
relations). Predictions matching only an *implicit* gold triple are ignored (neither TP nor FP).
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from reverify.ontology import load_ontology
from reverify.verifier import contains, norm


def _resolve(doc: dict, s: str, match: str = "lenient") -> set[str]:
    n = norm(s) if isinstance(s, str) else ""
    if not n:
        return set()
    exact = {e["id"] for e in doc["entities"] if any(norm(x) == n for x in e["names"])}
    if exact or match == "strict":
        return exact
    ntok = len(n.split())
    cands = set()
    for e in doc["entities"]:
        for x in e["names"]:
            g = norm(x)
            if not g:
                continue
            gtok = len(g.split())
            if (contains(n, g) and ntok - gtok <= 3) or (ntok >= 2 and contains(g, n)):
                cands.add(e["id"])
                break
    return cands if len(cands) == 1 else set()


def score_doc(doc: dict, preds: list[dict], match: str = "lenient") -> dict:
    onto = load_ontology()
    explicit = {(r["subject_id"], r["relation"], r["object_id"]): r for r in doc["relations"] if r["status"] == "explicit"}
    implicit = {(r["subject_id"], r["relation"], r["object_id"]) for r in doc["relations"] if r["status"] != "explicit"}
    text_n = norm(" ".join(doc["sents"]))
    matched: set = set()
    out = Counter()
    per_rel = defaultdict(Counter)
    ev_scores = []
    for p in preds:
        rel = p.get("r")
        S, O = _resolve(doc, p.get("s", ""), match), _resolve(doc, p.get("o", ""), match)
        cands = [(s, rel, o) for s in S for o in O if s != o]
        if rel in onto.relations and onto.is_symmetric(rel):
            cands += [(o, rel, s) for (s, _, o) in cands]
        hit = next((c for c in cands if c in explicit and c not in matched), None)
        grounded = all(isinstance(p.get(k), str) and contains(text_n, norm(p[k])) for k in ("s", "o"))
        out["pred"] += 1
        out["ungrounded"] += int(not grounded)
        if hit:
            matched.add(hit)
            out["tp"] += 1
            per_rel[rel]["tp"] += 1
            g = explicit[hit]
            if g["evidence_source"] == "gold":
                P = {i for i in (p.get("ev") or []) if isinstance(i, int)}
                G = set(g["evidence"])
                inter = len(P & G)
                prec = inter / len(P) if P else 0.0
                rec = inter / len(G) if G else 0.0
                f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                ev_scores.append((f, P == G))
            continue
        if any(c in implicit for c in cands) or any(c in matched for c in cands):
            out["ignored"] += 1
            out["pred"] -= 1
            continue
        out["fp"] += 1
        per_rel[rel]["fp"] += 1
        if any((o, rel, s) in explicit for (s, _, o) in cands):
            out["fp_reversed"] += 1
        elif any((s, r2, o) in explicit or (o, r2, s) in explicit for (s, _, o) in cands for r2 in onto.names):
            out["fp_wrong_relation"] += 1
        elif not S or not O:
            out["fp_unresolved_entity"] += 1
        else:
            out["fp_spurious"] += 1
    out["gold"] = len(explicit)
    out["fn"] = len(explicit) - len(matched)
    for k, g in explicit.items():
        per_rel[k[1]]["gold"] += 1
    return {"counts": out, "per_rel": per_rel, "ev": ev_scores,
            "abstain_case": len(explicit) == 0, "abstained": out["pred"] == 0}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(doc_scores: list[dict]) -> dict:
    c = Counter()
    per_rel = defaultdict(Counter)
    ev = []
    abst_tot = abst_ok = 0
    for d in doc_scores:
        c.update(d["counts"])
        for r, cc in d["per_rel"].items():
            per_rel[r].update(cc)
        ev += d["ev"]
        if d["abstain_case"]:
            abst_tot += 1
            abst_ok += int(d["abstained"])
    p, r, f = _prf(c["tp"], c["fp"], c["fn"])
    rel_f = {}
    for rel, cc in per_rel.items():
        if cc["gold"] > 0:
            rel_f[rel] = round(_prf(cc["tp"], cc["fp"], cc["gold"] - cc["tp"])[2], 4)
    pred = max(c["pred"], 1)
    return {
        "docs": len(doc_scores), "gold": c["gold"], "pred": c["pred"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        "precision": round(p, 4), "recall": round(r, 4), "micro_f1": round(f, 4),
        "macro_f1": round(sum(rel_f.values()) / len(rel_f), 4) if rel_f else 0.0,
        "per_relation_f1": dict(sorted(rel_f.items())),
        "reversal_error_rate": round(c["fp_reversed"] / pred, 4),
        "wrong_relation_rate": round(c["fp_wrong_relation"] / pred, 4),
        "unsupported_rate": round(c["fp"] / pred, 4),
        "ungrounded_rate": round(c["ungrounded"] / pred, 4),
        "fp_breakdown": {k: c[k] for k in ("fp_reversed", "fp_wrong_relation", "fp_unresolved_entity", "fp_spurious")},
        "ignored_implicit_matches": c["ignored"],
        "evidence_f1": round(sum(x[0] for x in ev) / len(ev), 4) if ev else None,
        "evidence_exact": round(sum(x[1] for x in ev) / len(ev), 4) if ev else None,
        "abstention_accuracy": round(abst_ok / abst_tot, 4) if abst_tot else None,
        "abstention_cases": abst_tot,
    }


def score(gold_docs: list[dict], preds: dict, match: str = "lenient") -> tuple[dict, list[dict]]:
    ds = [score_doc(d, preds.get(d["id"], []), match) for d in gold_docs]
    return aggregate(ds), ds


def bootstrap_ci(doc_scores: list[dict], metric="micro_f1", n=1000, seed=0) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [doc_scores[rng.randrange(len(doc_scores))] for _ in doc_scores]
        vals.append(aggregate(sample)[metric])
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def paired_bootstrap(ds_a: list[dict], ds_b: list[dict], metric="micro_f1", n=1000, seed=0) -> dict:
    """P(system A <= system B) under document resampling; ds_a/ds_b aligned by document."""
    assert len(ds_a) == len(ds_b)
    rng = random.Random(seed)
    wins = 0
    deltas = []
    for _ in range(n):
        idx = [rng.randrange(len(ds_a)) for _ in ds_a]
        a = aggregate([ds_a[i] for i in idx])[metric]
        b = aggregate([ds_b[i] for i in idx])[metric]
        deltas.append(a - b)
        wins += a > b
    deltas.sort()
    return {"p_a_not_better": round(1 - wins / n, 4), "delta_ci95": (round(deltas[int(0.025 * n)], 4),
            round(deltas[int(0.975 * n) - 1], 4)), "mean_delta": round(sum(deltas) / n, 4)}
