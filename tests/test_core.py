import random

import pytest

from reverify.data.challenge import available, load_challenge, name_hash
from reverify.data.redocred import convert_doc, detok
from reverify.data.synthetic import Pools, make_doc, to_unified, validate_doc
from reverify.eval.metrics import score, score_doc
from reverify.format import json_grammar, parse_output, serialize_target, target_relations, to_public_json
from reverify.ontology import load_ontology
from reverify.verifier import norm, verify


def _doc(sents, ents, rels):
    return {"id": "d", "sents": sents,
            "entities": [{"id": f"e{i}", "names": [n], "mentions": []} for i, n in enumerate(ents)],
            "relations": [{"subject_id": s, "relation": r, "object_id": o, "evidence": ev,
                           "evidence_source": "gold", "status": st} for s, r, o, ev, st in rels]}


# ---------- ontology ----------
def test_ontology_canonical_and_inverse_mapping():
    o = load_ontology()
    assert o.docred_map["P112"] == ("FOUNDER_OF", True)       # founded by -> inverse
    assert o.docred_map["P108"] == ("EMPLOYEE_OF", False)
    assert o.docred_map["P749"] == ("PARENT_ORG_OF", True) and o.docred_map["P355"] == ("PARENT_ORG_OF", False)
    assert "LOCATED_IN" not in o.relations                      # D-005
    assert o.is_symmetric("PARTNERED_WITH") and not o.is_symmetric("ACQUIRED")


def test_redocred_direction_inversion_and_status():
    raw = {"title": "t", "sents": [["Acme", "was", "founded", "by", "Jo", "Lee", "."], ["Unrelated", "."]],
           "vertexSet": [[{"pos": [0, 1], "type": "ORG", "sent_id": 0, "name": "Acme"}],
                         [{"pos": [4, 6], "type": "PER", "sent_id": 0, "name": "Jo Lee"}]],
           "labels": [{"r": "P112", "h": 0, "t": 1, "evidence": [0]}, {"r": "P17", "h": 0, "t": 1, "evidence": []}]}
    rec, st = convert_doc(raw, "train", 0, load_ontology(), {})
    assert st["unmapped"] == 1
    (r,) = rec["relations"]
    assert (r["subject_id"], r["relation"], r["object_id"]) == ("e1", "FOUNDER_OF", "e0")
    assert r["status"] == "explicit" and r["evidence_source"] == "gold"


def test_heuristic_and_implicit_status():
    raw = {"title": "t", "sents": [["Jo", "joined", "Acme", "."], ["Bo", "slept", "."], ["Zed", "Corp", "."]],
           "vertexSet": [[{"pos": [0, 1], "type": "PER", "sent_id": 0}], [{"pos": [2, 3], "type": "ORG", "sent_id": 0}],
                         [{"pos": [0, 1], "type": "PER", "sent_id": 1}], [{"pos": [0, 2], "type": "ORG", "sent_id": 2}]],
           "labels": [{"r": "P108", "h": 0, "t": 1, "evidence": []}, {"r": "P108", "h": 2, "t": 3, "evidence": []}]}
    rec, _ = convert_doc(raw, "train", 0, load_ontology(), {})
    st = {(r["subject_id"], r["object_id"]): (r["status"], r["evidence_source"]) for r in rec["relations"]}
    assert st[("e0", "e1")] == ("explicit", "heuristic")
    assert st[("e2", "e3")] == ("implicit", "none")


def test_detok():
    assert detok(["Acme", ",", "Inc", ".", "(", "US", ")", "'s"]) == "Acme, Inc. (US)'s"


# ---------- format ----------
def test_parse_valid_truncated_and_prose():
    rels, ok = parse_output('Sure: {"relations":[{"s":"A","r":"OWNS","o":"B","ev":[0]}]} done')
    assert ok and rels[0]["o"] == "B"
    rels, ok = parse_output('{"relations":[{"s":"A","r":"OWNS","o":"B","ev":[0]},{"s":"C","r":"OW')
    assert not ok and len(rels) == 1
    assert parse_output("garbage") == ([], False)


def test_public_json_expands_evidence():
    out = to_public_json([{"s": "A", "r": "OWNS", "o": "B", "ev": [1]}], ["x.", "A owns B."])
    assert out["relations"][0]["evidence"] == "A owns B." and out["abstain"] is False
    assert to_public_json([], ["x"])["abstain"] is True


def test_grammar_lists_all_labels():
    g = json_grammar()
    for n in load_ontology().names:
        assert n in g


# ---------- verifier ----------
def test_verifier_drop_reasons():
    sents = ["Acme acquired Beta.", "Gamma is considering buying Delta."]
    rels = [{"s": "Acme", "r": "ACQUIRED", "o": "Beta", "ev": [0]},
            {"s": "Acme", "r": "BOUGHT", "o": "Beta", "ev": [0]},
            {"s": "Acme", "r": "OWNS", "o": "Zeta", "ev": [0]},
            {"s": "Gamma", "r": "ACQUIRED", "o": "Delta", "ev": [1]},
            {"s": "Acme", "r": "ACQUIRED", "o": "Beta", "ev": [0]}]
    res = verify(rels, sents)
    reasons = [r for _, r in res.dropped]
    assert reasons == ["invalid_relation", "entity_not_in_text", "duplicate"]
    res2 = verify(rels, sents, hedge_filter=True)
    assert "hedged_evidence" in [r for _, r in res2.dropped]
    assert [k["o"] for k in res2.kept] == ["Beta"]


def test_verifier_repairs_out_of_range_evidence():
    res = verify([{"s": "Acme", "r": "ACQUIRED", "o": "Beta", "ev": [7]}], ["x.", "Acme acquired Beta."])
    assert res.kept[0]["ev"] == [1]


def test_norm():
    assert norm("The Riverside Gazette") == norm("riverside gazette")
    # regression F-002: single-letter names must survive normalization
    assert norm("Company A") != norm("Company B")
    assert norm("A") == "a"


# ---------- metrics ----------
def test_metrics_tp_fp_reversal_implicit():
    doc = _doc(["Acme acquired Beta.", "Beta is in Ohio."], ["Acme", "Beta", "Ohio"],
               [("e0", "ACQUIRED", "e1", [0], "explicit"), ("e1", "HEADQUARTERED_IN", "e2", [], "implicit")])
    d = score_doc(doc, [{"s": "Beta", "r": "ACQUIRED", "o": "Acme", "ev": [0]}])
    assert d["counts"]["fp_reversed"] == 1 and d["counts"]["tp"] == 0
    d = score_doc(doc, [{"s": "acme", "r": "ACQUIRED", "o": "Beta", "ev": [0]},
                        {"s": "Beta", "r": "HEADQUARTERED_IN", "o": "Ohio", "ev": [1]}])
    assert d["counts"]["tp"] == 1 and d["counts"]["fp"] == 0 and d["counts"]["ignored"] == 1
    assert d["ev"] == [(1.0, True)]


def test_metrics_symmetric_and_abstention():
    doc = _doc(["A partnered with B."], ["A", "B"], [("e0", "PARTNERED_WITH", "e1", [0], "explicit")])
    assert score_doc(doc, [{"s": "B", "r": "PARTNERED_WITH", "o": "A", "ev": [0]}])["counts"]["tp"] == 1
    empty = _doc(["Nothing here."], [], [])
    agg, _ = score([empty], {"d": []})
    assert agg["abstention_accuracy"] == 1.0
    agg, _ = score([empty], {"d": [{"s": "X", "r": "OWNS", "o": "Y", "ev": [0]}]})
    assert agg["abstention_accuracy"] == 0.0 and agg["fp"] == 1


def test_metrics_no_double_counting():
    doc = _doc(["A owns B."], ["A", "B"], [("e0", "OWNS", "e1", [0], "explicit")])
    agg, _ = score([doc], {"d": [{"s": "A", "r": "OWNS", "o": "B", "ev": [0]}] * 2})
    assert agg["tp"] == 1


# ---------- benchmark & synthetic ----------
needs_challenge = pytest.mark.skipif(not available(), reason="private challenge file not present (public clone)")


@needs_challenge
def test_challenge_gold_self_score_perfect():
    docs = load_challenge()
    agg, _ = score(docs, {d["id"]: target_relations(d) for d in docs})
    assert agg["micro_f1"] == 1.0 and agg["abstention_accuracy"] == 1.0 and agg["evidence_f1"] == 1.0


def test_synthetic_valid_and_disjoint_from_challenge():
    from reverify.data.challenge import banned_name_hashes
    banned = banned_name_hashes()  # works from the published hash list when the challenge file is absent
    rng = random.Random(0)
    pools = Pools(rng, banned)
    for i in range(300):
        d = make_doc(rng, pools, i, "train")
        assert validate_doc(d) == []
        u = to_unified(d)
        for e in u["entities"]:
            if e["names"][0] in pools.people or e["names"][0] in pools.orgs:
                assert name_hash(e["names"][0]) not in banned
        assert serialize_target(target_relations(u)).startswith('{"relations":[')
