"""One test per bug in FAILURES.md (F-006 .. F-017). Each reproduces the original failure."""
import json

from reverify.data.build import UNLABELABLE_CUES
from reverify.eval.metrics import score_doc
from reverify.format import parse_output
from reverify.hardware import adapt
from reverify.verifier import verify


def _doc(sents, names, rels):
    return {"id": "d", "sents": sents,
            "entities": [{"id": f"e{i}", "names": n if isinstance(n, list) else [n], "mentions": []} for i, n in enumerate(names)],
            "relations": [{"subject_id": s, "relation": r, "object_id": o, "evidence": ev, "evidence_source": "gold",
                           "status": "explicit"} for s, r, o, ev in rels]}


def test_f006_grounding_is_token_bounded():
    r = verify([{"s": "Ma", "r": "OWNS", "o": "Beta Corp", "ev": [0]}], ["Marta owns Beta Corp."])
    assert r.kept == [] and r.dropped[0][1] == "entity_not_in_text"
    r = verify([{"s": "Marta", "r": "OWNS", "o": "Beta Corp", "ev": [0]}], ["Marta owns Beta Corp."])
    assert len(r.kept) == 1


def test_f007_month_may_is_not_a_hedge():
    r = verify([{"s": "Acme", "r": "ACQUIRED", "o": "Beta", "ev": [0]}], ["In May 2020, Acme acquired Beta."],
               hedge_filter=True)
    assert len(r.kept) == 1
    r = verify([{"s": "Acme", "r": "ACQUIRED", "o": "Beta", "ev": [0]}], ["Acme may acquire Beta."], hedge_filter=True)
    assert r.kept == []


def test_f008_malformed_fields_never_crash_scorer():
    doc = _doc(["A owns B."], ["A", "B"], [("e0", "OWNS", "e1", [0])])
    for raw in ['{"relations":[{"s":null,"r":"OWNS","o":"B","ev":[0]}]}', '{"relations":[{"s":"A","r":"OWNS","o":"B","ev":0}]}',
                '{"relations":[{"s":["A"],"r":"OWNS","o":"B"}]}', '{"relations":{"s":"A"}}', '{"relations":[{"s":"A","r":"OWNS","o":"B","ev":"0, 1"}]}']:
        rels, _ = parse_output(raw)
        for r in rels:
            assert isinstance(r["s"], str) and isinstance(r["o"], str) and all(isinstance(i, int) for i in r["ev"])
        score_doc(doc, rels)
    rels, _ = parse_output('{"relations":[{"s":"A","r":"OWNS","o":"B","ev":"0, 1"}]}')
    assert rels[0]["ev"] == [0, 1]


def test_f009_lenient_unique_entity_matching():
    doc = _doc(["Ostrava Motors makes the Kestrel electric scooter.", "Coralstone Bank and Coralstone Group differ."],
               ["Ostrava Motors", "Kestrel", "Coralstone Bank", "Coralstone Group"],
               [("e0", "MANUFACTURER_OF", "e1", [0])])
    pred = [{"s": "Ostrava Motors", "r": "MANUFACTURER_OF", "o": "Kestrel electric scooter", "ev": [0]}]
    assert score_doc(doc, pred)["counts"]["tp"] == 1
    assert score_doc(doc, pred, match="strict")["counts"]["tp"] == 0
    # ambiguous containment ("Coralstone" in two entities) and single-token fragments never match
    assert score_doc(doc, [{"s": "Coralstone", "r": "OWNS", "o": "Kestrel", "ev": [1]}])["counts"]["fp_unresolved_entity"] == 1
    assert score_doc(doc, [{"s": "Motors", "r": "MANUFACTURER_OF", "o": "Kestrel", "ev": [0]}])["counts"]["tp"] == 0


def test_f010_t4_profile_fits_memory():
    t = adapt({"cuda": True, "gpu_mem_gb": 14.7, "bf16": False}, 0.5, 1024)
    assert t["precision"] == "fp16" and t["load_dtype"] == "fp32"
    logits_gb = t["micro_batch"] * 1024 * 152_000 * 10 / 1e9
    assert logits_gb < 0.5 * 14.7 and t["micro_batch"] * t["grad_accum"] == 16


def test_f011_unlabelable_cue_filter():
    for s in ["It was acquired by Sony in 2004.", "The joint venture began in 1999.", "Investors include Acme."]:
        assert UNLABELABLE_CUES.search(s)
    assert not UNLABELABLE_CUES.search("Acme owns the stadium.")


def test_split_sentences_for_space_input():
    from reverify.format import split_sentences
    t = ("Before founding Tidepool Health, Nadia Hesse was a nurse at St. Brigid's Hospital. Acme Inc. bought "
         "Beta in May. J. Smith is the CEO!\nA new line starts here")
    assert split_sentences(t) == ["Before founding Tidepool Health, Nadia Hesse was a nurse at St. Brigid's Hospital.",
                                  "Acme Inc. bought Beta in May.", "J. Smith is the CEO!", "A new line starts here"]
    assert split_sentences("") == []
