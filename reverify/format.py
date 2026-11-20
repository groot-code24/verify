"""Prompt construction, target serialization and robust output parsing.

Model-facing output (compact, JSON-valid, grammar-constrainable):
    {"relations":[{"s":"Elon Musk","r":"FOUNDER_OF","o":"SpaceX","ev":[0]}]}
An empty list means the model abstains (no explicitly supported relation).

Public API output (see to_public_json) expands evidence ids to exact sentence text.
"""
from __future__ import annotations

import json
import re

from reverify.ontology import Ontology, load_ontology

SYSTEM_SHORT = (
    "You are RE-Verify. Extract only relations that the text explicitly states, using the allowed "
    "relation labels in their canonical direction. Cite evidence as sentence numbers. Do not output "
    "speculative, planned, rumored, negated or merely implied relations. If none, return an empty list. "
    "Output JSON only."
)


ONE_SHOT = (
    "\n\nExample input:\n[0] Orbis Foods was founded by Karl Meyer in 1990.\n[1] Last year it was acquired by "
    "Delta Brands.\n[2] Delta Brands may also buy Sunfield Dairy.\n"
    'Example output: {"relations":[{"s":"Karl Meyer","r":"FOUNDER_OF","o":"Orbis Foods","ev":[0]},'
    '{"s":"Delta Brands","r":"ACQUIRED","o":"Orbis Foods","ev":[1]}]}\n'
    "(Sentence 2 is speculative, so no relation is extracted from it.)"
)


def system_prompt(onto: Ontology | None = None, with_definitions: bool = True, one_shot: bool = False) -> str:
    onto = onto or load_ontology()
    if not with_definitions:
        return SYSTEM_SHORT + "\nLabels: " + ", ".join(onto.names)
    return (
        SYSTEM_SHORT
        + "\n\nAllowed relations (subject -> object):\n"
        + onto.describe()
        + '\n\nFormat: {"relations":[{"s":"<subject as written in text>","r":"<LABEL>",'
          '"o":"<object as written in text>","ev":[<sentence numbers>]}]}'
        + (ONE_SHOT if one_shot else "")
    )


_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "inc", "ltd", "co", "corp", "no", "vs", "u.s", "u.k",
           "e.g", "i.e", "etc", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


def split_sentences(text: str) -> list[str]:
    """Rule-based splitter for raw user text (the model expects numbered sentences).
    Splits on newlines and on . ! ? followed by whitespace and an upper-case/quote/digit start,
    except after common abbreviations and single initials ("J. Smith")."""
    out = []
    for para in re.split(r"\n\s*\n|\n", text.strip()):
        para = para.strip()
        start = 0
        for m in re.finditer(r"[.!?][\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])", para):
            prev = para[start:m.start()].split()
            last = prev[-1].lower().rstrip(".") if prev else ""
            if last in _ABBREV or (len(last) == 1 and last.isalpha()):
                continue
            out.append(para[start:m.end()].strip())
            start = m.end()
        if para[start:].strip():
            out.append(para[start:].strip())
    return out


def user_prompt(sents: list[str]) -> str:
    return "Text:\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))


def entity_surface(doc: dict, eid: str, evidence: list[int] | None = None) -> str:
    """Pick the surface string for an entity: first mention inside evidence sentences, else first mention."""
    ent = next(e for e in doc["entities"] if e["id"] == eid)
    if evidence:
        for m in ent["mentions"]:
            if m["sent_id"] in evidence:
                return m["text"]
    return ent["names"][0] if ent["names"] else ent["mentions"][0]["text"]


def target_relations(doc: dict) -> list[dict]:
    """Training target: explicit relations only, deterministic order."""
    out = []
    for r in doc["relations"]:
        if r["status"] != "explicit":
            continue
        ev = r["evidence"]
        out.append({"s": entity_surface(doc, r["subject_id"], ev), "r": r["relation"],
                    "o": entity_surface(doc, r["object_id"], ev), "ev": ev})
    out.sort(key=lambda x: (min(x["ev"]) if x["ev"] else 99, x["s"], x["r"], x["o"]))
    return out


def serialize_target(rels: list[dict]) -> str:
    return json.dumps({"relations": rels}, ensure_ascii=False, separators=(",", ":"))


_OBJ_RE = re.compile(r"\{[^{}]*\}")


def _clean(r) -> dict | None:
    """Coerce one model-produced relation to {s:str, r:str, o:str, ev:[int]} or reject it.
    Raw model output can contain null / numbers / lists in any field; before this, a single
    `"s": null` crashed the scorer (FAILURES.md F-008)."""
    if not isinstance(r, dict):
        return None
    out = {}
    for k in ("s", "r", "o"):
        v = r.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = str(v)
        if not isinstance(v, str) or not v.strip():
            return None
        out[k] = v.strip()
    ev = r.get("ev", [])
    if isinstance(ev, (int, str)):
        ev = [ev]
    if not isinstance(ev, list):
        ev = []
    ids = []
    for x in ev:
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            ids.append(x)
        elif isinstance(x, str):
            ids += [int(t) for t in re.findall(r"\d+", x)]
    out["ev"] = ids
    return out


def parse_output(text: str) -> tuple[list[dict], bool]:
    """Return (relations, parse_ok). Tolerates prose around JSON and truncated output.
    Every returned relation is a sanitized dict (see _clean)."""
    text = text.strip()
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    rels = obj.get("relations", []) if isinstance(obj, dict) else []
                    if not isinstance(rels, list):
                        return [], False
                    return [c for c in map(_clean, rels) if c], True
    # fallback: salvage complete inner objects from truncated/invalid JSON
    rels = []
    for m in _OBJ_RE.finditer(text):
        try:
            c = _clean(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
        if c:
            rels.append(c)
    return rels, False


def to_public_json(rels: list[dict], sents: list[str]) -> dict:
    out = []
    for r in rels:
        ev = [i for i in r.get("ev", []) if isinstance(i, int) and 0 <= i < len(sents)]
        out.append({"subject": r["s"], "relation": r["r"], "object": r["o"],
                    "evidence": " ".join(sents[i] for i in ev), "evidence_sentence_ids": ev,
                    "status": r.get("status", "VERIFIED")})
    return {"relations": out, "abstain": len(out) == 0}


def json_grammar(onto: Ontology | None = None) -> str:
    """GBNF grammar for llama.cpp constraining output to the compact schema and allowed labels."""
    onto = onto or load_ontology()
    labels = " | ".join(f'"\\"{n}\\""' for n in onto.names)
    return rf'''
root ::= "{{\"relations\":[" ( rel ( "," rel )* )? "]}}"
rel ::= "{{\"s\":" str ",\"r\":" label ",\"o\":" str ",\"ev\":[" num ( "," num )* "]}}"
label ::= {labels}
str ::= "\"" ( [^"\\\n] | "\\" ["\\/bfnrt] ){{1,80}} "\""
num ::= [0-9] [0-9]?
'''
