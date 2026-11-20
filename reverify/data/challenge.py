"""Load the hand-authored challenge benchmark into the unified format.

Authoring format (benchmark/challenge_v*.jsonl):
  {"id","category","sents":[...],"rels":[[subject, RELATION, object, [evidence ids]]]}
Entities are derived from the strings used in rels. Every gold relation is 'explicit' with
gold evidence. Items with empty rels are abstention cases.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reverify.ontology import load_ontology
from reverify.verifier import norm

ROOT = Path(__file__).resolve().parents[2]
CHALLENGE = ROOT / "benchmark" / "challenge_v1.jsonl"
# sha256(norm(name)) of every challenge entity. Public stand-in for the private test file, so the synthetic
# generator can avoid challenge entities without the test items being published (FAILURES.md F-017).
ENTITY_HASHES = ROOT / "benchmark" / "challenge_v1.entity_hashes.txt"


def available(path: Path = CHALLENGE) -> bool:
    return Path(path).exists()


def name_hash(name: str) -> str:
    return hashlib.sha256(norm(name).encode("utf-8")).hexdigest()


def banned_name_hashes() -> set[str]:
    """Hashes of challenge entity names: from the challenge file when present (and refresh the public list),
    otherwise from the published hash list."""
    if available():
        hashes = {name_hash(n) for d in load_challenge() for e in d["entities"] for n in e["names"]}
        ENTITY_HASHES.write_text("\n".join(sorted(hashes)) + "\n", encoding="utf-8")
        return hashes
    if ENTITY_HASHES.exists():
        return {l.strip() for l in ENTITY_HASHES.read_text(encoding="utf-8").splitlines() if l.strip()}
    raise FileNotFoundError("Neither benchmark/challenge_v1.jsonl nor its entity hash list is present.")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_challenge(path: Path = CHALLENGE) -> list[dict]:
    onto = load_ontology()
    docs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        names: dict[str, str] = {}
        for s, _, o, _ in it["rels"]:
            for x in (s, o):
                names.setdefault(norm(x), x)
        ents = [{"id": f"e{i}", "type": "UNK", "names": [n],
                 "mentions": [{"sent_id": j, "text": n} for j, st in enumerate(it["sents"]) if norm(n) in norm(st)]}
                for i, n in enumerate(names.values())]
        idx = {norm(e["names"][0]): e["id"] for e in ents}
        rels = []
        for s, r, o, ev in it["rels"]:
            if r not in onto.relations:
                raise ValueError(f"{it['id']}: unknown relation {r}")
            for x in (s, o):
                if norm(x) not in norm(" ".join(it["sents"])):
                    raise ValueError(f"{it['id']}: '{x}' not in text")
            if not ev or max(ev) >= len(it["sents"]):
                raise ValueError(f"{it['id']}: bad evidence {ev}")
            si, oi = idx[norm(s)], idx[norm(o)]
            if onto.is_symmetric(r):
                si, oi = sorted([si, oi], key=lambda e: int(e[1:]))
            rels.append({"subject_id": si, "relation": r, "object_id": oi, "evidence": ev,
                         "evidence_source": "gold", "status": "explicit", "source_labels": ["curated"]})
        docs.append({"id": it["id"], "dataset": "challenge_v1", "split": "test", "category": it["category"],
                     "title": it["id"], "sents": it["sents"], "entities": ents, "relations": rels,
                     "provenance": {"file": str(Path(path).name), "author": "curated (Claude Opus 5.5 session)"}})
    return docs
