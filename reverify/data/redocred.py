"""Convert Re-DocRED into the RE-Verify unified document format.

Unified record (one JSON object per line):
{
  "id": "redocred-train-00012", "dataset": "redocred", "split": "train",
  "title": str, "sents": [str, ...],
  "entities": [{"id": "e0", "type": "ORG", "names": [..unique mention strings..],
                "mentions": [{"sent_id": 0, "text": "..."}]}],
  "relations": [{"subject_id": "e3", "relation": "FOUNDER_OF", "object_id": "e0",
                 "evidence": [0], "evidence_source": "gold"|"heuristic"|"none",
                 "status": "explicit"|"implicit", "source_labels": ["P112"]}],
  "provenance": {"repo": ..., "revision": ..., "orig_index": int}
}

status=explicit  -> gold evidence exists, or no gold evidence but subject and object co-occur
                    in at least one sentence (evidence_source=heuristic).
status=implicit  -> neither; excluded from training targets and ignored in scoring.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from reverify.ontology import Ontology, load_ontology

_NO_SPACE_BEFORE = re.compile(r"\s+([,.;:!?%)\]}]|'s\b|'S\b|n't\b|'re\b|'ve\b|'ll\b|'d\b|'m\b)")
_NO_SPACE_AFTER = re.compile(r"([(\[{$])\s+")


def detok(tokens: list[str]) -> str:
    s = " ".join(tokens).replace("``", '"').replace("''", '"').replace("-LRB-", "(").replace("-RRB-", ")")
    s = _NO_SPACE_BEFORE.sub(r"\1", s)
    s = _NO_SPACE_AFTER.sub(r"\1", s)
    return s


def convert_doc(doc: dict, split: str, idx: int, onto: Ontology, provenance: dict) -> tuple[dict, dict]:
    sents = [detok(s) for s in doc["sents"]]
    stats = {"bad_span": 0, "unmapped": 0}
    entities = []
    for eid, cluster in enumerate(doc["vertexSet"]):
        mentions = []
        for m in cluster:
            text = detok(doc["sents"][m["sent_id"]][m["pos"][0]:m["pos"][1]])
            if text not in sents[m["sent_id"]]:
                stats["bad_span"] += 1
                continue
            mentions.append({"sent_id": m["sent_id"], "text": text})
        names = list(dict.fromkeys(m["text"] for m in mentions))
        entities.append({"id": f"e{eid}", "type": cluster[0]["type"], "names": names, "mentions": mentions})

    merged: dict[tuple, dict] = {}
    for lab in doc["labels"]:
        if lab["r"] not in onto.docred_map:
            stats["unmapped"] += 1
            continue
        rel, inverse = onto.docred_map[lab["r"]]
        s, o = (lab["t"], lab["h"]) if inverse else (lab["h"], lab["t"])
        if onto.is_symmetric(rel):
            s, o = min(s, o), max(s, o)
        key = (s, rel, o)
        r = merged.setdefault(key, {"subject_id": f"e{s}", "relation": rel, "object_id": f"e{o}",
                                    "evidence": set(), "source_labels": []})
        r["evidence"].update(lab.get("evidence") or [])
        r["source_labels"].append(lab["r"])

    relations = []
    for (s, rel, o), r in merged.items():
        ev = sorted(r["evidence"])
        if ev:
            r.update(evidence=ev, evidence_source="gold", status="explicit")
        else:
            s_sents = {m["sent_id"] for m in entities[s]["mentions"]}
            o_sents = {m["sent_id"] for m in entities[o]["mentions"]}
            co = sorted(s_sents & o_sents)
            if co:
                r.update(evidence=co, evidence_source="heuristic", status="explicit")
            else:
                r.update(evidence=[], evidence_source="none", status="implicit")
        relations.append(r)

    rec = {"id": f"redocred-{split}-{idx:05d}", "dataset": "redocred", "split": split, "title": doc["title"],
           "sents": sents, "entities": entities, "relations": relations,
           "provenance": {**provenance, "orig_index": idx}}
    return rec, stats


def convert_split(raw_path: Path, split: str, onto: Ontology, provenance: dict) -> tuple[list[dict], dict]:
    docs = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    out, agg = [], {"docs": 0, "bad_span": 0, "unmapped": 0}
    for i, d in enumerate(docs):
        rec, st = convert_doc(d, split, i, onto, provenance)
        out.append(rec)
        agg["docs"] += 1
        for k in ("bad_span", "unmapped"):
            agg[k] += st[k]
    return out, agg


def load_default(split: str, raw_dir: Path = Path("data/raw")) -> tuple[list[dict], dict]:
    manifest = json.loads((raw_dir / "MANIFEST.json").read_text())
    meta = manifest[f"redocred_{split}.json"]
    prov = {"repo": meta["repo"], "file": meta["file"], "revision": meta["revision"]}
    return convert_split(raw_dir / f"redocred_{split}.json", split, load_ontology(), prov)
