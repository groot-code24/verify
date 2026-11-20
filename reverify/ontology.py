"""Relation ontology loader and dataset label mapping layer."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Ships inside the package so `pip install` (e.g. the Hugging Face Space) has it (FAILURES.md F-016).
DEFAULT_ONTOLOGY = Path(__file__).resolve().parent / "resources" / "ontology_v0.1.yaml"


@dataclass(frozen=True)
class Relation:
    name: str
    meaning: str
    symmetric: bool
    subject_types: tuple
    object_types: tuple


@dataclass
class Ontology:
    version: str
    relations: dict  # name -> Relation
    docred_map: dict  # P-id -> (relation name, inverse: bool)

    @property
    def names(self) -> list[str]:
        return list(self.relations)

    def is_symmetric(self, rel: str) -> bool:
        return self.relations[rel].symmetric

    def describe(self) -> str:
        return "\n".join(f"{r.name}: {r.meaning}" for r in self.relations.values())


@lru_cache(maxsize=4)
def load_ontology(path: str | Path = DEFAULT_ONTOLOGY) -> Ontology:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rels, dmap = {}, {}
    for name, spec in cfg["relations"].items():
        rels[name] = Relation(name, spec["meaning"].strip(), bool(spec.get("symmetric", False)),
                              tuple(spec.get("subject", [])), tuple(spec.get("object", [])))
        for src in (spec.get("sources") or {}).get("docred", []) or []:
            inverse = src.endswith("^-1")
            pid = src.replace("^-1", "")
            if pid in dmap:
                raise ValueError(f"DocRED label {pid} mapped twice ({dmap[pid][0]}, {name})")
            dmap[pid] = (name, inverse)
    return Ontology(str(cfg["version"]), rels, dmap)
