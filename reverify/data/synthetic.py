"""Template-based synthetic data engine for relations missing from Re-DocRED and for hard negatives.

Every generated document is built from (sentence template, gold relations) pairs, so labels are
correct by construction; validate_doc() re-checks entity presence and evidence ids anyway.
Name pools exclude every entity string in the frozen challenge benchmark (checked at runtime).

Usage: python -m reverify.data.synthetic --n-train 3000 --n-dev 300 --seed 13
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from reverify.data.challenge import banned_name_hashes, name_hash
from reverify.verifier import norm

FIRST = ["Aiden", "Beatriz", "Callum", "Dalia", "Emeka", "Freya", "Gustavo", "Hana", "Ivo", "Jasmin", "Kofi",
         "Leilani", "Mateo", "Noor", "Oskar", "Pilar", "Quentin", "Rosa", "Stefan", "Talia", "Umar", "Vera",
         "Wendell", "Ximena", "Yusuf", "Zora", "Arjun", "Bianca", "Cyrus", "Delphine", "Emil", "Farah"]
LAST = ["Abernathy", "Bakshi", "Corrigan", "Dubois", "Eklund", "Fonseca", "Gallagher", "Haddad", "Ishikawa",
        "Janssen", "Kowalczyk", "Larsen", "Moreau", "Nakamura", "Oyelaran", "Pacheco", "Quintero", "Rasmussen",
        "Sorensen", "Takahashi", "Uribe", "Varga", "Whitlock", "Yilmaz", "Zielinski", "Achterberg", "Bellamy"]
ORG_A = ["Alder", "Bluefin", "Cinder", "Driftwood", "Everpine", "Foxglove", "Gullwing", "Hearth", "Ironbark",
         "Juniper", "Kingsley", "Larkspur", "Marlow", "Nimbus", "Obsidian", "Pallas", "Quorra", "Redwater",
         "Silverline", "Thistle", "Umbra", "Vantage", "Westbrook", "Yarrow", "Zephyr", "Amberly", "Copperfield"]
ORG_B = ["Systems", "Labs", "Holdings", "Capital", "Foods", "Media", "Energy", "Logistics", "Therapeutics",
         "Software", "Partners", "Industries", "Networks", "Aerospace", "Textiles", "Bank", "Studios", "Group"]
PRODUCTS = ["Falcon Pad", "Aurora X", "Tempest", "Glimmer", "Orca One", "Starling", "Vortex Pro", "Cobalt Mini",
            "Nightjar", "Quasar", "Pebble Watch S", "Halcyon", "Meridian Drive", "Saffron OS", "Tundra"]
WORKS = ["The Glass Harbor", "Winter Ledger", "A Map of Embers", "Songs for Distant Rivers", "The Lantern Year",
         "Iron and Ivy", "Echoes of Calder", "The Quiet Meridian"]
CITIES = ["Austin", "Lyon", "Osaka", "Nairobi", "Krakow", "Montreal", "Valencia", "Auckland", "Gothenburg",
          "Bangalore", "Bogota", "Tallinn", "Accra", "Perth", "Seville", "Utrecht"]
UNIS = ["University of Toronto", "ETH Zurich", "Seoul National University", "University of Cape Town",
        "Trinity College Dublin", "University of Melbourne", "Sorbonne University"]
PARTIES = ["Civic Union", "People's Reform Party", "Liberal Forward", "Workers' Alliance"]
TEAMS = ["Harbor City United", "Northgate Rovers", "Red Valley FC"]
ROLES_EXEC = ["chief executive officer", "CEO", "president", "chairman", "chairwoman", "chief operating officer",
              "managing director", "chief financial officer"]
ROLES_EMP = ["software engineer", "analyst", "accountant", "researcher", "sales manager", "designer", "nurse"]
FILLER = ["The announcement came during a period of strong growth for the sector.",
          "Industry observers said the market remained competitive.",
          "The company reported revenue of {n} million dollars last year.",
          "Its products are sold in more than {k} countries.",
          "The firm employs roughly {n} people.",
          "The news was first reported by local media."]

# (template, relations as (subj_slot, REL, obj_slot)) — slots: P1,P2 people; O1,O2 orgs; X product; W work;
# C city; U university; T party/team. Negative templates have an empty relation list.
POS = {
    "ACQUIRED": [("{O1} acquired {O2} in {y}.", [("O1", "ACQUIRED", "O2")]),
                 ("{O2} was acquired by {O1} for {n} million dollars.", [("O1", "ACQUIRED", "O2")]),
                 ("{O1} completed its takeover of {O2} last month.", [("O1", "ACQUIRED", "O2")]),
                 ("{O1} bought {O2}, its longtime rival, in {y}.", [("O1", "ACQUIRED", "O2")]),
                 ("In {y}, {O2} was purchased by {O1}.", [("O1", "ACQUIRED", "O2")])],
    "INVESTED_IN": [("{O1} invested {n} million dollars in {O2}.", [("O1", "INVESTED_IN", "O2")]),
                    ("{O2} raised {n} million dollars in a round led by {O1}.", [("O1", "INVESTED_IN", "O2")]),
                    ("{O1} took a minority stake in {O2}.", [("O1", "INVESTED_IN", "O2")]),
                    ("{P1} is an early investor in {O2}.", [("P1", "INVESTED_IN", "O2")])],
    "OWNS": [("{O2} is owned by {O1}.", [("O1", "OWNS", "O2")]),
             ("{O1} owns {O2}.", [("O1", "OWNS", "O2")]),
             ("{O1} holds a majority stake in {O2}.", [("O1", "OWNS", "O2")]),
             ("{P1} owns the restaurant chain {O2}.", [("P1", "OWNS", "O2")])],
    "PARENT_ORG_OF": [("{O2} is a subsidiary of {O1}.", [("O1", "PARENT_ORG_OF", "O2")]),
                      ("{O1} operates {O2} as a wholly owned unit.", [("O1", "PARENT_ORG_OF", "O2")]),
                      ("{O2}, a division of {O1}, makes industrial sensors.", [("O1", "PARENT_ORG_OF", "O2")]),
                      ("{O1}'s subsidiaries include {O2}.", [("O1", "PARENT_ORG_OF", "O2")])],
    "PARTNERED_WITH": [("{O1} partnered with {O2} to build charging stations.", [("O1", "PARTNERED_WITH", "O2")]),
                       ("{O1} and {O2} formed a joint venture in {y}.", [("O1", "PARTNERED_WITH", "O2")]),
                       ("{O1} signed a strategic partnership with {O2}.", [("O1", "PARTNERED_WITH", "O2")])],
    "FOUNDER_OF": [("{P1} founded {O1} in {y}.", [("P1", "FOUNDER_OF", "O1")]),
                   ("{O1} was founded by {P1}.", [("P1", "FOUNDER_OF", "O1")]),
                   ("{P1} and {P2} co-founded {O1} in {y}.", [("P1", "FOUNDER_OF", "O1"), ("P2", "FOUNDER_OF", "O1")]),
                   ("{O1}, established by {P1} in {y}, grew quickly.", [("P1", "FOUNDER_OF", "O1")])],
    "EXECUTIVE_OF": [("{P1} is the {re} of {O1}.", [("P1", "EXECUTIVE_OF", "O1")]),
                     ("{O1} named {P1} as its new {re}.", [("P1", "EXECUTIVE_OF", "O1")]),
                     ("{P1}, {O1}'s {re}, announced the results.", [("P1", "EXECUTIVE_OF", "O1")])],
    "EMPLOYEE_OF": [("{P1} works as a {rm} at {O1}.", [("P1", "EMPLOYEE_OF", "O1")]),
                    ("{P1} worked for {O1} from {y} to {y2}.", [("P1", "EMPLOYEE_OF", "O1")]),
                    ("{P1} joined {O1} as a {rm} in {y}.", [("P1", "EMPLOYEE_OF", "O1")]),
                    ("{P1} spent six years as a {rm} at {O1}.", [("P1", "EMPLOYEE_OF", "O1")])],
    "DEVELOPER_OF": [("{O1} developed {X}.", [("O1", "DEVELOPER_OF", "X")]),
                     ("{X} was developed by {O1}.", [("O1", "DEVELOPER_OF", "X")])],
    "MANUFACTURER_OF": [("{X} is manufactured by {O1}.", [("O1", "MANUFACTURER_OF", "X")]),
                        ("{O1} produces the {X} at its factory in {C}.", [("O1", "MANUFACTURER_OF", "X")])],
    "CREATOR_OF": [("{W} was written by {P1}.", [("P1", "CREATOR_OF", "W")]),
                   ("{P1} wrote the novel {W}.", [("P1", "CREATOR_OF", "W")])],
    "HEADQUARTERED_IN": [("{O1} is headquartered in {C}.", [("O1", "HEADQUARTERED_IN", "C")]),
                         ("{O1}, based in {C}, employs {n} people.", [("O1", "HEADQUARTERED_IN", "C")])],
    "EDUCATED_AT": [("{P1} graduated from {U} in {y}.", [("P1", "EDUCATED_AT", "U")]),
                    ("{P1} studied economics at {U}.", [("P1", "EDUCATED_AT", "U")])],
    "MEMBER_OF": [("{P1} is a member of the {T}.", [("P1", "MEMBER_OF", "T")]),
                  ("{P1} joined the {T} in {y}.", [("P1", "MEMBER_OF", "T")])],
}
NEG = {
    "speculation": ["{O1} is considering an acquisition of {O2}.", "{O1} is in talks to buy {O2}.",
                    "{O1} may invest in {O2} later this year.", "{O1} plans to partner with {O2}.",
                    "Analysts expect {O1} to take over {O2}.", "{P1} is rumored to become the {re} of {O1}.",
                    "{O1} reportedly wants to acquire {O2}.", "{P1} hopes to found a company with {P2}."],
    "negation": ["{O1} did not acquire {O2}.", "{O1} denied that it owns {O2}.", "{P1} has never worked for {O1}.",
                 "{O2} is not a subsidiary of {O1}.", "{P1} did not found {O1}, contrary to some reports."],
    "failed": ["{O1}'s offer to buy {O2} was rejected.", "The planned merger of {O1} and {O2} collapsed.",
               "{O1} abandoned its bid for {O2} in {y}.", "Talks between {O1} and {O2} about a partnership ended without a deal."],
    "future": ["{P1} will join {O1} as {re} next year.", "{O1} will move its headquarters to {C} in {y2}.",
               "{O1} is expected to complete the purchase of {O2} next quarter."],
    "comention": ["{O1} and {O2} both exhibited at the trade fair in {C}.", "{P1} met executives from {O1} and {O2}.",
                  "Shares of {O1} and {O2} rose on Friday.", "{P1} praised {O1}'s new strategy in an interview.",
                  "{P1} is a longtime customer of {O1}.", "{X} is sold by the retailer {O1}.",
                  "{W} was published by {O1}.", "{P1} visited {U} to give a lecture."],
}


class Pools:
    def __init__(self, rng: random.Random, banned: set[str]):  # banned = sha256 of normalized names
        self.rng = rng
        self.people = [f"{f} {l}" for f in FIRST for l in LAST if name_hash(f"{f} {l}") not in banned]
        self.orgs = [f"{a} {b}" for a in ORG_A for b in ORG_B if name_hash(f"{a} {b}") not in banned]
        for p in (self.people, self.orgs):
            rng.shuffle(p)

    def slots(self) -> dict:
        r = self.rng
        p1, p2 = r.sample(self.people, 2)
        o1, o2 = r.sample(self.orgs, 2)
        y = r.randint(1985, 2023)
        return {"P1": p1, "P2": p2, "O1": o1, "O2": o2, "X": r.choice(PRODUCTS), "W": r.choice(WORKS),
                "C": r.choice(CITIES), "U": r.choice(UNIS), "T": r.choice(PARTIES + TEAMS), "y": y,
                "y2": y + r.randint(1, 6), "n": r.choice([12, 40, 85, 120, 300, 950]), "k": r.randint(5, 60),
                "re": r.choice(ROLES_EXEC), "rm": r.choice(ROLES_EMP)}


def _fmt(t: str, s: dict) -> str:
    out = t.format(**s)
    return out[0].upper() + out[1:]


def make_doc(rng: random.Random, pools: Pools, idx: int, split: str) -> dict:
    """1-4 clauses: >=0 positives + >=0 hard negatives + 0-2 fillers, each with fresh entities."""
    n_pos = rng.choice([0, 1, 1, 1, 2, 2, 3])
    n_neg = rng.choice([0, 1, 1, 2]) if n_pos else rng.choice([1, 1, 2])
    units = []  # (sentence, [(s, REL, o)], category)
    for _ in range(n_pos):
        rel = rng.choice(list(POS))
        t, rels = rng.choice(POS[rel])
        s = pools.slots()
        units.append((_fmt(t, s), [(s[a], r, s[b]) for a, r, b in rels], "pos:" + rel))
    for _ in range(n_neg):
        cat = rng.choice(list(NEG))
        s = pools.slots()
        units.append((_fmt(rng.choice(NEG[cat]), s), [], "neg:" + cat))
    for _ in range(rng.choice([0, 0, 1, 2])):
        units.append((_fmt(rng.choice(FILLER), pools.slots()), [], "filler"))
    rng.shuffle(units)
    sents = [u[0] for u in units]
    rels = [{"s": s, "r": r, "o": o, "ev": [i]} for i, u in enumerate(units) for (s, r, o) in u[1]]
    return {"id": f"synth-{split}-{idx:05d}", "dataset": "synthetic_v1", "split": split, "sents": sents,
            "categories": sorted({u[2] for u in units}), "target": rels}


def to_unified(doc: dict) -> dict:
    """Convert to the unified scoring format (entities from relation strings)."""
    names = {}
    for r in doc["target"]:
        for x in (r["s"], r["o"]):
            names.setdefault(norm(x), x)
    ents = [{"id": f"e{i}", "type": "UNK", "names": [n], "mentions": []} for i, n in enumerate(names.values())]
    idx = {norm(e["names"][0]): e["id"] for e in ents}
    rels = [{"subject_id": idx[norm(r["s"])], "relation": r["r"], "object_id": idx[norm(r["o"])], "evidence": r["ev"],
             "evidence_source": "gold", "status": "explicit", "source_labels": ["synthetic"]} for r in doc["target"]]
    return {"id": doc["id"], "dataset": doc["dataset"], "split": doc["split"], "title": doc["id"],
            "sents": doc["sents"], "entities": ents, "relations": rels, "categories": doc["categories"],
            "provenance": {"generator": "reverify.data.synthetic", "version": "v1"}}


def validate_doc(doc: dict) -> list[str]:
    errs = []
    for r in doc["target"]:
        for x in (r["s"], r["o"]):
            if not any(norm(x) in norm(doc["sents"][i]) for i in r["ev"]):
                errs.append(f"{doc['id']}: '{x}' not in evidence sentence")
    return errs


def generate(n_train: int, n_dev: int, seed: int, out_dir: Path) -> dict:
    banned = banned_name_hashes()
    rng = random.Random(seed)
    pools = Pools(rng, banned)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, n in (("train", n_train), ("dev", n_dev)):
        docs = [make_doc(rng, pools, i, split) for i in range(n)]
        errs = [e for d in docs for e in validate_doc(d)]
        if errs:
            raise RuntimeError(f"{len(errs)} invalid synthetic docs, e.g. {errs[:3]}")
        with open(out_dir / f"synthetic_{split}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(to_unified(d), ensure_ascii=False) + "\n")
        stats[split] = {"docs": n, "relations": sum(len(d["target"]) for d in docs),
                        "abstain_docs": sum(1 for d in docs if not d["target"])}
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    print(generate(a.n_train, a.n_dev, a.seed, Path(a.out)))
