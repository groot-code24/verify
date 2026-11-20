"""Build versioned processed datasets with validation and leakage reports.

Usage: python -m reverify.data.build
Outputs data/processed/{redocred_train,redocred_dev,redocred_test,synthetic_*}.jsonl and
data/processed/DATASET_REPORT.json (counts, validation, leakage, sha256 of every output).
"""
from __future__ import annotations

import hashlib
import re
import json
from collections import Counter
from pathlib import Path

from reverify.data.challenge import CHALLENGE, available, file_sha256, load_challenge
from reverify.data.redocred import load_default
from reverify.data.synthetic import generate
from reverify.ontology import load_ontology
from reverify.verifier import norm

OUT = Path("data/processed")
DATA_VERSION = "data-v1.1"

# Statements of relation types that DocRED's schema cannot label (ACQUIRED / INVESTED_IN / PARTNERED_WITH).
# In such Re-DocRED docs the acquirer is labelled OWNS / PARENT_ORG_OF (131 of 451 explicit train triples)
# and the acquisition itself is a silent negative, which contradicts the synthetic ACQUIRED supervision
# (FAILURES.md F-011, DECISIONS.md D-013). These docs are removed from the *training* file only.
UNLABELABLE_CUES = re.compile(
    r"\b(acquir(ed|es|ing)|acquisition|bought|purchas(ed|e of)|took over|takeover|invest(ed|ment|or)s?|"
    r"joint venture|partnership|partnered|merg(ed|er))\b", re.I)


def shingles(text: str, k: int = 8) -> set:
    w = norm(text).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def validate(docs: list[dict]) -> dict:
    issues = Counter()
    for d in docs:
        ids = {e["id"] for e in d["entities"]}
        for r in d["relations"]:
            if r["subject_id"] not in ids or r["object_id"] not in ids:
                issues["dangling_entity"] += 1
            if any(i >= len(d["sents"]) or i < 0 for i in r["evidence"]):
                issues["evidence_out_of_range"] += 1
            if r["status"] == "explicit" and not r["evidence"]:
                issues["explicit_without_evidence"] += 1
            if r["subject_id"] == r["object_id"]:
                issues["self_relation"] += 1
        keys = [(r["subject_id"], r["relation"], r["object_id"]) for r in d["relations"]]
        issues["duplicate_triples"] += len(keys) - len(set(keys))
        # conflicting labels: same ordered pair with mutually exclusive relations
        pairs = Counter((r["subject_id"], r["object_id"]) for r in d["relations"])
        issues["multi_label_pairs"] += sum(1 for c in pairs.values() if c > 1)
    return dict(issues)


def leakage(a: list[dict], b: list[dict], name: str) -> dict:
    ta = {norm(d.get("title", "")) for d in a if d.get("title")}
    tb = {norm(d.get("title", "")) for d in b if d.get("title")}
    xa = {norm(" ".join(d["sents"])) for d in a}
    sh_a: dict[str, int] = {}
    for i, d in enumerate(a):
        for s in shingles(" ".join(d["sents"])):
            sh_a.setdefault(s, i)
    near = 0
    for d in b:
        sb = shingles(" ".join(d["sents"]))
        if sb and len([s for s in sb if s in sh_a]) / len(sb) >= 0.5:
            near += 1
    sents_a = {norm(s) for d in a for s in d["sents"] if len(s.split()) >= 6}
    sent_overlap = sum(1 for d in b for s in d["sents"] if len(s.split()) >= 6 and norm(s) in sents_a)
    return {"pair": name, "title_overlap": len(ta & tb) if ta and tb else 0,
            "exact_doc_overlap": sum(1 for d in b if norm(" ".join(d["sents"])) in xa),
            "near_dup_docs(>=50% 8-gram)": near, "exact_sentence_overlap(>=6 words)": sent_overlap}


def entity_overlap(train: list[dict], test: list[dict]) -> float:
    tr = {norm(n) for d in train for e in d["entities"] for n in e["names"]}
    te = [norm(n) for d in test for e in d["entities"] for n in e["names"]]
    return round(sum(1 for n in te if n in tr) / max(1, len(te)), 4)


def write(docs: list[dict], name: str) -> str:
    p = OUT / f"{name}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(name: str) -> list[dict]:
    return [json.loads(l) for l in open(OUT / f"{name}.jsonl", encoding="utf-8")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    onto = load_ontology()
    report = {"data_version": DATA_VERSION, "ontology_version": onto.version, "files": {}, "validation": {},
              "relation_counts": {}, "leakage": [], "challenge_sha256": file_sha256(CHALLENGE) if available() else "not present"}
    splits = {}
    for s in ("train", "dev", "test"):
        docs, agg = load_default(s)
        if s == "dev":  # drop dev docs that duplicate / near-duplicate a train doc (found by leakage check)
            train_sh = set().union(*(shingles(" ".join(d["sents"])) for d in splits["train"]))
            keep = []
            for d in docs:
                sh = shingles(" ".join(d["sents"]))
                if sh and len(sh & train_sh) / len(sh) >= 0.5:
                    report.setdefault("removed_for_leakage", []).append(d["id"])
                else:
                    keep.append(d)
            docs = keep
        splits[s] = docs
        report["files"][f"redocred_{s}"] = write(docs, f"redocred_{s}")
        if s == "train":
            clean = [d for d in docs if not any(UNLABELABLE_CUES.search(x) for x in d["sents"])]
            report["files"]["redocred_train_clean"] = write(clean, "redocred_train_clean")
            report["train_clean_filter"] = {"kept": len(clean), "removed": len(docs) - len(clean)}
        report["validation"][f"redocred_{s}"] = {**validate(docs), **agg}
        report["relation_counts"][f"redocred_{s}"] = dict(Counter(r["relation"] for d in docs for r in d["relations"] if r["status"] == "explicit"))
    report["synthetic"] = generate(3000, 300, 13, OUT)
    syn_tr, syn_dev = read("synthetic_train"), read("synthetic_dev")
    for n in ("synthetic_train", "synthetic_dev"):
        report["files"][n] = hashlib.sha256((OUT / f"{n}.jsonl").read_bytes()).hexdigest()
        report["validation"][n] = validate(read(n))
        report["relation_counts"][n] = dict(Counter(r["relation"] for d in read(n) for r in d["relations"]))
    ch = load_challenge() if available() else []
    report["validation"]["challenge_v1"] = validate(ch)
    report["leakage"] = [leakage(splits["train"], splits["dev"], "redocred train->dev"),
                         leakage(splits["train"], splits["test"], "redocred train->test"),
                         leakage(splits["dev"], splits["test"], "redocred dev->test"),
                         leakage(splits["train"] + syn_tr, ch, "all train->challenge_v1"),
                         leakage(syn_tr, syn_dev, "synthetic train->dev")]
    report["entity_name_overlap_train_to_test"] = {
        "redocred_test": entity_overlap(splits["train"], splits["test"]),
        "challenge_v1": entity_overlap(splits["train"] + syn_tr, ch)}
    (OUT / "DATASET_REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("validation", "leakage", "entity_name_overlap_train_to_test", "synthetic")}, indent=1))


if __name__ == "__main__":
    main()
