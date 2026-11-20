"""Re-score saved raw outputs with the current parser + scorer (no model calls).

python scripts/rescore.py [run_id ...]      # default: every results/runs/*.json that has raw outputs

Use after any change to format.parse_output, verifier or eval/metrics so every reported number comes from
the same scoring code. The previous scores are kept under "previous_scores" for the audit trail.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reverify.format import parse_output  # noqa: E402
from reverify.inference.llamacpp_runner import ROOT, evaluate_outputs, load_eval_docs  # noqa: E402


def rescore(run_path: Path) -> dict | None:
    res = json.loads(run_path.read_text())
    raw = ROOT / "results" / "raw_outputs" / f"{run_path.stem}.jsonl"
    if not raw.exists():
        return None
    meta = res["meta"]
    docs = load_eval_docs(meta["data"], meta["n_docs"])
    outputs = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        o["relations"], o["parse_ok"] = parse_output(o["raw"])
        outputs[o["doc_id"]] = o
    new = evaluate_outputs(docs, outputs)
    new["meta"] = {**meta, "rescored_with": "scorer v1.1 (lenient unique match, token-bounded grounding)"}
    new["previous_scores"] = {k: res[k]["micro_f1"] for k in ("A_raw", "B_verifier", "B+_verifier_hedge")}
    run_path.write_text(json.dumps(new, indent=2))
    return new


if __name__ == "__main__":
    runs = [ROOT / "results" / "runs" / f"{r}.json" for r in sys.argv[1:]] or sorted((ROOT / "results" / "runs").glob("*.json"))
    for p in runs:
        r = rescore(p)
        if r is None:
            print(f"skip {p.stem} (no raw outputs)")
            continue
        cells = "  ".join(f"{k}={r[k]['micro_f1']:.3f} (was {r['previous_scores'][k]:.3f}, strict {r[k]['micro_f1_strict_match']:.3f})"
                          for k in ("A_raw", "B_verifier", "B+_verifier_hedge"))
        print(f"{p.stem}: {cells}")
