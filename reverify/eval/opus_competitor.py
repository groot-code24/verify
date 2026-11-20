"""Run Claude Opus 5.5 on the frozen benchmark with the SAME inputs, ontology, schema and scorer.

Requires credentials (ANTHROPIC_API_KEY or an `ant auth login` profile) and a current SDK:
    pip install -U "anthropic>=1.0"
Status 2026-11-20: NOT RUN - no credentials available on the dev machine.

    python -m reverify.eval.opus_competitor --data challenge_v1 --run-id opus55_challenge_v1
    python -m reverify.eval.opus_competitor --data redocred_test --batch --run-id opus55_redocred_test

Fairness protocol (see docs/benchmark_protocol.md):
- Opus receives the FULL prompt: definitions for every relation + one-shot example + output format.
  RE-Verify receives only the short label-list prompt, so the prompt gap favours Opus (conservative).
- Same parser, same deterministic verifier variants, same scorer as RE-Verify.
- No temperature control exists on this model (sending it returns a 400); thinking is always on.
  Effort is fixed explicitly (default "high") and recorded with every output.
- No refusal fallbacks: a fallback would answer with a different model. Refusals are stored and scored
  as empty predictions, and their count is reported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from reverify.format import parse_output, system_prompt, user_prompt
from reverify.inference.llamacpp_runner import ROOT, evaluate_outputs, load_eval_docs

MODEL = "claude-opus-5-5"


def build_params(doc: dict, effort: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": 16000,
        # The prompt is ~700 tokens, which may be below the minimum cacheable prefix; caching is harmless if so.
        "system": [{"type": "text", "text": system_prompt(with_definitions=True, one_shot=True),
                    "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": user_prompt(doc["sents"]) + "\n\nReturn only the JSON object."}],
    }


def record(msg, dt: float, effort: str) -> dict:
    text = "".join(b.text for b in msg.content if b.type == "text")
    rels, ok = parse_output(text)
    u = msg.usage
    return {"raw": text, "relations": rels, "parse_ok": ok, "latency_s": round(dt, 3),
            "prompt_tokens": u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
            "completion_tokens": u.output_tokens, "finish_reason": msg.stop_reason,
            "refused": msg.stop_reason == "refusal", "model": msg.model, "effort": effort,
            "request_id": getattr(msg, "_request_id", None)}


def run_sync(client, docs, done, raw_path, effort):
    import anthropic
    with open(raw_path, "a", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            if d["id"] in done:
                continue
            t0 = time.perf_counter()
            try:
                msg = client.messages.create(**build_params(d, effort))
            except anthropic.BadRequestError as e:
                raise SystemExit(f"bad request on {d['id']}: {e.message}")
            o = record(msg, time.perf_counter() - t0, effort)
            o["doc_id"] = d["id"]
            done[d["id"]] = o
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i+1}/{len(docs)} {o['latency_s']}s", flush=True)


def run_batch(client, docs, done, raw_path, effort, state_path: Path):
    todo = [d for d in docs if d["id"] not in done]
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not state.get("batch_id"):
        b = client.messages.batches.create(
            requests=[{"custom_id": d["id"], "params": build_params(d, effort)} for d in todo])
        state = {"batch_id": b.id}
        state_path.write_text(json.dumps(state))  # resumable: re-running polls the same batch
        print("submitted batch", b.id, flush=True)
    while client.messages.batches.retrieve(state["batch_id"]).processing_status != "ended":
        time.sleep(60)
    with open(raw_path, "a", encoding="utf-8") as f:
        for r in client.messages.batches.results(state["batch_id"]):  # any order: key by custom_id
            if r.result.type != "succeeded":
                o = {"raw": "", "relations": [], "parse_ok": False, "latency_s": 0.0, "prompt_tokens": 0,
                     "completion_tokens": 0, "finish_reason": r.result.type, "refused": False, "effort": effort}
            else:
                o = record(r.result.message, 0.0, effort)  # batch latency is not per-request latency
            o["doc_id"] = r.custom_id
            done[r.custom_id] = o
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main():
    import anthropic
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--batch", action="store_true", help="Message Batches API (50%% cost, async)")
    a = ap.parse_args()

    docs = load_eval_docs(a.data, a.limit)
    raw_path = ROOT / "results" / "raw_outputs" / f"{a.run_id}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if raw_path.exists():
        for l in open(raw_path, encoding="utf-8"):
            r = json.loads(l)
            done[r["doc_id"]] = r
    client = anthropic.Anthropic()
    if a.batch:
        run_batch(client, docs, done, raw_path, a.effort, raw_path.with_suffix(".batch.json"))
    else:
        run_sync(client, docs, done, raw_path, a.effort)
    res = evaluate_outputs(docs, done)
    res["meta"] = {"run_id": a.run_id, "model": MODEL, "effort": a.effort, "data": a.data, "n_docs": len(docs),
                   "refusals": sum(o.get("refused", False) for o in done.values()), "batch": a.batch,
                   "date": time.strftime("%Y-%m-%d %H:%M")}
    out = ROOT / "results" / "runs" / f"{a.run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k]["micro_f1"] for k in ("A_raw", "B_verifier", "B+_verifier_hedge")}), res["meta"])


if __name__ == "__main__":
    main()
