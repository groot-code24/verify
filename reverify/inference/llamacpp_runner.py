"""CPU inference with llama.cpp (GGUF) + grammar-constrained compact JSON output.

python -m reverify.inference.llamacpp_runner --model models/x.gguf --data redocred_dev --limit 50 \
    --run-id base_qwen25_05b --threads 4 [--short-prompt] [--no-think]

Writes results/raw_outputs/<run_id>.jsonl (raw text, parsed relations, timings) and
results/runs/<run_id>.json (scores for Architecture A = raw, B = +deterministic verifier,
B+ = +verifier+hedge filter).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

from reverify.eval.metrics import bootstrap_ci, score
from reverify.format import json_grammar, parse_output, system_prompt, user_prompt
from reverify.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def load_eval_docs(name: str, limit: int | None, seed: int = 0) -> list[dict]:
    if name == "challenge_v1":
        from reverify.data.challenge import load_challenge
        docs = load_challenge()
    else:
        docs = [json.loads(l) for l in open(ROOT / "data" / "processed" / f"{name}.jsonl", encoding="utf-8")]
    if limit and limit < len(docs):
        docs = random.Random(seed).sample(docs, limit)  # fixed seeded subset
        docs.sort(key=lambda d: d["id"])
    return docs


def chatml(system: str, user: str, no_think: bool) -> str:
    p = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return p + ("<think>\n\n</think>\n\n" if no_think else "")


class LlamaExtractor:
    def __init__(self, model_path: str, threads: int = 4, n_ctx: int = 4096, short_prompt: bool = False,
                 no_think: bool = False, grammar: bool = True, one_shot: bool = False):
        from llama_cpp import Llama, LlamaGrammar
        t0 = time.perf_counter()
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads, n_batch=512, verbose=False, seed=0)
        self.load_s = time.perf_counter() - t0
        self.grammar = LlamaGrammar.from_string(json_grammar(), verbose=False) if grammar else None
        self.system = system_prompt(with_definitions=not short_prompt, one_shot=one_shot)
        self.no_think = no_think

    def extract(self, sents: list[str], max_tokens: int = 512) -> dict:
        prompt = chatml(self.system, user_prompt(sents), self.no_think)
        t0 = time.perf_counter()
        out = self.llm.create_completion(prompt, max_tokens=max_tokens, temperature=0.0, top_k=1,
                                         grammar=self.grammar, stop=["<|im_end|>"])
        dt = time.perf_counter() - t0
        text = out["choices"][0]["text"]
        rels, ok = parse_output(text)
        return {"raw": text, "relations": rels, "parse_ok": ok, "latency_s": round(dt, 3),
                "prompt_tokens": out["usage"]["prompt_tokens"], "completion_tokens": out["usage"]["completion_tokens"],
                "finish_reason": out["choices"][0]["finish_reason"]}


def evaluate_outputs(docs: list[dict], outputs: dict) -> dict:
    # Only the documents of this evaluation: a resumed run file can hold outputs for docs outside the
    # current subset, which previously leaked into the efficiency stats (FAILURES.md F-013).
    outputs = {d["id"]: outputs[d["id"]] for d in docs}
    res = {}
    variants = {
        "A_raw": lambda o, d: o["relations"],
        "B_verifier": lambda o, d: verify(o["relations"], d["sents"]).kept,
        "B+_verifier_hedge": lambda o, d: verify(o["relations"], d["sents"], hedge_filter=True).kept,
    }
    for name, fn in variants.items():
        preds = {d["id"]: fn(outputs[d["id"]], d) for d in docs}
        agg, ds = score(docs, preds)
        agg["micro_f1_ci95"] = bootstrap_ci(ds, n=500)
        agg["micro_f1_strict_match"] = score(docs, preds, match="strict")[0]["micro_f1"]
        res[name] = agg
    lat = [o["latency_s"] for o in outputs.values()]
    res["efficiency"] = {
        "mean_latency_s": round(sum(lat) / len(lat), 3), "p50_latency_s": round(sorted(lat)[len(lat) // 2], 3),
        "max_latency_s": round(max(lat), 3),
        "mean_completion_tokens": round(sum(o["completion_tokens"] for o in outputs.values()) / len(lat), 1),
        "mean_prompt_tokens": round(sum(o["prompt_tokens"] for o in outputs.values()) / len(lat), 1),
        "parse_ok_rate": round(sum(o["parse_ok"] for o in outputs.values()) / len(lat), 4),
        "truncated_rate": round(sum(o["finish_reason"] == "length" for o in outputs.values()) / len(lat), 4)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="redocred_dev")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--short-prompt", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--one-shot", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    docs = load_eval_docs(a.data, a.limit)
    raw_dir = ROOT / "results" / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{a.run_id}.jsonl"
    done = {}
    if raw_path.exists():  # resumable: skip docs already processed
        for l in open(raw_path, encoding="utf-8"):
            r = json.loads(l)
            done[r["doc_id"]] = r
    ex = LlamaExtractor(a.model, a.threads, short_prompt=a.short_prompt, no_think=a.no_think,
                        grammar=not a.no_grammar, one_shot=a.one_shot)
    print(f"model loaded in {ex.load_s:.2f}s; {len(done)} cached outputs", flush=True)
    with open(raw_path, "a", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            if d["id"] in done:
                continue
            o = ex.extract(d["sents"], a.max_tokens)
            o["doc_id"] = d["id"]
            done[d["id"]] = o
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i+1}/{len(docs)} {o['latency_s']}s ctok={o['completion_tokens']}", flush=True)
    res = evaluate_outputs(docs, done)
    res["meta"] = {"run_id": a.run_id, "model": a.model, "model_bytes": os.path.getsize(a.model), "data": a.data,
                   "n_docs": len(docs), "limit": a.limit, "threads": a.threads, "short_prompt": a.short_prompt,
                   "grammar": not a.no_grammar, "one_shot": a.one_shot, "load_s": round(ex.load_s, 2), "cpu": platform.processor(),
                   "date": time.strftime("%Y-%m-%d %H:%M")}
    out = ROOT / "results" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.run_id}.json").write_text(json.dumps(res, indent=2))
    for k in ("A_raw", "B_verifier", "B+_verifier_hedge"):
        r = res[k]
        print(f"{k:20s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['micro_f1']:.3f} macroF1={r['macro_f1']:.3f} "
              f"evF1={r['evidence_f1']} unsup={r['unsupported_rate']:.3f} rev={r['reversal_error_rate']:.3f} "
              f"abst={r['abstention_accuracy']}")
    print(res["efficiency"])


if __name__ == "__main__":
    main()
