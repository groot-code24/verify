"""Evaluate a LoRA adapter with transformers on CPU/GPU (no GGUF export needed); same scorer as every other run.

python scripts/eval_hf_adapter.py --base HuggingFaceTB/SmolLM2-135M-Instruct --adapter checkpoints/pilot_cpu/final_adapter \
    --data synthetic_dev --limit 20 --run-id pilot_smollm135_syndev20
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reverify.format import parse_output, system_prompt, user_prompt  # noqa: E402
from reverify.inference.llamacpp_runner import ROOT, evaluate_outputs, load_eval_docs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="omit to evaluate the untuned base with the SAME short prompt")
    ap.add_argument("--data", default="synthetic_dev")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.adapter or a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    docs = load_eval_docs(a.data, a.limit)
    outputs = {}
    for d in docs:
        msgs = [{"role": "system", "content": system_prompt(with_definitions=False)},
                {"role": "user", "content": user_prompt(d["sents"])}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                      enable_thinking=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False)
        dt = time.perf_counter() - t0
        new = out[0, ids["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        rels, ok = parse_output(text)
        outputs[d["id"]] = {"doc_id": d["id"], "raw": text, "relations": rels, "parse_ok": ok, "latency_s": round(dt, 3),
                            "prompt_tokens": int(ids["input_ids"].shape[1]), "completion_tokens": int(len(new)),
                            "finish_reason": "length" if len(new) >= a.max_new_tokens else "stop"}
    raw = ROOT / "results" / "raw_outputs" / f"{a.run_id}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in outputs.values()) + "\n", encoding="utf-8")
    res = evaluate_outputs(docs, outputs)
    res["meta"] = {"run_id": a.run_id, "base": a.base, "adapter": a.adapter, "data": a.data, "n_docs": len(docs),
                   "runtime": "transformers fp32", "date": time.strftime("%Y-%m-%d %H:%M")}
    (ROOT / "results" / "runs" / f"{a.run_id}.json").write_text(json.dumps(res, indent=2))
    for k in ("A_raw", "B_verifier", "B+_verifier_hedge"):
        r = res[k]
        print(f"{k:18s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['micro_f1']:.3f} CI={r['micro_f1_ci95']} "
              f"unsup={r['unsupported_rate']:.3f} abst={r['abstention_accuracy']} fp={r['fp_breakdown']}")
    print(res["efficiency"])


if __name__ == "__main__":
    main()
