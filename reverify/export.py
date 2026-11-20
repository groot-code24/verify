"""Merge a LoRA adapter into its base model, convert to GGUF and quantize for CPU deployment.

python -m reverify.export --base Qwen/Qwen2.5-0.5B-Instruct --adapter checkpoints/x/final_adapter \
    --out models/reverify-qwen25-05b --quants q8_0 q4_0 --llama-cpp-dir llama.cpp

Steps
1. merge_and_unload() in fp16, save HF model + tokenizer to <out>/merged
2. llama.cpp/convert_hf_to_gguf.py -> <out>/reverify-f16.gguf   (needs a llama.cpp checkout + `pip install gguf`)
3. llama_cpp.llama_model_quantize   -> <out>/reverify-<q>.gguf   (no compiled llama-quantize binary needed)
4. Writes <out>/PACKAGE.json with file sizes and sha256 (the deployment size = gguf; tokenizer is embedded).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

QTYPES = {"q8_0": "LLAMA_FTYPE_MOSTLY_Q8_0", "q4_0": "LLAMA_FTYPE_MOSTLY_Q4_0", "q4_k_m": "LLAMA_FTYPE_MOSTLY_Q4_K_M",
          "q5_k_m": "LLAMA_FTYPE_MOSTLY_Q5_K_M", "q3_k_m": "LLAMA_FTYPE_MOSTLY_Q3_K_M", "q2_k": "LLAMA_FTYPE_MOSTLY_Q2_K"}


def merge(base: str, adapter: str, out: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    dst = out / "merged"
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(dst)
    return dst


def to_gguf(merged: Path, out: Path, llama_cpp_dir: Path) -> Path:
    f16 = out / "reverify-f16.gguf"
    subprocess.run([sys.executable, str(llama_cpp_dir / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16),
                    "--outtype", "f16"], check=True)
    return f16


def quantize(src: Path, q: str, dst: Path | None = None, allow_requantize: bool = False) -> Path:
    import llama_cpp
    dst = dst or src.with_name(f"reverify-{q}.gguf")
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, QTYPES[q])
    params.nthread = 4
    params.allow_requantize = allow_requantize
    rc = llama_cpp.llama_model_quantize(str(src).encode(), str(dst).encode(), ctypes.byref(params))
    if rc != 0:
        raise RuntimeError(f"quantize {q} failed rc={rc}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize-only", default=None, help="existing GGUF to (re)quantize; skips merge/convert")
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quants", nargs="+", default=["q8_0", "q4_0"])
    ap.add_argument("--llama-cpp-dir", default="llama.cpp")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.quantize_only:
        src = Path(a.quantize_only)
        for q in a.quants:
            d = quantize(src, q, out / f"{src.stem}-{q}.gguf", allow_requantize=True)
            print(d, round(d.stat().st_size / 1e6, 1), "MB")
        return
    if not (a.base and a.adapter):
        ap.error("--base and --adapter are required unless --quantize-only is given")
    merged = merge(a.base, a.adapter, out)
    f16 = to_gguf(merged, out, Path(a.llama_cpp_dir))
    files = [f16] + [quantize(f16, q) for q in a.quants]
    pkg = {p.name: {"bytes": p.stat().st_size, "mb": round(p.stat().st_size / 1e6, 1),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    (out / "PACKAGE.json").write_text(json.dumps(pkg, indent=1))
    print(json.dumps(pkg, indent=1))


if __name__ == "__main__":
    main()
