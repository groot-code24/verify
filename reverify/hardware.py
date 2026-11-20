"""Hardware detection and adaptive training configuration.

Fixes recorded in FAILURES.md F-010:
- bf16 is decided by compute capability >= 8.0 (Ampere+). torch.cuda.is_bf16_supported() defaults to
  including_emulation=True and reports True on a T4, which has no native bf16.
- Without native bf16 the base model is loaded in fp32 and trained with fp16 autocast (AMP), which
  avoids fp16 overflow in Qwen activations and the "unscale FP16 gradients" failure mode.
- Micro-batch size accounts for the logits tensor: with a 152k vocabulary, 16 x 1024 tokens of logits
  alone need ~15 GB, which is more than a whole T4.
"""
from __future__ import annotations

import os
import platform
import shutil


def detect() -> dict:
    import torch
    info = {"platform": platform.platform(), "cpu_count": os.cpu_count(), "cuda": torch.cuda.is_available(),
            "torch": torch.__version__}
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 1)
    except ImportError:
        pass
    info["disk_free_gb"] = round(shutil.disk_usage(".").free / 2**30, 1)
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 1), capability=f"{p.major}.{p.minor}",
                    bf16=p.major >= 8, cuda_version=torch.version.cuda, n_gpus=torch.cuda.device_count())
    return info


def _pow2_floor(x: float) -> int:
    n = 1
    while n * 2 <= x:
        n *= 2
    return n


def adapt(hw: dict, model_params_b: float, requested_seq_len: int = 1024, vocab_size: int = 152_000,
          effective_batch: int = 16) -> dict:
    """Conservative defaults. OOM at runtime is still handled by halving micro_batch (train/sft.py)."""
    if not hw["cuda"]:
        return {"precision": "fp32", "load_dtype": "fp32", "micro_batch": 1, "grad_accum": 8,
                "seq_len": min(requested_seq_len, 512), "grad_checkpointing": False, "workers": 0, "quant_4bit": False}
    mem = hw["gpu_mem_gb"]
    bf16 = bool(hw.get("bf16"))
    precision = "bf16" if bf16 else "fp16"
    load_dtype = "bf16" if bf16 else ("fp32" if model_params_b <= 2.0 else "fp16")
    bytes_per_param = 4 if load_dtype == "fp32" else 2
    weights_gb = model_params_b * bytes_per_param
    quant = model_params_b * 2 > 0.45 * mem  # 16-bit weights would crowd out activations -> QLoRA
    if quant:
        weights_gb, load_dtype = model_params_b * 0.6, ("bf16" if bf16 else "fp16")
    free_gb = max(1.0, mem - weights_gb - 2.0)  # CUDA context + LoRA optimizer state + fragmentation
    logits_bytes_per_tok = vocab_size * 10       # 16-bit logits + fp32 upcast + fp32 grad
    act_bytes_per_tok = 1.3e6 * model_params_b    # rough, without gradient checkpointing
    tokens = 0.6 * free_gb * 1e9 / (logits_bytes_per_tok + act_bytes_per_tok)
    micro = max(1, min(16, _pow2_floor(tokens / requested_seq_len)))
    return {"precision": precision, "load_dtype": load_dtype, "micro_batch": micro,
            "grad_accum": max(1, effective_batch // micro), "seq_len": requested_seq_len,
            "grad_checkpointing": micro < 2, "workers": 2, "quant_4bit": quant}


if __name__ == "__main__":
    import json
    hw = detect()
    print(json.dumps(hw, indent=1))
    for b in (0.5, 1.5):
        print(b, "B ->", adapt(hw, b))
