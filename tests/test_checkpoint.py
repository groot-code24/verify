import json

from reverify.hardware import adapt
from reverify.train.sft import find_latest_checkpoint, verify_checkpoint, write_manifest


def _ckpt(root, step, payload=b"weights"):
    c = root / f"checkpoint-{step}"
    c.mkdir(parents=True)
    (c / "adapter_model.safetensors").write_bytes(payload)
    (c / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    write_manifest(c, {"global_step": step})
    return c


def test_manifest_detects_corruption(tmp_path):
    c = _ckpt(tmp_path, 10)
    assert verify_checkpoint(c)
    (c / "adapter_model.safetensors").write_bytes(b"truncated")
    assert not verify_checkpoint(c)


def test_latest_skips_corrupt_and_uses_sync_dir(tmp_path):
    local, sync = tmp_path / "local", tmp_path / "sync"
    _ckpt(local, 10)
    bad = _ckpt(local, 30)
    (bad / "adapter_model.safetensors").unlink()           # interrupted mid-save
    _ckpt(sync, 20)
    assert find_latest_checkpoint(local, sync).name == "checkpoint-20"


def test_no_checkpoint_returns_none(tmp_path):
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_adapt_cpu_and_gpu_profiles():
    cpu = adapt({"cuda": False}, 0.135, 1024)
    assert cpu["precision"] == "fp32" and cpu["seq_len"] == 512 and not cpu["quant_4bit"]
    t4 = adapt({"cuda": True, "gpu_mem_gb": 14.7, "bf16": False}, 0.5, 1024)
    assert t4["precision"] == "fp16" and not t4["quant_4bit"] and t4["micro_batch"] >= 1
    big = adapt({"cuda": True, "gpu_mem_gb": 14.7, "bf16": True}, 7.0, 1024)
    assert big["quant_4bit"] and big["precision"] == "bf16"
