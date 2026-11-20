"""Generate notebooks/train_gpu.ipynb (the notebook is build output; edit this file instead).

python scripts/make_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "data" / "processed" / "DATASET_REPORT.json").read_text())["files"]

CELLS = [
("markdown", """# RE-Verify — train, export, evaluate, publish (Kaggle or Colab)

One run = one experiment config: **install → build data → LoRA SFT → merge → GGUF → quantize → dev evaluation → (optional) publish**.

**Before running** (details in `docs/KAGGLE_COLAB_GUIDE.md`):
1. Hugging Face token with **write** access, stored as a secret named `HF_TOKEN`
   (Kaggle: *Add-ons → Secrets*; Colab: the 🔑 *Secrets* panel, with notebook access on).
2. Kaggle: *Settings → Accelerator → GPU T4 x2* (one GPU is used), *Internet → On* (needs a phone-verified account).
   Colab: *Runtime → Change runtime type → T4 GPU*.
3. Fill in the parameters cell.

**Crash-safe:** every checkpoint is uploaded to your private repo `CKPT_REPO`. After a disconnect, just *Run All* again:
training resumes from the newest hash-verified checkpoint.

Do **not** evaluate on `redocred_test` or the challenge set here. Final, one-time evaluation happens after model selection."""),
("code", """# 1. PARAMETERS — edit these
REPO_URL   = "https://github.com/YOUR_GITHUB_USER/re-verify.git"
GIT_REF    = "main"
CONFIG     = "configs/train_qwen25_05b_lora.yaml"   # or train_qwen3_06b_lora.yaml / train_qwen25_15b_lora.yaml
HF_USER    = "YOUR_HF_USERNAME"
CKPT_REPO  = f"{HF_USER}/re-verify-checkpoints"      # PRIVATE repo, created automatically
PUBLISH    = False                                   # True = upload model (+ Space) at the end
MODEL_REPO = f"{HF_USER}/re-verify-qwen25-0.5b"
SPACE_REPO = f"{HF_USER}/re-verify-demo"
PUBLISH_PRIVATE = True                               # first upload private; make public on the Hub when satisfied"""),
("code", """# 2. environment + secrets (no token is ever printed)
import os, shutil, subprocess, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # one GPU: DataParallel on T4 x2 would change the effective batch
ON_KAGGLE = os.path.exists("/kaggle")
if ON_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
else:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
assert os.environ.get("HF_TOKEN"), "HF_TOKEN secret missing"
BASE = "/kaggle/working" if ON_KAGGLE else "/content"
print("platform:", "Kaggle" if ON_KAGGLE else "Colab")"""),
("code", """# 3. source
WORK = f"{BASE}/re-verify"
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", GIT_REF, REPO_URL, WORK], check=True)
os.chdir(WORK)
!git log -1 --format="commit %h %s\""""),
("code", """# 4. dependencies: training extras + prebuilt llama.cpp CPU wheel (0.3.20 has no Linux wheel; 0.3.19 does)
!pip -q install -e ".[train,dev]"
!pip -q install "llama-cpp-python==0.3.19" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
!python -m reverify.hardware
import torch
assert torch.cuda.is_available(), "No GPU. Kaggle: Settings -> Accelerator -> GPU. Colab: Runtime -> T4 GPU."
print(torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))"""),
("code", f"""# 5. data (deterministic) + integrity check against the published hashes + unit tests
!python -m reverify.data.download
!python -m reverify.data.build > /dev/null
import json
EXPECTED = {json.dumps(EXPECTED, indent=1)}
got = json.load(open("data/processed/DATASET_REPORT.json"))["files"]
bad = {{k for k in EXPECTED if got.get(k) != EXPECTED[k]}}
print("data hashes:", "ALL MATCH" if not bad else f"MISMATCH in {{bad}} - record this in FAILURES.md before training")
!python -m pytest -q"""),
("code", """# 6. train (auto-resume; checkpoints -> private hub repo under <experiment_id>/).
#    If this experiment already finished (adapters on the hub), download them instead of training again,
#    so a fresh session can go straight to export / evaluation / publishing with Run All.
import yaml
from huggingface_hub import HfApi, snapshot_download
cfg = yaml.safe_load(open(CONFIG))
cfg["hub_repo"], cfg["sync_dir"] = CKPT_REPO, None
yaml.safe_dump(cfg, open("configs/_active.yaml", "w"))
EXP = cfg["experiment_id"]
try:
    done = f"{EXP}/final_adapter/adapter_config.json" in HfApi().list_repo_files(CKPT_REPO)
except Exception:
    done = False
if done:
    snapshot_download(CKPT_REPO, allow_patterns=[f"{EXP}/final_adapter/*", f"{EXP}/best_adapter/*"], local_dir="_hub")
    for name in ("final_adapter", "best_adapter"):
        if os.path.exists(f"_hub/{EXP}/{name}"):
            shutil.copytree(f"_hub/{EXP}/{name}", f"{cfg['output_dir']}/{name}", dirs_exist_ok=True)
    print(f"{EXP} already trained: adapters restored from {CKPT_REPO}. To retrain, use a new experiment_id.")
else:
    !python -m reverify.train.sft --config configs/_active.yaml --resume auto"""),
("code", """# 7. export: merge -> GGUF f16 -> q4_0 / q4_k_m / q8_0
out_dir = cfg["output_dir"]
ADAPTER = f"{out_dir}/best_adapter" if os.path.exists(f"{out_dir}/best_adapter") else f"{out_dir}/final_adapter"
if not os.path.exists(f"{BASE}/llama.cpp"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", f"{BASE}/llama.cpp"], check=True)
!git -C {BASE}/llama.cpp log -1 --format="llama.cpp %h %cd"
!pip -q install {BASE}/llama.cpp/gguf-py
OUT = f"exports/{EXP}"
!python -m reverify.export --base {cfg['base_model']} --adapter {ADAPTER} --out {OUT} --quants q4_0 q4_k_m q8_0 --llama-cpp-dir {BASE}/llama.cpp
print("adapter used:", ADAPTER)"""),
("code", """# 8. dev evaluation of the quantized model on CPU (same runner + scorer as every other result)
THINK = "--no-think" if "Qwen3" in cfg["base_model"] else ""
NT = os.cpu_count()
RUNS = [f"{EXP}_q4_0_redocred_dev150", f"{EXP}_q4_0_synthetic_dev100"]
!python -m reverify.inference.llamacpp_runner --model {OUT}/reverify-q4_0.gguf --data redocred_dev --limit 150 --run-id {RUNS[0]} --short-prompt --no-grammar --threads {NT} {THINK}
!python -m reverify.inference.llamacpp_runner --model {OUT}/reverify-q4_0.gguf --data synthetic_dev --limit 100 --run-id {RUNS[1]} --short-prompt --no-grammar --threads {NT} {THINK}
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="results", path_in_repo=f"results/{EXP}", repo_id=CKPT_REPO, commit_message=f"dev results {EXP}")
HfApi().upload_folder(folder_path=OUT, path_in_repo=f"exports/{EXP}", repo_id=CKPT_REPO, allow_patterns=["*.gguf", "PACKAGE.json"],
                      ignore_patterns=["*f16.gguf"], commit_message=f"quantized {EXP}")
print("results + quantized files saved to", CKPT_REPO)"""),
("code", """# 9. publish (only if PUBLISH=True): model repo with generated model card (+ demo Space)
if PUBLISH:
    priv = "--private" if PUBLISH_PRIVATE else ""
    github = REPO_URL.removesuffix(".git")
    !python scripts/publish_hf.py --model-repo {MODEL_REPO} --gguf {OUT}/reverify-q4_0.gguf {OUT}/reverify-q4_k_m.gguf {OUT}/reverify-q8_0.gguf --adapter {ADAPTER} --base-model {cfg['base_model']} --runs {RUNS[0]} {RUNS[1]} --github-url {github} --space-repo {SPACE_REPO} --hardware "{'Kaggle' if ON_KAGGLE else 'Colab'} CPU, {NT} threads" {priv}
else:
    print("PUBLISH=False: nothing uploaded publicly. Review results first.")"""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU"}, "cells": []}
    for kind, src in CELLS:
        c = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        nb["cells"].append(c)
    out = ROOT / "notebooks" / "train_gpu.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
