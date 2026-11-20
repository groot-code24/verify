# Step by step: GitHub → Kaggle/Colab training → Hugging Face model + Space

Total hands-on time ≈ 30 min. GPU time ≈ 1–2 h per 0.5B experiment (estimate; the first run measures it).

---
## Part A — One-time accounts and tokens

**A1. GitHub.** Create a new **empty public** repository named `re-verify`. Don't add a README or licence; the project already has them.

**A2. Hugging Face.** Sign up at huggingface.co. Then go to *Settings → Access Tokens → Create new token*:
- Type **Write**. A fine-grained token also works if it can create and write repos in your namespace.
- Name it `re-verify`.
- Copy the token. Never paste it into a notebook cell or a file.

**A3. Kaggle** (skip if you'll use Colab):
- Sign up and **verify your phone number** (*Settings → Phone verification*). GPU and Internet stay disabled until you do.
- In any notebook: *Add-ons → Secrets → Add secret*. Label `HF_TOKEN`, value = the token from A2.

**A4. Colab** (skip if Kaggle): in a notebook, click the 🔑 *Secrets* icon → *Add new secret*. Name `HF_TOKEN`, value = token, and switch on *Notebook access*.

---
## Part B — Put the code on GitHub (from this machine, in `D:\verify`)

**B1. Keep the private test set private.** `benchmark/challenge_v1.jsonl` is already in `.gitignore`.
- Publishing it now would let future models, Claude Opus included, train on it, and it would stop being a fair held-out test.
- Back it up somewhere private, e.g. a private Hugging Face dataset repo. Its sha256 is `86efe21d72f123b08c6ecd6dca98484e9c4261a5d2c5657471c4ee44b773cfe5`.
- Only its entity-name hashes (`benchmark/challenge_v1.entity_hashes.txt`) are published, which is enough to rebuild the data exactly.

**B2. Commit and push:**
```powershell
cd D:\verify
git add -A
git status            # check: no .venv, models/, data/, checkpoints/, *.gguf, benchmark/challenge_v1.jsonl
git commit -m "RE-Verify: data pipeline, benchmark tooling, training/export/eval, Space"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USER/re-verify.git
git push -u origin main
```
GitHub Actions runs the test suite on every push (`.github/workflows/tests.yml`). Check the *Actions* tab: it should be green, with 1 test skipped (the private challenge set).

---
## Part C — Train on Kaggle (recommended) or Colab

**C1. Open the notebook.**
- **Kaggle:** *Create → New Notebook → File → Import Notebook* → upload `notebooks/train_gpu.ipynb` from your repo. Then in the right panel:
  - *Session options → Accelerator → GPU T4 x2* (the notebook uses one GPU on purpose)
  - *Internet → On*
  - *Add-ons → Secrets*: tick `HF_TOKEN` for this notebook
- **Colab:** *File → Open notebook → GitHub* → paste your repo URL → pick `notebooks/train_gpu.ipynb`. Then *Runtime → Change runtime type → T4 GPU*.
- Avoid Kaggle's **P100**: recent PyTorch builds no longer support its GPU architecture.

**C2. Edit cell 1:**
- `REPO_URL` = your GitHub repo
- `HF_USER` = your Hugging Face username
- Leave `CONFIG` as `configs/train_qwen25_05b_lora.yaml` for the first run (EXP-004)
- Leave `PUBLISH = False` for now

**C3. Run all cells** (Kaggle: *Run All*; Colab: *Runtime → Run all*). What each cell does and what to look for:

| Cell | Does | Check |
|---|---|---|
| 2 | Loads `HF_TOKEN` from secrets; pins one GPU | prints the platform; no assertion error |
| 4 | Installs deps; prints GPU | `Tesla T4 \| capability (7, 5)`; hardware line shows `"bf16": false` |
| 5 | Downloads and builds data; runs tests | **`data hashes: ALL MATCH`**; tests pass (1 skipped) |
| 6 | Trains | `[train cfg]` shows `micro_batch: 2, grad_accum: 8, load_dtype: fp32`; loss falls; `[sync] uploaded checkpoint-N -> hf.co/<you>/re-verify-checkpoints` every 100 steps |
| 7 | Merge → GGUF → quantize | prints sizes; q4_0 for 0.5B should be ≈ 400 MB |
| 8 | Dev evaluation on CPU | prints P/R/F1 for pipelines A, B, B+; uploads `results/` |

**C4. If the session dies** (timeout, disconnect, OOM kill):
- Open the notebook again and *Run All*.
- Cell 6 prints `[resume] from .../checkpoint-N`; that checkpoint came from your private hub repo and passed its sha256 check.
- An out-of-memory error is handled automatically: micro-batch is halved, then gradient checkpointing is enabled. Each change is logged in `oom_events.json`.

**C5. Second and third experiments.** Repeat C2–C3 with
- `CONFIG = "configs/train_qwen3_06b_lora.yaml"` (EXP-005). Its Q4_0 file measured **381.6 MB**, inside the 400 MB target.
- Optionally `train_qwen25_15b_lora.yaml` (EXP-006, accuracy-first; ~1 GB, misses the size target).

Change `MODEL_REPO` accordingly.

**C6. Bring results home.** The dev result JSONs are in your private repo under `results/<EXP>/runs/`. Download them into `D:\verify\results\runs\`, then ask me (or run `python scripts/rescore.py`) to compare experiments and pick the model by the rule in `docs/model_selection.md`.

---
## Part D — Publish on Hugging Face (model + Space)

**D1.** Pick the winning experiment. In its notebook set:
- `PUBLISH = True`
- `PUBLISH_PRIVATE = True`

Then *Run All*. Cell 6 sees that this experiment already finished, restores its adapters from your private checkpoint repo instead of retraining, and the remaining cells export, evaluate and publish. Cell 9 creates:
- `HF_USER/re-verify-<model>`: GGUF files (q4_0, q4_k_m, q8_0), the LoRA adapter, the dev results, and a **model card generated from those measured results**
- `HF_USER/re-verify-demo`: a Gradio Space on free CPU hardware running the same prompt → parser → verifier pipeline as the evaluation

**D2.** Open the Space page. The first build takes about 3–5 min. Try the built-in examples.

**D3.** Read the model card. If it's accurate, make both repos public: *Settings → Change repository visibility*. Keep `re-verify-checkpoints` **private**.

**D4.** Add the links to the GitHub README (*Model* and *Demo* lines) and push.

---
## What is NOT done by these steps
- **The Claude Opus 5.5 comparison.** It needs an Anthropic API key and runs once, on the frozen test sets, after model selection:
  ```powershell
  python -m reverify.eval.opus_competitor --data challenge_v1 --run-id opus55_challenge_v1
  ```
  Until it has run, the model card states that no comparison was made.
- **Final test-set numbers for RE-Verify** (`redocred_test`, `challenge_v1`). Same rule: run once, after model selection.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `HF_TOKEN secret missing` | Kaggle: tick the secret under *Add-ons → Secrets* for this notebook. Colab: enable *Notebook access* on the secret |
| `No GPU` assertion | Accelerator not set, or Kaggle phone verification missing |
| `data hashes: MISMATCH` | Don't train. Report the mismatching file; usually a different Re-DocRED revision on the Hub |
| pip resolver warnings about Kaggle's preinstalled packages | Usually harmless; the run only fails if an import fails in cell 4 or later |
| `convert_hf_to_gguf.py` error | llama.cpp's latest version broke compatibility. Re-run cell 7 after `git -C /kaggle/working/llama.cpp checkout <older commit>`; note the commit in FAILURES.md |
| Space build fails on `re-verify @ git+...` | The GitHub repo must be **public**, and `GIT_REF` (default `main`) must exist |
