# RE-Verify

Evidence-grounded relation extraction with a compact model that runs offline on a CPU.
**Model:** _not yet released_ · **Demo:** _not yet released_ · **Status: research in progress.** See [PROJECT_STATE.md](PROJECT_STATE.md) for what has actually been run and measured. No trained RE-Verify model exists yet, and no comparison with Claude Opus 5.5 has been run.

```
text (numbered sentences) ──► small LM (GGUF, llama.cpp, CPU) ──► compact JSON ──► deterministic verifier ──► public JSON
                                                                  {"s","r","o","ev"}   (entity-in-text, ontology,     (evidence ids expanded to
                                                                                         evidence check, dedup)        exact sentence text)
```

## Layout
| Path | Contents |
|---|---|
| `reverify/resources/ontology_v0.1.yaml` | 15 relations, canonical directions, DocRED mapping layer |
| `configs/train_*.yaml` | Experiment configs (CPU pilot, Qwen2.5-0.5B, Qwen3-0.6B, Qwen2.5-1.5B) |
| `reverify/data/` | download (sha256 manifest) · Re-DocRED conversion · synthetic hard negatives · build + validation + leakage report · challenge loader |
| `reverify/format.py` | prompts, target serialization, robust JSON parsing, public output |
| `reverify/verifier.py` | deterministic verifier (Architecture B) |
| `reverify/eval/metrics.py` | the single scorer used for every system, plus bootstrap CIs |
| `reverify/inference/llamacpp_runner.py` | resumable CPU evaluation runs |
| `reverify/eval/opus_competitor.py` | Claude Opus 5.5 runner (same inputs and scorer) |
| `reverify/train/sft.py` | LoRA/QLoRA SFT with hash-verified, resumable checkpoints |
| `reverify/export.py` | merge → GGUF → quantize |
| `notebooks/train_gpu.ipynb` | Kaggle/Colab GPU training |
| `benchmark/challenge_v1.entity_hashes.txt` | public hashes of the private, frozen hand-written test set (D-014) |
| `scripts/publish_hf.py`, `space/` | Hugging Face model card generator + upload; Gradio demo Space |
| `scripts/rescore.py`, `scripts/eval_hf_adapter.py`, `scripts/make_notebook.py` | re-score saved outputs; evaluate an adapter; regenerate the notebook |
| `PROJECT_STATE.md`, `EXPERIMENT_LOG.md`, `DECISIONS.md`, `FAILURES.md`, `docs/` | project memory |

## Install
```bash
# CPU inference only
pip install "re-verify[infer] @ git+https://github.com/YOUR_GITHUB_USER/re-verify" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
Usage with a released GGUF is shown in the model card.

## Reproduce (CPU, Windows/Linux, Python 3.10+)
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux: .venv/bin/pip
python -m reverify.data.download        # Re-DocRED + rel_info, sha256-verified
python -m reverify.data.build           # data/processed/*.jsonl + DATASET_REPORT.json (deterministic)
python -m pytest -q                     # the private challenge-set test is skipped in public clones
python -m reverify.inference.llamacpp_runner --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
       --data redocred_dev --limit 40 --run-id my_run --one-shot --no-grammar
```

## Train and publish
Step by step for Kaggle/Colab, Hugging Face and GitHub: **[docs/KAGGLE_COLAB_GUIDE.md](docs/KAGGLE_COLAB_GUIDE.md)**.
In short: open `notebooks/train_gpu.ipynb`, add an `HF_TOKEN` secret, set `REPO_URL`/`HF_USER`/`CONFIG`, Run All.
Checkpoints go to a private Hub repo, so a disconnected session resumes where it stopped.

## Licences
Code and released weights: Apache-2.0 (`LICENSE`). Data: Re-DocRED (MIT), DocRED relation info (MIT). Base models: Qwen2.5 / Qwen3 / SmolLM2 (Apache-2.0).
