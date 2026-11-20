# PROJECT_STATE — RE-Verify

_Last updated: 2026-11-20 (after code review). Read this first after any restart._

## Current phase
**Ready for GPU training and publishing.** Code review complete: 14 bugs found and fixed (F-006 to F-019), each with a regression test. Blocked only on running the notebook on Kaggle/Colab (user accounts and tokens).

## Gates
| Gate | Status | Evidence |
|---|---|---|
| 1 Feasibility | **PASS (CPU)** / GPU untested | llama.cpp load 1.0–1.7 s; deps verified in a clean venv; data sha256 verified and deterministic |
| 2 Baseline | **PASS** | EXP-002 (re-scored): untuned models score Re-DocRED dev F1 0.000–0.015 and synthetic dev up to 0.344 (Qwen3, B+) |
| 3 Pilot training | **PASS (mechanics)** | EXP-003: crash → resume from mirror; eval loss 0.702 → 0.506. Hub-based resume unit-tested (fake hub) |
| 4 Competitive performance | NOT STARTED | Run `notebooks/train_gpu.ipynb` (EXP-004/005) |
| 5 Reliability | PARTIAL (untuned only) | Verifier/hedge filter raise precision with zero TP loss on baselines |
| 6 Compression | PARTIAL | Quantize path verified (Qwen3-0.6B Q4_0 = 381.6 MB); GGUF conversion runs in the notebook |
| 7 CPU deployment | PARTIAL | Runner + Space app verified end to end on CPU with a base GGUF (gradio 5.9.1, hub 1.4.1) |
| 8 Frontier comparison | **BLOCKED** | No Anthropic credentials. **Competitive claim untested.** |

## Dataset versions
`data-v1.1` (adds `redocred_train_clean`, D-013), ontology `0.1` (now at `reverify/resources/ontology_v0.1.yaml`). Hashes are in `data/processed/DATASET_REPORT.json` and embedded in the notebook's integrity check.
Challenge set: **private** (git-ignored, D-014), sha256 `86efe21d72f123b08c6ecd6dca98484e9c4261a5d2c5657471c4ee44b773cfe5`.

## Best checkpoint / validation score
Pilot only (not a candidate). No real candidate trained yet.

## Publishing assets (ready, not yet used)
- GitHub: `LICENSE` (Apache-2.0), `pyproject.toml`, CI (`.github/workflows/tests.yml`). The publish set was audited: 67 files, no private test set, no secrets.
- Hugging Face: `scripts/publish_hf.py` (model card generated from measured runs; private by default) and `space/` (Gradio demo).
- Guide: `docs/KAGGLE_COLAB_GUIDE.md`.

## Known gaps
1. No GPU or accounts here: training, GGUF conversion of a trained model, and publishing all happen in the notebook.
2. ACQUIRED / INVESTED_IN / PARTNERED_WITH come from templates only. No temporal attribute (D-006). VERIFY mode not implemented.
3. Research-log entries marked K need primary-source checks.

## Strategy for the Opus comparison
Teacher→student distillation within the Qwen3 family (identical tokenizer and Apache-2.0 verified for 0.6B–14B): `docs/TRANSFER_DISTILLATION_PLAN.md`. Plan only, not started. Student = Qwen3-0.6B (Q4_0 381.6 MB).

## Next action
1. **User:** follow `docs/KAGGLE_COLAB_GUIDE.md` Parts A–C. Push to GitHub, then run EXP-004 and EXP-005.
2. **Me:** compare dev results → pick model → EXP-007 ablation → error analysis on dev.
3. **User:** publish (Part D) only after reviewing the generated model card.
