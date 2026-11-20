# Experiment log

Machine for all local runs: AMD Ryzen 3 7320U (4C/8T), 7.3 GB RAM (~0.7 GB available), Windows 11, llama-cpp-python 0.3.20, 4 threads. Raw outputs are in `results/raw_outputs/`, scores in `results/runs/`.

---
## EXP-001 Data pipeline v1 (data-v1) — 2026-11-20 — DONE
- **Hypothesis:** Re-DocRED mapped to ontology v0.1 gives enough explicit business-relation supervision; templates fill relations missing from it.
- **Result:** `data/processed/DATASET_REPORT.json`.

  | Split | Docs | Explicit triples (ontology v0.1) |
  |---|---|---|
  | Re-DocRED train | 3,053 | ≈6,100 |
  | Re-DocRED dev | 499 | — |
  | Re-DocRED test | 500 | — |
  | Synthetic train | 3,000 | 4,399 (409 abstain docs) |
  | Synthetic dev | 300 | — |

  - 0 invalid spans, 0 dangling entities, 0 duplicate triples.
  - **Leakage:** train→test 0 near-duplicates; all train→challenge 0 shared sentences; one train→dev duplicate removed (F-003).
- **Decisions triggered:** D-005 (LOCATED_IN was 71% of mapped triples) and D-008.

---
## EXP-002 Gate 2 baselines: untuned base models, one-shot prompt, no grammar — 2026-11-20 — DONE
- **Setup:** full definitions prompt + one-shot example, greedy decoding, max 384 new tokens. Fixed seeded subsets of 40 docs per dataset. Runs: `base_*`.
- **Smoke tests (8 docs, not kept):** zero-shot with grammar → always empty; one-shot with grammar → 40.2 s/doc (vs 9.5 s without) → D-010.

| Model (file) | Data | Arch | P | R | Micro F1 [95% CI] | Unsupported | Abst. acc | Mean latency | Trunc. |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 (428 MB) | Re-DocRED dev | A | .002 | .014 | .004 [0, .011] | .998 | 0.0 | 12.2 s | 75% |
| | | B | .016 | .014 | .015 [0, .054] | .984 | 0.4 | | |
| | synthetic dev | A | .008 | .047 | .014 [0, .033] | .992 | 0.0 | 6.6 s | 48% |
| | | B | .045 | .047 | .046 [0, .099] | .955 | 0.25 | | |
| | | B+ | .064 | .047 | .054 [0, .112] | .936 | 0.5 | | |
| Qwen3-0.6B Q8_0 (639 MB) | Re-DocRED dev | A | .000 | .000 | .000 | 1.000 | 0.95 | 15.4 s | 15% |
| | | B | — | .000 | .000 | — (all dropped) | 1.0 | | |
| | synthetic dev | A | .165 | .328 | .220 [.126, .310] | .835 | 0.0 | 7.1 s | 15% |
| | | B | .250 | .328 | .284 [.184, .398] | .750 | 0.0 | | |
| | | B+ | .356 | .328 | **.342 [.220, .465]** | .644 | 0.75 | | |

- **Findings:**
  1. Untuned ≤0.6B models cannot do the task; Re-DocRED F1 ≈ 0.
  2. The dominant error is `fp_unresolved_entity`: argument strings that aren't real entities (e.g. "In 2000", "120 million").
  3. Neither the verifier (B) nor the hedge filter (B+) lost a single true positive in any run, while precision rose up to 2×. Early, low-n evidence for Gate 5; must be re-measured on fine-tuned models.
  4. Qwen3-0.6B > Qwen2.5-0.5B untuned on synthetic, but this says little about fine-tuned ranking (D-013 pending EXP-004/005).
- **Decision:** proceed to SFT (Gate 3).

---
## EXP-003 Gate 3 CPU pilot: SmolLM2-135M LoRA r8, 40 steps, interrupt + resume — 2026-11-20 — DONE (mechanics PASS; model quality FAIL, as expected)
- **Purpose:** prove mechanics (data → masked loss → eval → checkpoint → crash → resume from persistent mirror), not model quality.
- **Config:** `configs/train_pilot_cpu.yaml`. 395/450 train and 44/50 eval examples fit 512 tokens (the rest are dropped, never truncated). 2.44M trainable params (1.78%).
- **Session 1** (`--stop-after-steps 20`), 675 s wall, ~30 s/step on CPU:
  - Train loss 2.01 (step 5) → 1.34 (step 10) → 0.88 (step 20). Eval loss at step 20: **0.702**.
  - Checkpoints 10 and 20 manifest-verified (sha256) and mirrored to `checkpoints_sync/`.
- **Session 2:** local `checkpoints/pilot_cpu/` deleted to simulate a wiped notebook disk, then `--resume auto`, 655 s wall.
  - Found the newest verified checkpoint (step 20) in the mirror, copied it back, and resumed.
  - The LR continued mid-cosine (1.24e-4 at step 25; no warmup restart). Log history holds both sessions.
  - Train loss 0.63 (step 25) → 0.46 (step 40). **Eval loss 0.702 → 0.506.** Final adapter saved and mirrored.
- **Functional check** (`scripts/eval_hf_adapter.py`; 30 synthetic dev docs; same short prompt for both; greedy decoding; 256 max new tokens):

  | | schema prefix | 1st subject is a text span | 1st label in ontology | truncated | micro F1 (A/B/B+) |
  |---|---|---|---|---|---|
  | SmolLM2-135M base | 0/30 | 0/30 | 0/30 | 1/30 | 0 / 0 / 0 |
  | + pilot LoRA (40 steps) | 30/30 | 28/30 | 18/30 | **30/30** | 0 / 0 / 0 |

- **Conclusion:** the pipeline transmits signal end to end (format and grounding learned from ~160 examples). Termination and relation semantics are not learned: outputs loop on repeated keys. One observed hard-negative failure: "abandoned its bid" → ACQUIRED. A 135M model at 40 steps proves nothing about final quality; real runs need the GPU (EXP-004+).
- **Follow-ups:**
  1. Add a repetition guard to decoding (llama.cpp `repeat_penalty` or stop at the closing `]}`). To be measured on a trained model, not assumed.
  2. Watch termination rate as a first-class metric in EXP-004.

---
## Code review re-score — 2026-11-20 — DONE
- After F-006 to F-009 (grounding, parser sanitization, lenient unique matching), every saved run was re-scored from its raw outputs with `scripts/rescore.py`. No model was re-run.
- **Changes:** at most +0.002 micro F1. Qwen3-0.6B synthetic dev is now A 0.220 / B 0.286 / B+ 0.344; Qwen2.5-0.5B synthetic dev B 0.047 / B+ 0.055. All others unchanged.
- **Conclusions of EXP-002/003 unchanged.** Previous values are kept inside each run file as `previous_scores`.

## Quantization check — 2026-11-20 — DONE
- `python -m reverify.export --quantize-only models/Qwen3-0.6B-Q8_0.gguf --quants q4_0` → 381.6 MB, loads and generates in llama-cpp-python 0.3.20.
- This is a size measurement only. Accuracy after quantization must come from a fine-tuned model (Gate 6). Re-quantizing from Q8_0 loses slightly more than quantizing from f16, which is what the export path does for trained models.

## Planned
- **EXP-004** Qwen2.5-0.5B LoRA r16 (data-v1.1), Kaggle T4.
- **EXP-005** Qwen3-0.6B LoRA r16 (data-v1.1).
- **EXP-006** Qwen2.5-1.5B (optional, accuracy-first).
- **EXP-007** best of 004/005 with `redocred_train` instead of `redocred_train_clean`: ablation for D-013.
