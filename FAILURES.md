# Failures and fixes

## F-001 peft 0.17.1 incompatible with transformers 5.9 (2026-11-20)
- **Symptom:** `ImportError: cannot import name 'HybridCache' from 'transformers'`.
- **Fix:** upgraded to peft 0.21.3 / trl 1.14.2. Pinned in requirements.txt.

## F-002 Text normalization merged distinct entities (2026-11-20)
- **Symptom:** two metric unit tests failed (symmetric TP = 0, duplicate TP = 0).
- **Root cause:** `verifier.norm()` removed the articles a/an/the anywhere in a string, so "Company A" and "Company B" both became "company", and the entity "A" became the empty string. This would have corrupted entity resolution in scoring and the verifier's entity-in-text check.
- **Fix:** only a leading "the" is stripped (`reverify/verifier.py`). Regression test `test_norm`. Data rebuilt.

## F-003 Train/dev duplicate in Re-DocRED (2026-11-20)
- **Symptom:** leakage report showed one exact train→dev document overlap.
- **Fix:** `build.py` removes dev docs whose 8-gram overlap with train is ≥50% (removed `redocred-dev-00469`). The test split had no train overlap. One dev→test near-duplicate remains; it is harmless because dev is never trained on, and it is documented.

## F-004 Base 0.5B model unusable without fine-tuning (2026-11-20, expected)
- Zero-shot under grammar: always `{"relations":[]}`. One-shot: long lists of invented relations, 37–62% truncated at the token cap, F1 0.0 on 8 docs.
- Not a bug. It is the reason SFT is required (Gate 2 → Gate 3).

## F-005 Grammar-constrained decoding ~4× slower on CPU (2026-11-20)
- 40.2 s/doc with GBNF vs 9.5 s/doc without (Qwen2.5-0.5B Q4_0, 8 docs). Production decoding drops the grammar (D-010).

---
# Code review 2026-11-20 (second pass): every item below was reproduced before it was fixed; regression tests in `tests/test_regressions.py` and `tests/test_hub_sync.py`

## F-006 Entity grounding used plain substring matching
- **Repro:** verifier kept `{"s": "Ma", ...}` for text "Marta owns Beta Corp.". The scorer's `ungrounded` metric had the same flaw.
- **Fix:** `verifier.contains()` does token-boundary matching on normalized text; used by verifier and scorer.

## F-007 Hedge filter treated the month "May" as a hedge
- **Repro:** "In May 2020, Acme acquired Beta." → dropped as `hedged_evidence` (regex was case-insensitive).
- **Fix:** "may" matches lower case only; other cues stay case-insensitive.

## F-008 One malformed model output crashed the scorer
- **Repro:** `{"s": null, ...}` → `AttributeError` inside `score_doc` (pipeline A scores raw output). `"ev": "0, 1"` became a set of characters.
- **Fix:** `format.parse_output` coerces every relation to `{s:str, r:str, o:str, ev:[int]}` or drops it.

## F-009 Strict name matching scored correct extractions as errors
- **Repro:** challenge ch-006/012/060: "Kestrel electric scooter" vs gold "Kestrel" → FP + FN. This measures span conventions, not understanding.
- **Fix:** lenient *unique* matching (protocol v1.1, D-015). Exact match first. Otherwise a token-bounded containment that adds at most 3 tokens, accepted only if exactly one entity matches. Single-token fragments never match by containment. Strict-match F1 is still reported. All saved runs were re-scored (`scripts/rescore.py`): changes ≤ +0.002 F1, conclusions unchanged.

## F-010 GPU auto-config would have failed on the first Kaggle step
- **OOM:** `adapt()` picked micro-batch 16 × 1024 on a T4. Logits alone for a 152k vocabulary are ≈ 14.9 GB on a 14.7 GB GPU.
- **bf16:** `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and reports True on a T4 (no native bf16).
- **dtype:** fp16 weights + fp16 AMP risks overflow in Qwen activations.
- **Fix:**
  - Micro-batch now accounts for logits (T4/0.5B → 2 × 8 accumulation).
  - bf16 is decided by capability ≥ 8.0.
  - On fp16 GPUs the model is loaded in fp32 with fp16 autocast.
  - An OOM at runtime halves the micro-batch, then enables gradient checkpointing, then resumes from the newest verified checkpoint (`oom_events.json`).
  - The notebook pins one GPU (T4 x2 would have doubled the batch silently via DataParallel).

## F-011 Contradictory supervision from Re-DocRED acquisition documents
- **Repro:** 163 train docs state acquisitions/investments/partnerships. DocRED's schema cannot label them, so they become silent negatives for ACQUIRED, while 131 of 451 explicit OWNS/PARENT_ORG_OF triples come from acquisition sentences. Meanwhile the synthetic data teaches "X acquired Y" → ACQUIRED.
- **Fix (D-013):** `redocred_train_clean.jsonl` drops 173 cue docs (2,880 kept). The GPU configs use it (data-v1.1). The ontology states the policy (acquisition → ACQUIRED only). Dev/test are unchanged. Ablation EXP-007 planned.

## F-012 Kaggle checkpoint "sync" directory was not persistent
- `/kaggle/working/sync` lives on the same disk as the run and does not survive a crashed interactive session, so the recovery design protected nothing on Kaggle.
- **Fix:** checkpoints and adapters are uploaded to a private HF repo (`hub_repo`); `--resume auto` pulls the newest verified one.

## F-013 Runner efficiency stats could include documents outside the evaluated subset (resumed run files)
- **Fix:** `evaluate_outputs` restricts outputs to the evaluated documents.

## F-014 No best-checkpoint tracking
- **Fix:** `metric_for_best_model=eval_loss`; `best_adapter` is exported next to `final_adapter`.

## F-015 Export path would have failed or been slow on Linux
- `pip install gguf` (PyPI) can lag behind `convert_hf_to_gguf.py` on llama.cpp master. The notebook now installs `gguf-py` from the same checkout.
- llama-cpp-python 0.3.20 has **no Linux wheel** (checked the wheel index), so the notebook and Space would have compiled from source. They now pin 0.3.19, which has prebuilt CPU wheels.
- Quantize binding verified locally: Qwen3-0.6B Q8_0 → Q4_0 = 381.6 MB, loads and generates.

## F-016 Ontology file lived outside the Python package
- `pip install` (the Hugging Face Space) would ship without it. Moved to `reverify/resources/`, declared as package data, and verified in a clean torch-free venv with a non-editable install.

## F-017 Publishing the repo would have published the private test set
- The synthetic generator read `challenge_v1.jsonl` to exclude its entity names, so the file was needed to rebuild data.
- **Fix:** it now uses sha256 hashes of the normalized names (`challenge_v1.entity_hashes.txt`, public). The file is git-ignored. The data build is byte-identical with or without it (verified), and its tests skip in public clones.

## F-018 A patch corrupted the hedge regex with literal backspace characters (introduced and caught during this review)
- A shell heredoc halved the doubled backslash before `b`, so Python read the string escape for a backspace (0x08) instead of the regex word boundary. Two existing tests failed immediately.
- **Fix:** rewritten without escape processing. A repo-wide scan finds no other control characters.

## F-019 Experiments sharing one checkpoint repo would resume from each other's checkpoints (introduced and caught during this review)
- **Fix:** hub paths are prefixed with the experiment id; covered by `test_experiments_sharing_a_repo_stay_isolated`.

## Not bugs (checked)
- Data build determinism: identical sha256 under different `PYTHONHASHSEED` values, and with or without the private challenge file.
- Prompt/template parity between training (`apply_chat_template`) and inference (`chatml()`) for Qwen2.5, Qwen3 (`enable_thinking=False` ↔ `--no-think`) and SmolLM2.
- llama.cpp quantization type constants (the hard-coded enum values were correct; now read from the library).
