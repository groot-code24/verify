# Decisions

Each entry: problem → options → choice → evidence → status. Newest last.

## D-001 Training happens on free cloud GPUs; this machine does data, CPU inference and pilots (2026-11-20)
- **Measured:** Ryzen 3 7320U (4C/8T), 7.3 GB RAM with ~0.7 GB available during the session, AMD iGPU (no CUDA), 125 GB free on D:, 9.7 GB on C:.
- **Choice:** Full LoRA runs go through `notebooks/train_gpu.ipynb` on Kaggle/Colab. Locally: data pipeline, GGUF CPU inference/benchmarks, and a 135M-parameter CPU pilot that proves the mechanics.
- **Consequence:** Gates 4–8 need the user to run the notebook (no Kaggle/Colab credentials here).

## D-002 Datasets: Re-DocRED (+ DocRED relation info) as the real-data backbone; TACRED excluded (2026-11-20)
- **Options:** TACRED / Re-TACRED (LDC licence, only a loader script on HF, data not downloadable), DocRED (MIT, test labels hidden), Re-DocRED (MIT, re-annotated DocRED with labeled test split and evidence sentences), FewRel (relation classification only, no evidence and not document-level), REBEL (distantly supervised, noisy, CC BY-NC-SA).
- **Choice:** Re-DocRED train/dev/test. It has document-level relations, evidence sentences for 55% of labels, and a permissive licence.
- **Gap:** Re-DocRED has no ACQUIRED, INVESTED_IN or PARTNERED_WITH labels. These come only from the template generator (D-007) and are measured on the hand-written challenge set.

## D-003 One canonical direction per relation (2026-11-20)
- The spec listed inverse pairs (ACQUIRED/ACQUIRED_BY, OWNS/OWNED_BY, PARENT_OF/SUBSIDIARY_OF). Keeping both makes "X ACQUIRED_BY Y" and "Y ACQUIRED X" two correct answers and hides direction errors.
- **Choice:** One canonical label per relation; inverse surface forms are normalized in the mapping layer (`P749^-1`, `P127^-1`, ...). A reversed triple is scored as `fp_reversed`.

## D-004 Evidence is emitted as sentence ids, not copied text (2026-11-20)
- On CPU every output token costs ~40–100 ms. Copying evidence sentences would roughly triple output length.
- **Choice:** The model emits `"ev":[i,...]`; the API layer (`format.to_public_json`) expands ids to the exact sentence text, so the evidence string is always an exact substring by construction. Evidence quality is scored as sentence-set F1 against gold evidence.

## D-005 LOCATED_IN removed from ontology v0.1 (2026-11-20)
- **Evidence:** After mapping, LOCATED_IN (P131/P17/P276) was 14,867 of 20,987 explicit training triples (71%).
- Most of it is geographic containment that DocRED annotators inferred from world knowledge (P17 "country"). It is off-mission for business relation extraction and would dominate both the loss and the output length (latency).
- **Choice:** Excluded. HEADQUARTERED_IN kept. Revisit only if a user need appears.

## D-006 No current/former temporal attribute in v0.1 (2026-11-20)
- Re-DocRED's P108 (employer) does not distinguish current from former employment, so any temporal label on the real data would be a guess.
- **Choice:** EMPLOYEE_OF/EXECUTIVE_OF cover "is or was". Future and planned roles ("will join") are NOT extracted. A temporal attribute is deferred to v0.2 and needs curated data. **This spec requirement is not yet met.**

## D-007 Synthetic data: templates with correct-by-construction labels; no LLM teacher yet (2026-11-20)
- No API access means no teacher model. The template engine (`reverify/data/synthetic.py`) generates positives for the missing relations plus hard negatives (speculation, negation, failed deals, future, co-mention, publisher-vs-developer, seller-vs-manufacturer, customer-vs-employee).
- Name pools exclude every challenge-set entity. The leakage check finds 0 shared sentences with the challenge set.
- **Mix:** 3,000 synthetic + 3,053 Re-DocRED training docs (synthetic is ≤50% by doc count). Risk: template phrasing generalizes poorly. That is exactly what the challenge set measures.

## D-008 Gold triples with no textual support are ignored in scoring (2026-11-20)
- **Classification:** `explicit` = gold evidence exists, or subject and object co-occur in a sentence (heuristic evidence). `implicit` = neither (2,199 of 6,017 dev triples before the D-005 filter).
- Implicit triples are excluded from training targets, and predictions that match them are neither TP nor FP. The model is trained not to infer from world knowledge, but isn't penalized when it does on those few cases.

## D-009 Benchmark governance (2026-11-20)
- **Test sets:** `challenge_v1` (100 hand-written items, sha256 `86efe21d…`) and `redocred_test` (500 docs). Both frozen; neither is used for training, prompt tuning or error analysis.
- **Development data:** `redocred_dev` (499 docs after leakage removal) and `synthetic_dev` (300) only.
- **Disclosed bias:** the challenge set was written by Claude Opus 5.5 (this session), which is also the competitor. If anything this favours Opus (shared labelling conventions).

## D-010 No GBNF grammar in production decoding (2026-11-20)
- **Measured:** Qwen2.5-0.5B Q4_0 with a one-shot prompt took 40.2 s/doc with grammar vs 9.5 s/doc without (8 synthetic dev docs).
- The zero-shot model under grammar always emitted an empty list.
- **Choice:** Decode freely; rely on `parse_output` (salvages truncated JSON) plus the deterministic verifier.

## D-011 Opus competitor: no refusal fallbacks, explicit effort, no temperature (2026-11-20)
- A server-side fallback would answer some documents with a different model and invalidate the comparison. Refusals are logged and scored as empty predictions.
- Opus 5.5 rejects `temperature` and always thinks; effort is pinned (`high`) and stored with every output.
- Opus gets the full definitions + one-shot prompt; RE-Verify gets the short prompt. The prompt gap favours Opus.

## D-012 LoRA in 16-bit for ≤1.5B models; QLoRA only when weights exceed ~45% of GPU memory (2026-11-20)
- A 0.5–1.5B model in fp16 takes 1–3 GB, which fits a T4/P100 (16 GB) with room for batch 8×1024. 4-bit would add quantization noise for no memory benefit.
- `reverify/hardware.py:adapt` switches to QLoRA automatically for larger models.

## D-013 Train on Re-DocRED minus documents that state unlabelable relations (2026-11-20)
- **Evidence:** F-011.
- **Choice:** `redocred_train_clean` (2,880 of 3,053 docs) in all GPU configs. Ontology policy: an acquisition statement yields ACQUIRED only.
- **Cost:** some Re-DocRED evaluation gold labels an acquisition as OWNS/PARENT_ORG_OF. A model following the policy loses those points on `redocred_dev`/`redocred_test` (conservative for RE-Verify).
- **Ablation EXP-007:** same config with `redocred_train`. Keep the filter only if challenge-relevant dev metrics don't drop.

## D-014 The challenge set stays private until the frozen comparison has run (2026-11-20)
- Publishing it on GitHub makes it trainable for future models, including the competitor. It is git-ignored; only entity-name hashes are public, which is enough to rebuild the data byte-for-byte.
- **User action:** keep a private backup (sha256 `86efe21d…`). Publish it, with that hash, after Gate 8.

## D-015 Benchmark protocol v1.1: lenient *unique* entity matching is primary; strict is also reported (2026-11-20)
- Decided before any test-set evaluation of any system (no test numbers exist yet), so it cannot favour anyone. Applies identically to every system. Rationale: F-009.

## D-016 Linux runtime pins llama-cpp-python 0.3.19 (prebuilt CPU wheel); Windows dev box stays on 0.3.20 (2026-11-20)
- The wheel index has no 0.3.20 Linux wheel. Source builds on Kaggle/Spaces take 10+ min and can time out.

## D-017 Licence: Apache-2.0 for code and released weights (2026-11-20)
- Matches the base models (Qwen2.5/Qwen3: Apache-2.0); Re-DocRED is MIT. The project owner can change this before the first public push.

## D-018 Publishing flow: private first, model card generated from measured results only (2026-11-20)
- `scripts/publish_hf.py` builds the card from `results/runs/*.json` passed explicitly. With no competitor run supplied, the card states that no Opus comparison was made.
- Repos are created private by default; the owner flips visibility after reading the card.

## D-019 Distillation route uses the Qwen3 family; Qwen3-0.6B becomes the leading student (2026-11-20)
- **Verified on the Hub:** Qwen3 0.6B–14B are all Apache-2.0 with a byte-identical tokenizer.json (vocab 151,936), which enables exact logit-level KD.
- Qwen2.5-3B is not Apache-2.0 (licence "other"): excluded as a teacher.
- Qwen3-0.6B Q4_0 = 381.6 MB (inside the target) vs Qwen2.5-0.5B Q4_0 = 428.7 MB. EXP-004 (Qwen2.5) still runs as a comparison point.

## D-020 Opus 5.5 is measured, never used as a training-data source (2026-11-20)
- Anthropic's terms restrict using Claude outputs to develop competing models. No Opus output is used as a label, filter or training example.
- Teachers are open Apache-2.0 models; silver data is audited by a human.
