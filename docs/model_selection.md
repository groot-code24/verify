# Model selection (provisional, 2026-11-20)

Selection rule: the smallest model whose **fine-tuned, quantized** dev micro F1 (architecture B, `redocred_dev` + `synthetic_dev`) is within 2 points of the best candidate. Ties are broken by package size, then CPU latency. Nothing below is decided by benchmark popularity. Every row marked "planned" still needs its GPU run.

| Role | Model | Params | Licence | Est. deploy size | Fine-tune cost (T4) | CPU runtime | Status |
|---|---|---|---|---|---|---|---|
| **Primary** | Qwen2.5-0.5B-Instruct | 0.49B | Apache-2.0 | Q4_0 **428.7 MB** (measured file); Q3_K_M 432 MB on Hub (embeddings dominate) | ~20–30 min, 2 epochs × 6k docs (estimate) | llama.cpp; measured 1.0–1.7 s load, ~25 tok/s decode on dev CPU | EXP-004 planned |
| Backup | Qwen3-0.6B | 0.6B | Apache-2.0 | Q8_0 639 MB; **Q4_0 381.6 MB (measured, re-quantized from Q8_0; inside target)** | similar | llama.cpp (supported in 0.3.20) | EXP-005 planned |
| Smaller | SmolLM2-360M-Instruct | 0.36B | Apache-2.0 | Q8_0 ≈ 386 MB, Q4 ≈ 230 MB (estimates) | ~15 min | llama.cpp | add if 0.5B is clearly sufficient |
| Accuracy-first | Qwen2.5-1.5B-Instruct | 1.54B | Apache-2.0 | Q4_0 ≈ 935 MB (estimate) — misses the 400 MB target | ~1 h | llama.cpp | EXP-006 planned |
| Compression candidate | best of the above at Q4_0 → Q3_K_M → Q2_K | — | — | measured per variant | none | llama.cpp | Gate 6 |

## Rejected or deferred
- **BitNet b1.58 2B4T:** the 1.19 GB file is larger than a Q4 0.5B. It needs the bitnet.cpp runtime, and ternarizing an existing model post hoc is not what that paper validates. Infeasible on free compute.
- **Gemma-3 270M/1B, Llama-3.2-1B:** gated; no HF token in this environment. Gemma-3-270M is worth testing if the user provides a token.
- **LFM2-350M/1.2B-Extract:** strong fit for extraction, but the LFM Open License has commercial-use conditions. Testable for research if the user accepts the licence.

## Why not decide now
Measured base-model behavior (EXP-002) gives no signal: every untuned candidate scores ≈0. Model choice therefore has to come from fine-tuned dev results, which need GPU time.
