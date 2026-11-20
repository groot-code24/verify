# Research log

**Provenance column:** **V** = verified in this session (Hub API query, file download, or measurement); **K** = prior knowledge of the publication, not re-checked online in this session. K entries need a primary-source check before they are cited in a final report.

## Datasets and licences

| Source | Date checked | Finding | Prov. | Relevance / decision |
|---|---|---|---|---|
| `tonytan48/Re-DocRED` (HF) | 2026-11-20 | MIT; train 3,053 / dev 500 / test 500 docs; 85,932 train labels, 55% with evidence sentences; test labels present | V | Backbone (D-002) |
| Tan et al. 2022, "Revisiting DocRED – Addressing the False Negative Problem in Relation Extraction" (EMNLP) | — | Re-annotation adds many missing triples; DocRED's false negatives made precision look worse than it was | K | Explains why added triples lack evidence → D-008 |
| `thunlp/docred` (HF) | 2026-11-20 | MIT; 96 Wikidata relations; test labels hidden; 91 MB distant-supervision split | V | Used only for `rel_info` |
| Yao et al. 2019, "DocRED" (ACL) | — | Doc-level RE over Wikipedia; ~40% of facts need multi-sentence reasoning; evidence sentences annotated | K | Task framing |
| `DFKI-SLT/tacred` (HF) | 2026-11-20 | Licence "other"; loader script only — data requires an LDC licence | V | Excluded (D-002) |
| `Babelscape/rebel-dataset` | 2026-11-20 | CC BY-SA 4.0 on Hub card; 1.49 GB; distantly supervised | V | Not used v1 (noise, size). Candidate for org-relation pretraining later |
| `Babelscape/SREDFM` | 2026-11-20 | CC BY-SA 4.0; multilingual Wikidata-aligned | V | Not used v1 |
| `thunlp/few_rel` | 2026-11-20 | MIT; loader script | V | Not used: sentence-level classification, no evidence/abstention |
| `DFKI-SLT/scierc` | 2026-11-20 | 401 / not accessible | V | Skipped |

## Base models (all queried on the Hub 2026-11-20)

| Model | Licence | GGUF on Hub | Prov. | Notes |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Apache-2.0 | Q4_0 428.7 MB, Q4_K_M 491 MB, Q2_K 415 MB | V | Primary candidate. Large 151k vocab → embeddings ≈ 27% of params, so Q4 lands near 400 MB |
| Qwen3-0.6B | Apache-2.0 | Q8_0 639 MB only (official) | V | Alternative; needs `no_think` format |
| Qwen2.5-1.5B-Instruct | Apache-2.0 | — | V | Accuracy-first upper bound (~1 GB at Q4) |
| SmolLM2-135M / 360M / 1.7B-Instruct | Apache-2.0 | — | V | 135M used for CPU pilot only; 360M is a compact-first candidate (~230 MB Q4) |
| LiquidAI LFM2-350M/1.2B (+ "-Extract") | "other" (LFM Open License) | — | V | Purpose-built for extraction; licence has commercial-use conditions — flagged, not selected v1 |
| google/gemma-3-270m-it, gemma-3-1b-it | Gemma terms, **gated** | — | V | Needs HF token + licence acceptance; not available here |
| meta-llama/Llama-3.2-1B-Instruct | Llama 3.2 licence, **gated** | — | V | Same |
| microsoft/bitnet-b1.58-2B-4T-gguf | MIT | i2_s 1,188 MB | V | Ternary 2B is *larger* on disk than a Q4 0.5B; needs the bitnet.cpp runtime. Not a fit for the 400 MB target |
| numind/NuExtract-2.0-2B, -1.5-tiny | MIT | — | V | Generic JSON extraction; possible teacher/baseline later |

## Methods

| Topic | Source | Finding | Prov. | Can reproduce on free GPU? | Test it? |
|---|---|---|---|---|---|
| LoRA | Hu et al. 2021 | Low-rank adapters match full fine-tuning on many tasks at <1% trainable params | K | Yes | Yes (EXP-004..006) |
| QLoRA | Dettmers et al. 2023 | 4-bit NF4 base + LoRA; enables 33–65B on one GPU | K | Yes | Only if model >~3B (D-012) |
| BitNet b1.58 | Ma et al. 2024; "BitNet b1.58 2B4T" tech report 2025 | Ternary weights trained *from scratch*; post-hoc ternarization of a pretrained model loses much more accuracy | K | No (needs pretraining-scale compute) | No — record as infeasible |
| Generative doc-level RE with small LMs | various 2023–25 | Zero-shot LLMs score far below fine-tuned models on DocRED-style annotation conventions | K | — | Confirmed locally: base 0.5B scores F1 0.0 on 8 docs (EXP-002) |
| Hard negatives / contrastive data | various | Targeted negatives reduce false positives on speculation/negation | K | Yes | Yes — ablation planned (EXP-007) |
| Grammar-constrained decoding | llama.cpp GBNF | Guarantees schema; cost measured here at ~4× latency on this CPU | V | — | Rejected for production (D-010) |
