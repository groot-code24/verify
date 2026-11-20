# Plan: transfer learning + teacher→student distillation to beat Claude Opus 5.5 on RE-Verify's task

_Written 2026-11-20. Status: **plan only — nothing below has been run.** Every phase has a hypothesis, a pre-registered
keep/reject rule, and a gate. A phase that fails its gate stops the plan for re-planning; it is never reinterpreted
as a success._

## 0. The idea in one paragraph
RE-Verify already uses transfer learning: it fine-tunes a pretrained Qwen model, so the model keeps what Qwen learned
from trillions of tokens. This plan adds a second transfer:
1. Fine-tune a **larger Qwen3 "teacher"** (4B, then 8B if needed) on the task.
2. Use it to (a) label a large pool of *real* business text and (b) supply its full next-token probability distributions.
3. Train the **small Qwen3-0.6B "student"** on both.

The student is what ships (≈ 382 MB at Q4_0, CPU).

**Why Qwen3:** verified on the Hub (2026-11-20). Qwen3-0.6B, 1.7B, 4B, 8B and 14B are all Apache-2.0 and share a
**byte-identical tokenizer** (vocab 151,936), so teacher and student probabilities line up token for token.
Qwen2.5-3B is *not* Apache-2.0 and is excluded.

## 1. Can it beat Opus 5.5? Honest assessment before spending GPU time
- **Re-DocRED-style test (`redocred_test`): realistic.** A fine-tuned model learns the dataset's annotation conventions
  (which facts count as "stated", which span to output). A general model, however strong, does not. Published
  fine-tuned DocRED systems score far above zero-shot LLMs. _(Prior knowledge, not yet re-checked: research-log entry K.)_
- **Hard hand-written cases (`challenge_v1`): uncertain.** Coreference, contradictions, quoted claims and hypotheticals
  reward general reasoning, which is Opus's strength. With 100 items, only a large margin is detectable. "Not
  demonstrated" is a likely honest outcome on this set even if we win on the other.
- **Ceiling:** a student rarely exceeds its teacher. **The teacher must beat Opus on dev first (Gate T)**, otherwise
  distillation cannot get the student there.
- We **cannot know the target without measuring Opus.** Phase 1 does that on dev data only.

## 2. Ground rules (apply to every phase)
1. **Opus outputs are never used as training data, labels or filters.** Anthropic's terms restrict using Claude outputs
   to develop competing AI models. Check the terms that apply to your account (this is not legal advice). Opus is
   only *measured*. Note: the current template sentences and the challenge set were written in a Claude session. If
   your terms review flags that, replace the templates with human-written ones (a small change to
   `reverify/data/synthetic.py`).
2. **Test sets stay frozen and unseen** (`redocred_test`, `challenge_v1`): one evaluation, at the end (Phase 8).
3. **One change per experiment.** Results go in EXPERIMENT_LOG.md, decisions in DECISIONS.md.
4. **The shipped model stays ≤ 400 MB, CPU-only.** The teacher never ships.

## 3. Phases

### Phase 1 — Measure the target (Opus on DEV only)  · needs: Anthropic API key · cost: API tokens only
- **New:** `benchmark/challenge_dev_v1.jsonl`, ~80 items in the same 28 categories as `challenge_v1`.
  - Fresh entity names; leakage check against train, synthetic and `challenge_v1`.
  - Written or reviewed by a human; used for model selection and error analysis only.
- **Run** `reverify.eval.opus_competitor` on `redocred_dev` (150) + `challenge_dev_v1` (80). This gives Opus's dev F1,
  its unsupported rate, and the **categories where it fails** (our openings).
- **Deliverable:** `results/runs/opus55_dev_*.json`, plus a per-category table of the target to beat.
- **Without a key:** the plan still runs, but every "beat Opus" gate below becomes "unknown".

### Phase 2 — Student baseline by plain transfer learning (EXP-005)  · Kaggle T4 · ≈ 1–2 GPU-h (estimate)
- Qwen3-0.6B, LoRA r16, data-v1.1 (`configs/train_qwen3_06b_lora.yaml`; notebook as-is).
- **Gate S0:** dev micro F1 (pipeline B) beats the untuned baseline's CI upper bound on both dev sets. Otherwise debug
  before anything else.

### Phase 3 — Train the teacher (EXP-T1: Qwen3-4B; EXP-T2: Qwen3-8B only if needed)  · Kaggle T4 · QLoRA
- **Hypothesis:** a 4B model fine-tuned on the same data scores clearly higher than the 0.6B student.
- **Setup:**
  - 4-bit NF4 base + LoRA r16 (the code picks QLoRA automatically).
  - seq 1024, micro-batch 1, gradient checkpointing.
  - Checkpoints on the hub; resumable across Kaggle's 12-hour sessions.
- **Estimates (to be measured):** 4B ≈ 3–5 GPU-h; 8B ≈ 6–10 GPU-h. Kaggle allows ~30 GPU-h/week.
- **Gate T (critical):**
  1. Teacher dev F1 (B) ≥ Opus dev F1 on `redocred_dev`.
  2. Teacher dev F1 within 5 points of Opus on `challenge_dev_v1`.
  - If 4B fails → try 8B.
  - If 8B fails → **stop and re-plan** (more real data, a 14B teacher, or narrower claims).

### Phase 4 — Silver data: the teacher labels real business text (sequence-level distillation)
- **Problem it fixes:** ACQUIRED / INVESTED_IN / PARTNERED_WITH currently come from templates only (the biggest known weakness).
- **Unlabeled sources (licence-safe):**
  - English Wikipedia articles on companies and people (CC BY-SA). We train on them; if the silver set is ever
    redistributed, it must stay CC BY-SA.
  - SEC EDGAR 8-K filings and press-release exhibits (US public domain). They are dense in completed *and* planned
    acquisitions, investments and executive changes, i.e. real hard negatives.
  - Pre-filter: docs ≤ 12 sentences that contain business cue words. Keep 30% cue-free docs so the model still learns to abstain.
- **Labeling** (teacher on Kaggle; vLLM if it installs, otherwise batched `transformers` generate):
  1. Sample **5 outputs** per doc at temperature 0.7. Keep a triple only if **≥ 4/5** samples contain it (self-consistency).
  2. Pass the deterministic verifier (entity in text, valid evidence, ontology).
  3. A **second, different-family open teacher** (Qwen2.5-7B-Instruct fine-tuned the same way, Apache-2.0) must agree.
     Its errors are less correlated with the Qwen3 teacher's.
- **Size target:** 15–30k silver docs (≈ 5–10× the gold training set).
- **Human audit (pre-registered):** 200 random silver docs exported to a CSV/sheet; **you** mark each triple right or wrong.
  - **Gate Q:** estimated silver precision ≥ 0.90 (95% CI lower bound ≥ 0.85).
  - If it fails: raise the agreement threshold or drop the weakest source, then re-audit a fresh 200.
- **Compute (estimate):** 30k docs × 5 samples ≈ 15M generated tokens → 5–10 GPU-h on T4 with vLLM; more without.
  Uses both Kaggle T4s in parallel: one process per GPU, each on half the docs.

### Phase 5 — Distill into the student (EXP-D1, EXP-D2)
- **EXP-D1 — sequence-level KD:** student SFT on gold + silver (silver ≤ 3:1 vs gold per epoch; gold always included).
  - **Keep if:** dev F1 (B) improves over EXP-005 with paired-bootstrap P(not better) < 0.05 on at least one dev set, and does not drop on the other.
- **EXP-D2 — logit-level KD (white-box):**
  1. Precompute the teacher's **top-32 log-probabilities** at every target-token position (gold + silver). This runs
     offline, is resumable, and takes ≈ 0.2 GB. Exact alignment is possible because the tokenizers are identical.
  2. Student loss = α·CE(gold tokens) + (1−α)·T²·KL(teacher‖student) over the teacher's top-k, temperature T = 2.
     Grid: α ∈ {0.3, 0.5}.
  - **Keep if:** beats EXP-D1 under the same rule.
  - **Why it may help:** the teacher's probabilities carry its uncertainty between close labels (OWNS vs INVESTED_IN,
    abstain vs extract). Plain labels throw that away.
- **Code to add:**
  - `reverify/distill/topk.py`: precompute teacher top-k log-probabilities.
  - `sft.py`: KD loss option, with KD data loading alongside the existing collator.

### Phase 6 — Reliability: calibrated abstention + verifier choice
- **Confidence per relation:** the student's mean token log-probability over the relation label and argument tokens
  (llama.cpp returns log-probabilities).
- **Tune one threshold on dev:** maximize F1 subject to unsupported rate ≤ Opus's dev unsupported rate. Pick B vs B+
  (hedge filter) on dev.
- **Report:** expected calibration error before and after.
- **Gate R:** unsupported rate ≤ Opus's (dev) at no more than 1 point F1 cost.

### Phase 7 — Compression check (Gate 6)
- Export Q8_0, Q5_K_M, Q4_K_M and Q4_0, and measure each.
- **Ship** the smallest file that keeps ≥ 98% of the fp16 dev F1 **and** is ≤ 400 MB (Qwen3-0.6B Q4_0 = 381.6 MB, measured).
- Also record CPU latency, peak RAM and cold start.

### Phase 8 — The one frozen comparison (Gate 8)
- Run RE-Verify (final package, pipeline chosen on dev) and Opus 5.5 once each on `redocred_test` (500) and `challenge_v1` (100).
- Apply the claim rule in `docs/benchmark_protocol.md`: paired-bootstrap delta CI fully above 0, and unsupported rate
  not worse by > 0.02. Report the two sets **separately**, with every per-category loss included.
- Publish whatever the result is. If RE-Verify wins on Re-DocRED but not on the challenge set, the model card says exactly that.

### Phase 9 — Publish
Follow `docs/KAGGLE_COLAB_GUIDE.md` Part D. The model card is generated from the Phase 8 result files. Afterwards,
publish `challenge_v1.jsonl` (its sha256 is already public).

## 4. Budget and order (estimates; measured values replace them in EXPERIMENT_LOG.md)
| Week | Work | Kaggle GPU-h (est.) | You | Me |
|---|---|---|---|---|
| 1 | Phase 1 (dev) + Phase 2 | 2 | API key; run notebook | write `challenge_dev_v1` draft + review sheet; analyse |
| 1–2 | Phase 3 teacher 4B (+8B if needed) | 5–15 | run notebook | teacher config + labeling code |
| 2–3 | Phase 4 silver labeling + audit | 8–15 | audit 200 docs (~2 h) | corpus + labeling + audit tooling |
| 3–4 | Phase 5 D1, D2 | 4–8 | run notebook | KD code, comparisons |
| 4 | Phases 6–9 | 2 | final run, publish | calibration, final report |

## 5. What would make me stop early (and say so)
- Gate T fails with the 8B teacher: a 0.6B student will not beat Opus via this route.
- Gate Q fails twice: the silver data isn't trustworthy enough to train on.
- Phase 5 gains are inside noise: report EXP-005 as the release and skip the distillation claims.
