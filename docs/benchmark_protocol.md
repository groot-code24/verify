# RE-Verify benchmark protocol (v1.1, frozen 2026-11-20, before any test-set evaluation)

## Task
Input: a document as numbered sentences. Output: every relation from ontology v0.1 that the text **explicitly** states, as
`{"s": subject, "r": LABEL, "o": object, "ev": [sentence ids]}`. An empty list means abstain.

## Test sets (frozen — never used for training, prompt tuning or error analysis)
| Set | Size | Source | Integrity |
|---|---|---|---|
| `challenge_v1` | 100 items, 97 gold triples, 38 abstention items, 28 categories | Hand-written for this project (`benchmark/challenge_v1.jsonl`, **kept private until the comparison has run**, D-014) | sha256 `86efe21d72f123b08c6ecd6dca98484e9c4261a5d2c5657471c4ee44b773cfe5` |
| `redocred_test` | 500 docs | Re-DocRED test, mapped by ontology v0.1 | sha256 of processed file in `data/processed/DATASET_REPORT.json` |

Development uses only `redocred_dev` (499 docs) and `synthetic_dev` (300 docs; in-distribution with synthetic train by design).

## Scoring (`reverify/eval/metrics.py`, identical for every system)
- **Entity resolution (v1.1, D-015):** text is normalized (lowercase, punctuation removed, leading "the" dropped).
  - A predicted string matches an entity when it equals one of that entity's mentions.
  - Otherwise it matches by token-bounded containment: the prediction contains a mention and adds at most 3 tokens, or a prediction of 2+ tokens is contained in a mention. This only counts when exactly one entity matches.
  - Strict (exact-only) F1 is reported alongside as `micro_f1_strict_match`.
- **TP:** the resolved (s, r, o) is an unmatched explicit gold triple (either order for PARTNERED_WITH). Duplicates cannot score twice.
- **Ignored:** the prediction matches only an implicit gold triple (no textual support, D-008).
- **FP:** everything else, broken down into `reversed`, `wrong_relation`, `unresolved_entity` and `spurious`.
- **Reported metrics:** micro P/R/F1 (primary), macro F1 over relations with gold support, reversal-error rate, unsupported rate (FP / predictions), ungrounded rate (argument string not in text), and evidence sentence-F1 and exact match (TPs with gold evidence only). Also abstention accuracy (share of zero-gold docs with zero predictions), latency, tokens, parse rate and truncation rate.
- **Architectures scored from the same raw outputs:** A = raw; B = + deterministic verifier; B+ = + verifier + hedge-cue filter.
- **Statistics:** 95% bootstrap CI over documents (`bootstrap_ci`), and paired bootstrap over aligned documents for system comparisons (`paired_bootstrap`, reports P(A not better) and the CI of the delta).

## Competitive claim rule (decided before any competitor run)
RE-Verify is declared better than Opus 5.5 on a test set only if, on micro F1 under architecture B for both systems:
1. the paired-bootstrap delta CI95 lies entirely above 0 (1,000 resamples), **and**
2. RE-Verify's unsupported rate is not higher than Opus's by more than 0.02.

Both test sets are reported separately and never pooled. Per-category results on `challenge_v1` are reported in full, including categories where RE-Verify loses. With 100 items the challenge set can only detect large differences; a non-significant result is reported as "not demonstrated", never as a win.

## Known limitations
- `challenge_v1` was authored by Claude Opus 5.5 (the competitor). Shared labelling conventions likely favour Opus.
- Re-DocRED annotation conventions (which Wikidata facts count as "stated") differ from how a general model reads text. A fine-tuned model learns those conventions; Opus does not. A Re-DocRED win therefore partly measures convention-matching. This is why `challenge_v1` exists and is reported separately.
- Some ontology v0.1 business relations have small support in `redocred_test` (e.g. EXECUTIVE_OF 22, MANUFACTURER_OF 24 explicit triples), so their per-relation F1 has wide variance.
