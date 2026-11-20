#!/usr/bin/env bash
# Gate 2 baselines: untuned base models, one-shot prompt, no grammar (grammar measured 4x slower).
set -e
PY=.venv/Scripts/python
for data in redocred_dev synthetic_dev; do
  $PY -m reverify.inference.llamacpp_runner --model models/qwen2.5-0.5b-instruct-q4_0.gguf --data $data --limit 40 \
      --run-id base_qwen25-05b-q4_0_oneshot_${data}40 --one-shot --no-grammar --max-tokens 384
  $PY -m reverify.inference.llamacpp_runner --model models/Qwen3-0.6B-Q8_0.gguf --data $data --limit 40 \
      --run-id base_qwen3-06b-q8_oneshot_${data}40 --one-shot --no-grammar --no-think --max-tokens 384
done
