---
title: RE-Verify
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Evidence-grounded relation extraction on CPU
models:
  - {{MODEL_REPO}}
---

# RE-Verify demo

Runs `{{MODEL_REPO}}` (`{{GGUF_FILE}}`) with llama.cpp on the free CPU hardware.
Code: {{GITHUB_URL}}. Model card, measured results and limitations: https://huggingface.co/{{MODEL_REPO}}
