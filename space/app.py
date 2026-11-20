"""RE-Verify demo (Hugging Face Space, CPU).

Runs the SAME pipeline as the evaluation: reverify.format prompt -> llama.cpp GGUF -> parse_output ->
deterministic verifier -> public JSON with exact evidence sentences.
Configuration via Space variables: MODEL_REPO, GGUF_FILE, NO_THINK ("1" for Qwen3 bases), N_THREADS.
"""
import json
import os
import time

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from reverify.format import parse_output, split_sentences, system_prompt, to_public_json, user_prompt
from reverify.verifier import verify

MODEL_REPO = os.environ.get("MODEL_REPO", "{{MODEL_REPO}}")
GGUF_FILE = os.environ.get("GGUF_FILE", "{{GGUF_FILE}}")
NO_THINK = os.environ.get("NO_THINK", "{{NO_THINK}}") == "1"
N_THREADS = int(os.environ.get("N_THREADS", "2"))

t0 = time.perf_counter()
MODEL_PATH = hf_hub_download(MODEL_REPO, GGUF_FILE)
LLM = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=N_THREADS, verbose=False, seed=0)
LOAD_S = time.perf_counter() - t0
SYSTEM = system_prompt(with_definitions=False)
MAX_SENTS = 40

EXAMPLES = [
    "Halvorsen Robotics was founded by Ingrid Halvorsen in 2011. The company is headquartered in Oslo. "
    "Last year it acquired the drone maker Skyfin.",
    "Nexaline Corp is considering purchasing the lender Fairbridge Credit. Analysts expect a decision in May.",
    "Tidepool Health raised $20 million in a round led by Orchid Ventures. Orchid Ventures does not own Tidepool Health.",
]


def prompt(sents):
    p = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user_prompt(sents)}<|im_end|>\n<|im_start|>assistant\n"
    return p + ("<think>\n\n</think>\n\n" if NO_THINK else "")


def extract(text: str, hedge_filter: bool):
    sents = split_sentences(text)[:MAX_SENTS]
    if not sents:
        return {"relations": [], "abstain": True}, "", ""
    t = time.perf_counter()
    out = LLM.create_completion(prompt(sents), max_tokens=512, temperature=0.0, top_k=1, stop=["<|im_end|>"])
    dt = time.perf_counter() - t
    raw = out["choices"][0]["text"]
    rels, _ = parse_output(raw)
    res = verify(rels, sents, hedge_filter=hedge_filter)
    public = to_public_json(res.kept, sents)
    dropped = "\n".join(f"- {r.get('s')} {r.get('r')} {r.get('o')}  ->  dropped: {why}" for r, why in res.dropped) or "(none)"
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    info = (f"{len(sents)} sentences | {dt:.1f} s | {out['usage']['completion_tokens']} output tokens | "
            f"model load {LOAD_S:.1f} s\n\nSentences as the model saw them:\n{numbered}\n\nVerifier drops:\n{dropped}")
    return public, raw, info


with gr.Blocks(title="RE-Verify") as demo:
    gr.Markdown(
        "# RE-Verify\nEvidence-grounded relation extraction with a compact model on CPU. "
        "Only relations the text **explicitly** states are returned, each with its evidence sentence. "
        "Speculative, planned, negated or implied relations should be omitted. "
        f"Model: `{MODEL_REPO}/{GGUF_FILE}`. Research prototype — see the model card for measured accuracy and limits.")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Text", lines=8, placeholder="Paste a paragraph (English).")
            hedge = gr.Checkbox(label="Drop relations whose only evidence contains hedge/negation cues", value=False)
            btn = gr.Button("Extract", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            out_json = gr.JSON(label="Verified relations")
            with gr.Accordion("Details", open=False):
                out_info = gr.Textbox(label="Run info", lines=10)
                out_raw = gr.Textbox(label="Raw model output", lines=4)
    btn.click(extract, [inp, hedge], [out_json, out_raw, out_info])

if __name__ == "__main__":
    demo.launch()
