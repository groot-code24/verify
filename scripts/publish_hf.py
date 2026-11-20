"""Publish a trained RE-Verify model (and optionally the demo Space) to the Hugging Face Hub.

python scripts/publish_hf.py --model-repo USER/re-verify-qwen25-0.5b \
    --gguf exports/EXP-004.../reverify-q4_0.gguf exports/EXP-004.../reverify-q8_0.gguf \
    --adapter checkpoints/qwen25_05b_lora/best_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --runs EXP-004-qwen25-05b-lora-r16_q4_0_dev150 EXP-004-qwen25-05b-lora-r16_q4_0_syndev100 \
    --github-url https://github.com/USER/re-verify --space-repo USER/re-verify-demo [--private] [--dry-run]

The model card is generated ONLY from results/runs/<run>.json files passed with --runs. A comparison with
Claude Opus 5.5 appears only if --competitor-runs is given; otherwise the card states it was not run.
Token: HF_TOKEN env var (or `hf auth login`) with write access.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reverify.ontology import load_ontology  # noqa: E402


def fmt_runs(run_ids):
    rows = []
    for rid in run_ids:
        r = json.loads((ROOT / "results" / "runs" / f"{rid}.json").read_text())
        m, e = r["meta"], r["efficiency"]
        for arch in ("A_raw", "B_verifier", "B+_verifier_hedge"):
            x = r[arch]
            lo, hi = x["micro_f1_ci95"]
            rows.append(f"| {m['data']} (n={m['n_docs']}) | {arch} | {x['precision']:.3f} | {x['recall']:.3f} | "
                        f"**{x['micro_f1']:.3f}** [{lo:.3f}, {hi:.3f}] | {x['macro_f1']:.3f} | {x['unsupported_rate']:.3f} | "
                        f"{x['evidence_f1'] if x['evidence_f1'] is not None else '-'} | "
                        f"{x['abstention_accuracy'] if x['abstention_accuracy'] is not None else '-'} | "
                        f"{e['mean_latency_s']} s |")
    head = ("| Data | Pipeline | P | R | Micro F1 [95% CI] | Macro F1 | Unsupported rate | Evidence F1 | "
            "Abstention acc. | Mean CPU latency |\n|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows) if rows else "_No evaluation runs supplied._"


def model_card(a, gguf_sizes):
    onto = load_ontology()
    files = "\n".join(f"| `{n}` | {s:.1f} MB |" for n, s in gguf_sizes.items())
    comp = (fmt_runs(a.competitor_runs) if a.competitor_runs else
            "**Not run.** No comparison with Claude Opus 5.5 has been performed for this release, so no claim about "
            "relative performance is made. The frozen protocol and runner are in the GitHub repository "
            "(`docs/benchmark_protocol.md`, `reverify/eval/opus_competitor.py`).")
    think_suffix = ' + "<think>\\n\\n</think>\\n\\n"' if "Qwen3" in a.base_model else ""
    return f"""---
license: apache-2.0
base_model: {a.base_model}
library_name: gguf
pipeline_tag: text-generation
language: [en]
tags: [relation-extraction, information-extraction, evidence, gguf, llama-cpp, lora, cpu]
---

# RE-Verify — {a.model_repo.split('/')[-1]}

A compact model for **evidence-grounded relation extraction**, fine-tuned (LoRA) from
[`{a.base_model}`](https://huggingface.co/{a.base_model}) and quantized to GGUF for offline CPU inference with llama.cpp.

Given text, it returns only the relations that the text **explicitly** states, from a fixed ontology of
{len(onto.names)} relations, each with the sentence(s) that support it. It is trained to leave out speculative,
planned, negated and merely implied relations, and to return an empty list when nothing qualifies.

**Status: research prototype.** Read *Evaluation* and *Limitations* before use.

## Files
| File | Size |
|---|---|
{files}
| `adapter/` | LoRA adapter (PEFT) for `{a.base_model}` |

## Usage (CPU, no GPU needed)
```python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from reverify.format import split_sentences, system_prompt, user_prompt, parse_output, to_public_json
from reverify.verifier import verify      # pip install "re-verify @ git+{a.github_url}"

llm = Llama(hf_hub_download("{a.model_repo}", "{a.primary_gguf}"), n_ctx=4096, verbose=False)
sents = split_sentences("Elon Musk founded SpaceX in 2002. SpaceX developed Starship.")
prompt = ("<|im_start|>system\\n" + system_prompt(with_definitions=False) + "<|im_end|>\\n<|im_start|>user\\n"
          + user_prompt(sents) + "<|im_end|>\\n<|im_start|>assistant\\n"{think_suffix})
raw = llm.create_completion(prompt, max_tokens=512, temperature=0, stop=["<|im_end|>"])["choices"][0]["text"]
rels, _ = parse_output(raw)
print(to_public_json(verify(rels, sents).kept, sents))
```
The prompt format must match training exactly. Use the helpers from the `reverify` package rather than
writing your own prompt.

## Relations (ontology v{onto.version})
{", ".join(f"`{n}`" for n in onto.names)}. Each relation has one canonical direction (e.g. `ACQUIRED` is
acquirer → acquired). Definitions: `reverify/resources/ontology_v{onto.version}.yaml` in the GitHub repository.

## Evaluation
Scored with the project's single scorer (`reverify/eval/metrics.py`; lenient unique entity matching, protocol v1.1).
Pipelines: **A** = raw model output, **B** = + deterministic verifier, **B+** = + verifier + hedge-cue filter.
95% CIs are document-level bootstrap. Measured on: {a.hardware}.

{fmt_runs(a.runs)}

### Comparison with Claude Opus 5.5
{comp}

## Training data
- [Re-DocRED](https://huggingface.co/datasets/tonytan48/Re-DocRED) train (MIT), mapped to the ontology. Documents
  describing acquisitions, investments or partnerships are removed, because DocRED's schema cannot label them.
- About 3,000 template-generated documents with correct-by-construction labels: positives for relations DocRED lacks
  (`ACQUIRED`, `INVESTED_IN`, `PARTNERED_WITH`) plus hard negatives (speculation, negation, failed deals, future
  events, co-mentions).
- Full pipeline, hashes and leakage reports: {a.github_url}

## Limitations
- English only; trained on Wikipedia-style and template text. Expect lower accuracy on other domains.
- `ACQUIRED`, `INVESTED_IN` and `PARTNERED_WITH` were learned from templates only.
- No current-vs-former distinction: `EMPLOYEE_OF` / `EXECUTIVE_OF` mean "is or was".
- It can still output unsupported relations. Keep the verifier on and check the evidence before relying on a result.
- Not a fact checker: it extracts what the text says, not whether that is true.

## License
Apache-2.0 (same as the base model). Re-DocRED is MIT-licensed.
"""


def render(src: Path, dst: Path, subs: dict):
    text = src.read_text(encoding="utf-8")
    for k, v in subs.items():
        text = text.replace("{{" + k + "}}", v)
    dst.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--gguf", nargs="+", required=True, help="first file is the default for the demo")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--competitor-runs", nargs="*", default=[])
    ap.add_argument("--github-url", required=True)
    ap.add_argument("--git-ref", default="main")
    ap.add_argument("--space-repo", default=None)
    ap.add_argument("--hardware", default="the CPU recorded in each run's meta")
    ap.add_argument("--private", action="store_true", help="create repos private (recommended for a first upload)")
    ap.add_argument("--dry-run", action="store_true", help="render into exports/publish_preview/ without uploading")
    a = ap.parse_args()
    ggufs = [Path(g) for g in a.gguf]
    a.primary_gguf = ggufs[0].name
    sizes = {g.name: g.stat().st_size / 1e6 for g in ggufs}

    stage = ROOT / "exports" / "publish_preview"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "space").mkdir(parents=True)
    (stage / "README.md").write_text(model_card(a, sizes), encoding="utf-8")
    subs = {"MODEL_REPO": a.model_repo, "GGUF_FILE": a.primary_gguf, "GITHUB_URL": a.github_url,
            "GIT_REF": a.git_ref, "NO_THINK": "1" if "Qwen3" in a.base_model else "0"}
    for f in ("app.py", "requirements.txt", "README.md"):
        render(ROOT / "space" / f, stage / "space" / f, subs)
    print(f"rendered model card + Space files in {stage}")
    if a.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.model_repo, repo_type="model", private=a.private, exist_ok=True)
    for g in ggufs:
        api.upload_file(path_or_fileobj=str(g), path_in_repo=g.name, repo_id=a.model_repo, commit_message=f"add {g.name}")
    if a.adapter:
        api.upload_folder(folder_path=a.adapter, path_in_repo="adapter", repo_id=a.model_repo, commit_message="add LoRA adapter")
    for rid in a.runs + a.competitor_runs:
        api.upload_file(path_or_fileobj=str(ROOT / "results" / "runs" / f"{rid}.json"), path_in_repo=f"eval/{rid}.json",
                        repo_id=a.model_repo, commit_message=f"eval {rid}")
    api.upload_file(path_or_fileobj=str(stage / "README.md"), path_in_repo="README.md", repo_id=a.model_repo,
                    commit_message="model card")
    print(f"model: https://huggingface.co/{a.model_repo}")
    if a.space_repo:
        api.create_repo(a.space_repo, repo_type="space", space_sdk="gradio", private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(stage / "space"), repo_id=a.space_repo, repo_type="space",
                          commit_message="RE-Verify demo")
        print(f"space: https://huggingface.co/spaces/{a.space_repo}")


if __name__ == "__main__":
    main()
