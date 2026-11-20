"""Supervised fine-tuning (LoRA / QLoRA) for RE-Verify with resumable checkpoints.

python -m reverify.train.sft --config configs/train_pilot_cpu.yaml [--resume auto|<path>] [--stop-after-steps N]

Key properties
- Loss only on assistant tokens (the JSON target).
- Examples longer than seq_len are DROPPED (never truncated: a truncated target teaches broken JSON).
- Checkpoints include adapter, optimizer, scheduler, RNG and trainer state (HF Trainer), plus a
  MANIFEST.json with sha256 of every file. With sync_dir set, every checkpoint is mirrored there
  (e.g. a mounted Google Drive); --resume auto picks the newest *verified* checkpoint from
  output_dir or sync_dir.
- With hub_repo set (a PRIVATE Hugging Face model repo, token from HF_TOKEN), every checkpoint is also
  uploaded there. This is the only persistent store on Kaggle: /kaggle/working does not survive a crashed
  interactive session (FAILURES.md F-012). --resume auto checks local, sync_dir, then the hub.
- CUDA out-of-memory: micro_batch is halved (grad_accum doubled, so the effective batch is unchanged),
  then gradient checkpointing is enabled, and training resumes from the newest verified checkpoint.
  Each retry is appended to <output_dir>/oom_events.json.
- The best checkpoint by eval_loss is exported as best_adapter next to final_adapter.
- --stop-after-steps simulates a session interruption (used by the recovery test).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import yaml

from reverify.format import serialize_target, system_prompt, target_relations, user_prompt

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ data
def load_mix(cfg: dict, split: str) -> list[dict]:
    rng = random.Random(cfg.get("seed", 13))
    docs = []
    for name, n in cfg["data"][split].items():
        rows = [json.loads(l) for l in open(ROOT / "data" / "processed" / f"{name}.jsonl", encoding="utf-8")]
        if n != "all" and int(n) < len(rows):
            rows = rng.sample(rows, int(n))
        docs += rows
    rng.shuffle(docs)
    return docs


def build_example(tok, doc: dict, seq_len: int):
    msgs = [{"role": "system", "content": system_prompt(with_definitions=False)},
            {"role": "user", "content": user_prompt(doc["sents"])}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    target = serialize_target(target_relations(doc)) + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(t_ids) > seq_len:
        return None
    return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ------------------------------------------------------------------ checkpoints
def write_manifest(ckpt: Path, extra: dict) -> None:
    files = {}
    for p in sorted(ckpt.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files[str(p.relative_to(ckpt)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    (ckpt / "MANIFEST.json").write_text(json.dumps({"files": files, **extra}, indent=1))


def verify_checkpoint(ckpt: Path) -> bool:
    m = ckpt / "MANIFEST.json"
    if not m.exists() or not (ckpt / "trainer_state.json").exists():
        return False
    man = json.loads(m.read_text())
    for rel, h in man["files"].items():
        p = ckpt / rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            return False
    return True


def find_latest_checkpoint(*dirs) -> Path | None:
    cands = []
    for d in dirs:
        if d and Path(d).exists():
            cands += [p for p in Path(d).glob("checkpoint-*") if p.is_dir()]
    cands.sort(key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for c in cands:
        if verify_checkpoint(c):
            return c
        print(f"[resume] skipping unverified/corrupt checkpoint {c}")
    return None


def _ckpt_names(files: list[str], prefix: str) -> list[str]:
    """checkpoint-N directory names under <prefix>/ in a hub file listing, newest first."""
    names = {f.split("/")[1] for f in files if f.startswith(f"{prefix}/checkpoint-") and f.count("/") >= 2}
    return sorted(names, key=lambda n: int(n.split("-")[-1]), reverse=True)


def hub_push(ckpt: Path, repo: str, prefix: str, keep: int = 3) -> None:
    """Upload to <repo>/<prefix>/<checkpoint-N>. The prefix (experiment id) keeps experiments that share one
    private repo apart; without it EXP-005 would resume from EXP-004's checkpoint-100 (FAILURES.md F-019)."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt), path_in_repo=f"{prefix}/{ckpt.name}", repo_id=repo,
                      commit_message=f"{prefix} {ckpt.name}")
    for old in _ckpt_names(api.list_repo_files(repo), prefix)[keep:]:
        api.delete_folder(f"{prefix}/{old}", repo_id=repo, commit_message=f"prune {prefix}/{old}")


def hub_pull_latest(repo: str, prefix: str, staging: Path) -> Path | None:
    from huggingface_hub import HfApi, snapshot_download
    try:
        files = HfApi().list_repo_files(repo)
    except Exception as e:  # repo does not exist yet / no access
        print(f"[resume] hub repo {repo} not readable: {type(e).__name__}")
        return None
    for n in _ckpt_names(files, prefix):
        snapshot_download(repo, allow_patterns=[f"{prefix}/{n}/*"], local_dir=str(staging))
        c = staging / prefix / n
        if verify_checkpoint(c):
            return c
        print(f"[resume] hub checkpoint {prefix}/{n} failed verification")
    return None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="'auto' or checkpoint path")
    ap.add_argument("--stop-after-steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                              set_seed)

    from reverify.hardware import adapt, detect

    set_seed(cfg.get("seed", 13))
    hw = detect()
    auto = adapt(hw, cfg["model_params_b"], cfg.get("seq_len", 1024))
    tcfg = {**auto, **{k: v for k, v in cfg.get("overrides", {}).items()}}
    print("[hw]", hw, "\n[train cfg]", tcfg, flush=True)

    out_dir = ROOT / cfg["output_dir"]
    sync_dir = Path(cfg["sync_dir"]) if cfg.get("sync_dir") else None
    hub_repo = cfg.get("hub_repo")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if hw.get("n_gpus", 1) > 1:
        print("[warn] >1 GPU visible: HF Trainer would use DataParallel and change the effective batch. "
              "Set CUDA_VISIBLE_DEVICES=0 (the notebook does).", flush=True)
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtypes[tcfg["precision"]]
    load_kw = {"dtype": dtypes[tcfg.get("load_dtype", tcfg["precision"])]}
    if tcfg["quant_4bit"]:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **load_kw)
    if tcfg["quant_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tcfg["grad_checkpointing"])
    elif tcfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def encode(split):
        docs = load_mix(cfg, split)
        ex = [build_example(tok, d, tcfg["seq_len"]) for d in docs]
        kept = [e for e in ex if e]
        print(f"[data] {split}: {len(kept)}/{len(ex)} examples fit seq_len={tcfg['seq_len']}", flush=True)
        return kept

    train_ds, eval_ds = encode("train"), encode("eval")

    class SyncAndStop(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt.exists():
                write_manifest(ckpt, {"global_step": state.global_step, "experiment": cfg["experiment_id"],
                                      "data_version": cfg.get("data_version"), "config": cfg,
                                      "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                if sync_dir:
                    dst = sync_dir / ckpt.name
                    shutil.copytree(ckpt, dst, dirs_exist_ok=True)
                    print(f"[sync] mirrored {ckpt.name} -> {dst}", flush=True)
                if hub_repo:
                    hub_push(ckpt, hub_repo, cfg["experiment_id"])
                    print(f"[sync] uploaded {ckpt.name} -> hf.co/{hub_repo}", flush=True)
            return control

        def on_step_end(self, args, state, control, **kw):
            if a.stop_after_steps and state.global_step >= a.stop_after_steps:
                print(f"[interrupt] simulated session end at step {state.global_step}", flush=True)
                control.should_save = True
                control.should_training_stop = True
            return control

    def make_trainer():
        targs = TrainingArguments(
            output_dir=str(out_dir), per_device_train_batch_size=tcfg["micro_batch"],
            per_device_eval_batch_size=tcfg["micro_batch"], gradient_accumulation_steps=tcfg["grad_accum"],
            learning_rate=cfg["lr"], num_train_epochs=cfg.get("epochs", 1), max_steps=cfg.get("max_steps", -1),
            lr_scheduler_type="cosine", warmup_steps=cfg.get("warmup_steps", 0),
            logging_steps=cfg.get("logging_steps", 10), eval_strategy="steps", eval_steps=cfg["eval_steps"],
            save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=cfg.get("save_total_limit", 3),
            metric_for_best_model="eval_loss", greater_is_better=False, load_best_model_at_end=False,
            bf16=tcfg["precision"] == "bf16", fp16=tcfg["precision"] == "fp16" and hw["cuda"],
            dataloader_num_workers=tcfg["workers"], report_to=[], seed=cfg.get("seed", 13), use_cpu=not hw["cuda"],
            optim="paged_adamw_8bit" if tcfg["quant_4bit"] else "adamw_torch", remove_unused_columns=False)
        return Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                       data_collator=Collator(tok.pad_token_id), callbacks=[SyncAndStop()])

    def latest():
        r = find_latest_checkpoint(out_dir, sync_dir)
        if r is None and hub_repo:
            r = hub_pull_latest(hub_repo, cfg["experiment_id"], out_dir / "_hub_staging")
        if r is not None and r.parent != out_dir:
            local = out_dir / r.name
            shutil.copytree(r, local, dirs_exist_ok=True)
            r = local
        return r

    resume = None
    if a.resume == "auto":
        resume = latest()
        print("[resume] " + (f"from {resume}" if resume else "no valid checkpoint, starting fresh"), flush=True)
    elif a.resume:
        resume = Path(a.resume)
        if not verify_checkpoint(resume):
            raise SystemExit(f"checkpoint {resume} failed integrity verification")
    t0 = time.time()
    oom_log = out_dir / "oom_events.json"
    for attempt in range(5):
        trainer = make_trainer()
        try:
            result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
            break
        except torch.cuda.OutOfMemoryError as e:
            event = {"attempt": attempt, "micro_batch": tcfg["micro_batch"], "grad_accum": tcfg["grad_accum"],
                     "grad_checkpointing": tcfg["grad_checkpointing"], "step": trainer.state.global_step,
                     "error": str(e)[:300], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            del trainer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if tcfg["micro_batch"] > 1:
                tcfg["micro_batch"] //= 2
                tcfg["grad_accum"] *= 2
            elif not tcfg["grad_checkpointing"]:
                tcfg["grad_checkpointing"] = True
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
            else:
                raise
            event["next"] = {k: tcfg[k] for k in ("micro_batch", "grad_accum", "grad_checkpointing")}
            events = json.loads(oom_log.read_text()) if oom_log.exists() else []
            oom_log.write_text(json.dumps(events + [event], indent=1))
            print(f"[oom] {event}", flush=True)
            resume = latest()
    else:
        raise SystemExit("out of memory after 5 attempts")
    metrics = {**result.metrics, "wall_s": round(time.time() - t0, 1)}
    metrics.update(trainer.evaluate())
    if not a.stop_after_steps:
        exports = {"final_adapter": out_dir / "final_adapter"}
        model.save_pretrained(exports["final_adapter"])
        tok.save_pretrained(exports["final_adapter"])
        best = trainer.state.best_model_checkpoint
        if best and Path(best).exists():
            b = out_dir / "best_adapter"
            b.mkdir(exist_ok=True)
            for f in Path(best).glob("adapter_*"):
                shutil.copy(f, b / f.name)
            tok.save_pretrained(b)
            exports["best_adapter"] = b
            metrics["best_checkpoint"] = Path(best).name
            metrics["best_eval_loss"] = trainer.state.best_metric
        for name, path in exports.items():
            if sync_dir:
                shutil.copytree(path, sync_dir / name, dirs_exist_ok=True)
            if hub_repo:
                from huggingface_hub import HfApi
                HfApi().upload_folder(folder_path=str(path), path_in_repo=f"{cfg['experiment_id']}/{name}", repo_id=hub_repo,
                                      commit_message=f"export {name}")
    log = {"experiment_id": cfg["experiment_id"], "metrics": metrics, "log_history": trainer.state.log_history,
           "hw": hw, "train_cfg": tcfg, "resumed_from": str(resume) if resume else None}
    (out_dir / f"run_{int(time.time())}.json").write_text(json.dumps(log, indent=1))
    print("[done]", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
