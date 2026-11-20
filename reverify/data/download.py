"""Download raw datasets from the Hugging Face Hub and record SHA-256 hashes.

Usage:  python -m reverify.data.download [--out data/raw]

Every file is recorded in data/raw/MANIFEST.json with its source repo, revision
(commit sha) and sha256 so later runs can verify integrity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SOURCES = [
    # (repo_id, filename, local_name)
    ("tonytan48/Re-DocRED", "train_revised.json", "redocred_train.json"),
    ("tonytan48/Re-DocRED", "dev_revised.json", "redocred_dev.json"),
    ("tonytan48/Re-DocRED", "test_revised.json", "redocred_test.json"),
    ("thunlp/docred", "data/rel_info.json.gz", "docred_rel_info.json.gz"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = {}
    for repo, fname, local in SOURCES:
        rev = api.dataset_info(repo).sha
        p = Path(hf_hub_download(repo, fname, repo_type="dataset", revision=rev))
        dst = out / local
        dst.write_bytes(p.read_bytes())
        manifest[local] = {"repo": repo, "file": fname, "revision": rev, "sha256": sha256(dst),
                           "bytes": dst.stat().st_size}
        print(f"{local}: {manifest[local]['bytes']} bytes  sha256={manifest[local]['sha256'][:16]}")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify(out: Path) -> bool:
    manifest = json.loads((out / "MANIFEST.json").read_text())
    ok = True
    for local, meta in manifest.items():
        actual = sha256(out / local)
        if actual != meta["sha256"]:
            print(f"INTEGRITY FAIL {local}")
            ok = False
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if a.verify_only:
        raise SystemExit(0 if verify(Path(a.out)) else 1)
    download(Path(a.out))
    print("verify:", verify(Path(a.out)))
