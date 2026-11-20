"""Hub checkpoint sync against an in-memory fake of huggingface_hub (no network)."""
import fnmatch
import shutil
from pathlib import Path

import huggingface_hub
import pytest

from reverify.train import sft
from reverify.train.sft import hub_pull_latest, hub_push, write_manifest


class FakeHub:
    def __init__(self, root: Path):
        self.root = root

    # --- HfApi surface used by sft.py
    def create_repo(self, repo, private=True, exist_ok=True, **kw):
        (self.root / repo).mkdir(parents=True, exist_ok=True)

    def upload_folder(self, folder_path, path_in_repo, repo_id, **kw):
        shutil.copytree(folder_path, self.root / repo_id / path_in_repo, dirs_exist_ok=True)

    def list_repo_files(self, repo):
        base = self.root / repo
        if not base.exists():
            raise FileNotFoundError(repo)
        return [str(p.relative_to(base)).replace("\\", "/") for p in base.rglob("*") if p.is_file()]

    def delete_folder(self, path, repo_id, **kw):
        shutil.rmtree(self.root / repo_id / path)

    # --- snapshot_download
    def snapshot_download(self, repo, allow_patterns, local_dir):
        for f in self.list_repo_files(repo):
            if any(fnmatch.fnmatch(f, pat) for pat in allow_patterns):
                dst = Path(local_dir) / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.root / repo / f, dst)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path / "hub")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    return fake


def _ckpt(root: Path, step: int, payload: bytes = b"w") -> Path:
    c = root / f"checkpoint-{step}"
    c.mkdir(parents=True)
    (c / "adapter_model.safetensors").write_bytes(payload)
    (c / "trainer_state.json").write_text("{}")
    write_manifest(c, {"global_step": step})
    return c


def test_experiments_sharing_a_repo_stay_isolated(hub, tmp_path):
    for step in (100, 200):
        hub_push(_ckpt(tmp_path / "a", step, b"exp4"), "u/ckpt", "EXP-004")
    hub_push(_ckpt(tmp_path / "b", 100, b"exp5"), "u/ckpt", "EXP-005")
    got = hub_pull_latest("u/ckpt", "EXP-005", tmp_path / "stage")
    assert got.name == "checkpoint-100" and (got / "adapter_model.safetensors").read_bytes() == b"exp5"
    assert hub_pull_latest("u/ckpt", "EXP-004", tmp_path / "stage").name == "checkpoint-200"
    assert hub_pull_latest("u/ckpt", "EXP-006", tmp_path / "stage") is None


def test_prune_keeps_newest(hub, tmp_path):
    for step in (100, 200, 300, 400):
        hub_push(_ckpt(tmp_path / "a", step), "u/ckpt", "EXP-004", keep=2)
    names = sft._ckpt_names(hub.list_repo_files("u/ckpt"), "EXP-004")
    assert names == ["checkpoint-400", "checkpoint-300"]


def test_corrupt_hub_checkpoint_is_skipped(hub, tmp_path):
    hub_push(_ckpt(tmp_path / "a", 100), "u/ckpt", "EXP-004")
    hub_push(_ckpt(tmp_path / "a", 200), "u/ckpt", "EXP-004")
    (hub.root / "u/ckpt/EXP-004/checkpoint-200/adapter_model.safetensors").write_bytes(b"partial upload")
    assert hub_pull_latest("u/ckpt", "EXP-004", tmp_path / "stage").name == "checkpoint-100"


def test_missing_repo_returns_none(hub, tmp_path):
    assert hub_pull_latest("u/does-not-exist", "EXP-004", tmp_path / "stage") is None
