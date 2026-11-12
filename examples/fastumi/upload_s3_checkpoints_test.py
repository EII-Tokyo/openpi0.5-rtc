import hashlib
import json
from pathlib import Path

import pytest

from examples.fastumi import upload_s3_checkpoints as uploader


def test_selection_excludes_incomplete_and_unselected_checkpoints(tmp_path):
    for step in ("1000", "10000", "20000", "29999", "10000.orbax-checkpoint-tmp-0", "40000"):
        for item in ("params", "assets", "train_state"):
            (tmp_path / step / item).mkdir(parents=True)
    (tmp_path / "30000" / "params").mkdir(parents=True)
    assert [p.name for p in uploader.pending_checkpoints(tmp_path, {10000}, 10000, 29999)] == ["20000", "29999"]


@pytest.mark.parametrize("full", [False, True])
def test_backup_contents_and_completion_marker(tmp_path, monkeypatch, full):
    checkpoint = tmp_path / "29999"
    for item in ("params", "assets", "train_state"):
        (checkpoint / item).mkdir(parents=True)
        (checkpoint / item / "data").write_bytes(item.encode())
    objects = {}

    def fake_aws(*args):
        if args[:2] == ("s3", "cp"):
            contents = Path(args[2]).read_bytes()
            objects[args[3]] = contents
            return ""
        key = args[args.index("--key") + 1]
        contents = objects[f"s3://bucket/{key}"]
        return json.dumps(
            {"ContentLength": len(contents), "Metadata": {"sha256": hashlib.sha256(contents).hexdigest()}}
        )

    monkeypatch.setattr(uploader, "aws", fake_aws)
    uploader.upload_checkpoint(checkpoint, "s3://bucket/run", full=full)
    manifest = json.loads(objects["s3://bucket/run/29999/checkpoint_manifest.json"])
    assert manifest["full_training_state"] is full
    assert {f["path"] for f in manifest["files"]} == (
        {"params/data", "assets/data", "train_state/data"} if full else {"params/data", "assets/data"}
    )
    assert list(objects)[-1].endswith("checkpoint_manifest.json")


def test_mismatch_does_not_publish_completion_marker(tmp_path, monkeypatch):
    checkpoint = tmp_path / "29999"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "data").write_bytes(b"parameters")
    calls = []

    def fake_aws(*args):
        calls.append(args)
        return json.dumps({"ContentLength": 0, "Metadata": {}})

    monkeypatch.setattr(uploader, "aws", fake_aws)
    with pytest.raises(RuntimeError, match="mismatch"):
        uploader.upload_checkpoint(checkpoint, "s3://bucket/run", full=False)
    assert not any("checkpoint_manifest.json" in str(call) for call in calls)
