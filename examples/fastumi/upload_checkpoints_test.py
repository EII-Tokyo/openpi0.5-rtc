from pathlib import Path

from examples.fastumi.upload_checkpoints import pending_checkpoints


def test_only_completed_selected_checkpoints_are_uploaded(tmp_path: Path):
    for step in ("1000", "10000", "20000", "29999", "30000.orbax-checkpoint-tmp-0"):
        for name in ("params", "assets", "train_state"):
            (tmp_path / step / name).mkdir(parents=True)
    # A numeric folder without all checkpoint items is not ready.
    (tmp_path / "40000" / "params").mkdir(parents=True)
    pending = pending_checkpoints(tmp_path, {10000}, interval=10000, final_step=29999)
    assert [path.name for path in pending] == ["20000", "29999"]
