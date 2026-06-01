import json
import os

from voice_assistant_web.backend.app.main import _scan_rollout_tree


def _write_key_region(path, *, key_region_id, start_time, end_time):
    path.mkdir(parents=True)
    (path / "cam_high.mp4").write_bytes(b"mp4")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": key_region_id,
                "phase": "warmup",
                "reward": 0,
                "start_time": start_time,
                "end_time": end_time,
                "num_replay_transitions": 1,
            }
        )
    )
    os.utime(path, (end_time, end_time))


def test_key_region_directories_are_sorted_newest_first(tmp_path):
    old = tmp_path / "key_region_zz_old"
    new = tmp_path / "key_region_aa_new"
    _write_key_region(old, key_region_id="zz_old", start_time=10.0, end_time=20.0)
    _write_key_region(new, key_region_id="aa_new", start_time=30.0, end_time=40.0)

    tree = _scan_rollout_tree(tmp_path, "key_regions/task/day/warmup")

    names = [child["name"] for child in tree["children"] if child["type"] == "directory"]
    assert names[:2] == ["key_region_aa_new", "key_region_zz_old"]
