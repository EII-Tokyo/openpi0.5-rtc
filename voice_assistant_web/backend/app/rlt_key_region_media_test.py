import json
import os
import subprocess
import tempfile

from fastapi import HTTPException
import pytest

os.environ.setdefault("RLT_SEGMENT_DB_PATH", str((tempfile.gettempdir()) + "/rlt_key_region_media_test.sqlite3"))

from voice_assistant_web.backend.app import main


class _EmptyRLTControl:
    def list_segments(self, *, limit=500):
        return []


def _write_test_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5:duration=1",
            "-frames:v",
            "5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _write_key_region(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    key_region_id = "media"
    rollout_dir = rollout_root / "key_regions/task/2026-06-24/rl" / f"key_region_{key_region_id}"
    _write_test_video(rollout_dir / "cam_low.mp4")
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": key_region_id,
                "phase": "rl",
                "reward": 1,
                "start_time": 10.0,
                "end_time": 11.0,
                "score_time": 12.0,
                "duration_seconds": 1.0,
                "fps": 5.0,
                "num_frames": 5,
                "num_replay_transitions": 1,
                "segment_status": "committed",
                "train_eligible": True,
            }
        )
    )
    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "KEY_REGION_FRAME_CACHE_ROOT", tmp_path / "frame_cache")
    monkeypatch.setattr(main, "rlt_control", _EmptyRLTControl())
    return key_region_id


def test_key_region_media_metadata_reports_camera_frame_urls(tmp_path, monkeypatch):
    key_region_id = _write_key_region(tmp_path, monkeypatch)

    payload = main.rlt_key_region_media_metadata(key_region_id)

    assert payload["key_region_id"] == key_region_id
    assert payload["fps"] == 5.0
    assert payload["frame_count"] == 5
    assert payload["duration_seconds"] == 1.0
    assert payload["cameras"] == [
        {
            "camera": "cam_low",
            "frame_url": f"/api/rlt/key-region/{key_region_id}/frame?camera=cam_low&frame={{frame}}",
            "video_path": "key_regions/task/2026-06-24/rl/key_region_media/cam_low.mp4",
        }
    ]


def test_key_region_frame_endpoint_returns_cached_jpeg(tmp_path, monkeypatch):
    key_region_id = _write_key_region(tmp_path, monkeypatch)

    response = main.rlt_key_region_frame(key_region_id, camera="cam_low", frame=2)

    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert response.body.startswith(b"\xff\xd8")
    assert list((tmp_path / "frame_cache").glob("*.jpg"))


def test_key_region_frame_endpoint_rejects_unknown_camera(tmp_path, monkeypatch):
    key_region_id = _write_key_region(tmp_path, monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        main.rlt_key_region_frame(key_region_id, camera="cam_high", frame=0)

    assert exc_info.value.status_code == 404
