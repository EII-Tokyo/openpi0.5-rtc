from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import rebuild_online_rollout_paper_anchor_replay as rebuild


def test_compute_anchor_starts_appends_final_anchor_when_stride_misses_it() -> None:
    starts = rebuild.compute_anchor_starts(num_frames=59, train_horizon=10, chunk_stride=2)

    assert starts.tolist() == list(range(0, 40, 2)) + [39]


def test_compute_anchor_starts_matches_full_stride_grid() -> None:
    starts = rebuild.compute_anchor_starts(num_frames=118, train_horizon=10, chunk_stride=2)

    assert len(starts) == 50
    assert starts[0] == 0
    assert starts[-1] == 98


def test_build_action_windows_uses_anchor_and_horizon() -> None:
    actions = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    starts = np.asarray([0, 2, 3], dtype=np.int64)

    windows = rebuild.build_action_windows(actions, starts, train_horizon=3)

    np.testing.assert_array_equal(windows[0], actions[0:3])
    np.testing.assert_array_equal(windows[1], actions[2:5])
    np.testing.assert_array_equal(windows[2], actions[3:6])


def test_adjacent_exact_fraction_flattens_nonbatch_axes() -> None:
    array = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 5.0]],
        ],
        dtype=np.float32,
    )

    assert rebuild.adjacent_exact_fraction(array) == 0.5


def test_infer_collection_group_separates_legacy_base_from_runtime_actor() -> None:
    legacy = {"train_eligible": True, "created_at": 1783296544.3}
    runtime = {
        "replay_state_grain": "runtime_action_cache_block",
        "requires_offline_reencode": True,
        "created_at": 1783312531.4,
    }
    formal = {
        "replay_state_grain": "paper_subsampled_anchor",
        "formal_replay_ready": True,
    }

    assert rebuild.infer_collection_group(legacy) == "base142_legacy_unmarked"
    assert rebuild.infer_collection_group(runtime) == "actor93_runtime_cache_block"
    assert rebuild.infer_collection_group(formal) == "formal_paper_anchor"


def test_filter_candidates_by_collection_group() -> None:
    candidates = [
        rebuild.Candidate(
            key_region_id="a",
            source_shard_path=rebuild.Path("/tmp/a.npz"),
            rollout_dir=rebuild.Path("/tmp/a"),
            reward=1,
            num_frames=30,
            num_replay_transitions=5,
            train_horizon=10,
            chunk_stride=2,
            action_max_abs_diff=0.0,
            collection_group="base142_legacy_unmarked",
        ),
        rebuild.Candidate(
            key_region_id="b",
            source_shard_path=rebuild.Path("/tmp/b.npz"),
            rollout_dir=rebuild.Path("/tmp/b"),
            reward=0,
            num_frames=30,
            num_replay_transitions=5,
            train_horizon=10,
            chunk_stride=2,
            action_max_abs_diff=0.0,
            collection_group="actor93_runtime_cache_block",
        ),
    ]

    assert [row.key_region_id for row in rebuild.filter_candidates_by_collection_group(candidates, "all")] == ["a", "b"]
    assert [row.key_region_id for row in rebuild.filter_candidates_by_collection_group(candidates, "base142")] == ["a"]
    assert [row.key_region_id for row in rebuild.filter_candidates_by_collection_group(candidates, "actor93")] == ["b"]


def test_manifest_name_uses_dataset_label() -> None:
    assert rebuild.paper_anchor_manifest_name("20260706_base142") == "20260706_base142_paper_anchor_manifest.jsonl"


def test_extract_candidate_token_blocks_batches_vla_frames(tmp_path: Path, monkeypatch) -> None:
    class FakeReader:
        def __init__(self, rollout_dir: Path, convert_bgr_to_rgb: bool) -> None:
            self.rollout_dir = rollout_dir
            self.convert_bgr_to_rgb = convert_bgr_to_rgb
            self.closed = False

        def read_all(self, frame_index: int) -> dict[str, np.ndarray]:
            image = np.full((2, 2, 3), frame_index, dtype=np.uint8)
            return {
                "cam_high": image,
                "cam_low": image,
                "cam_left_wrist": image,
                "cam_right_wrist": image,
            }

        def close(self) -> None:
            self.closed = True

    class FakeExtractor:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def extract_batch(self, observations: list[dict]) -> list[dict[str, np.ndarray]]:
            self.batch_sizes.append(len(observations))
            outputs = []
            for obs in observations:
                frame_id = int(obs["images"]["cam_high"][0, 0, 0])
                outputs.append(
                    {
                        "low_tokens": np.full((2, 3), frame_id, dtype=np.float32),
                        "right_tokens": np.full((2, 3), frame_id + 100, dtype=np.float32),
                        "proprio": np.asarray(obs["state"], dtype=np.float32),
                    }
                )
            return outputs

    qpos = np.arange(10 * 32, dtype=np.float32).reshape(10, 32)
    monkeypatch.setattr(rebuild, "_load_qpos", lambda _: qpos)
    row = rebuild.Candidate(
        key_region_id="abc",
        source_shard_path=tmp_path / "source.npz",
        rollout_dir=tmp_path / "rollout",
        reward=1,
        num_frames=10,
        num_replay_transitions=4,
        train_horizon=2,
        chunk_stride=2,
        action_max_abs_diff=0.0,
        collection_group="actor93_runtime_cache_block",
    )
    out = tmp_path / "tokens.npz"
    extractor = FakeExtractor()

    rebuild.extract_candidate_token_blocks(
        row=row,
        extractor=extractor,
        out=out,
        overwrite=False,
        prompt="test prompt",
        vla_batch_size=2,
        reader_factory=FakeReader,
    )

    assert extractor.batch_sizes == [2, 2, 1]
    with np.load(out, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["current_frames"], np.asarray([0, 2, 4, 6], dtype=np.int64))
        np.testing.assert_array_equal(data["next_frames"], np.asarray([2, 4, 6, 8], dtype=np.int64))
        assert data["low_tokens"].shape == (4, 2, 3)
        np.testing.assert_array_equal(data["low_tokens"][:, 0, 0], np.asarray([0, 2, 4, 6], dtype=np.float16))
        np.testing.assert_array_equal(data["next_low_tokens"][:, 0, 0], np.asarray([2, 4, 6, 8], dtype=np.float16))
