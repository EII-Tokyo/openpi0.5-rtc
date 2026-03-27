from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")

from pi_value_function.training import data_loader as dl


def _make_sample(task: str = "Put the battery bank in the orange box") -> dict:
    return {
        "joint_position": np.arange(7, dtype=np.float32),
        "gripper_position": np.array(0.5, dtype=np.float32),
        "joint_velocity": np.arange(7, dtype=np.float32) + 10.0,
        "gripper_velocity": np.array(-0.25, dtype=np.float32),
        "exterior_image_1_left": np.ones((3, 2, 2), dtype=np.float32),  # CHW float
        "exterior_image_2_left": np.zeros((2, 2, 3), dtype=np.uint8),   # HWC uint8
        "wrist_image_left": np.full((2, 2, 3), 127, dtype=np.uint8),
        "task": task,
    }


def test_parse_observation_shapes_and_types() -> None:
    parsed = dl.parse_observation(_make_sample())

    assert parsed["state"].shape == (8,)
    assert parsed["image"]["base_0_rgb"].shape == (2, 2, 3)
    assert parsed["image"]["base_0_rgb"].dtype == np.uint8
    assert parsed["image"]["left_wrist_0_rgb"].shape == (2, 2, 3)
    assert parsed["image"]["right_wrist_0_rgb"].shape == (2, 2, 3)
    assert parsed["prompt"] == "Put the battery bank in the orange box"
    assert all(parsed["image_mask"].values())


def test_parse_observation_appends_velocity_when_enabled() -> None:
    parsed = dl.parse_observation(_make_sample(), include_velocity=True)

    assert parsed["state"].shape == (16,)
    assert np.allclose(parsed["state"][:7], np.arange(7, dtype=np.float32))
    assert np.allclose(parsed["state"][7:8], np.array([0.5], dtype=np.float32))
    assert np.allclose(parsed["state"][8:15], np.arange(7, dtype=np.float32) + 10.0)
    assert np.allclose(parsed["state"][15:16], np.array([-0.25], dtype=np.float32))


def test_failure_cost_manager_reads_and_normalizes_prompts(tmp_path: pathlib.Path) -> None:
    costs_path = tmp_path / "costs.json"
    costs_path.write_text(
        json.dumps(
            [
                {"prompt": "Pick Up Cup", "c_fail": 90.0},
                {"prompt": "Open Drawer", "c_fail": 120.0},
            ]
        )
    )

    manager = dl.FailureCostManager(costs_path, default_c_fail=200.0)
    assert manager.get_cost("pick up cup") == 90.0
    assert manager.get_cost("  OPEN DRAWER  ") == 120.0
    assert manager.get_cost("unknown task") == 200.0
    assert manager.get_cost(None) == 200.0


def test_compute_value_target_expected_values_and_clipping() -> None:
    # Success with T=10 and t=9 => raw=-1, normalize from [-10, 0] to [-1, 0] => -0.1
    v_success = dl.compute_value_target(
        timestep=9,
        episode_length=10,
        is_success=True,
        c_fail=0.0,
        raw_value_min=-10.0,
        raw_value_max=0.0,
    )
    assert np.isclose(v_success, -0.1, atol=1e-6)

    # Failure with C_fail=5 and same t => raw=-6 => normalized -0.6
    v_fail = dl.compute_value_target(
        timestep=9,
        episode_length=10,
        is_success=False,
        c_fail=5.0,
        raw_value_min=-10.0,
        raw_value_max=0.0,
    )
    assert np.isclose(v_fail, -0.6, atol=1e-6)

    # Very low raw value should clip to value_min.
    v_clip = dl.compute_value_target(
        timestep=0,
        episode_length=10,
        is_success=False,
        c_fail=100.0,
        raw_value_min=-10.0,
        raw_value_max=0.0,
    )
    assert np.isclose(v_clip, -1.0, atol=1e-6)

    # Degenerate range should return midpoint.
    v_mid = dl.compute_value_target(
        timestep=0,
        episode_length=10,
        is_success=True,
        c_fail=0.0,
        raw_value_min=-1.0,
        raw_value_max=-1.0,
    )
    assert np.isclose(v_mid, -0.5, atol=1e-6)


def test_extract_goal_pairs_parses_single_and_multi_goal_prompts() -> None:
    single = dl._extract_goal_pairs("Put the battery bank in the orange box")
    multi = dl._extract_goal_pairs(
        "Put the battery bank in the orange box and the phone in the blue box"
    )

    assert single == {("battery bank", "orange box")}
    assert multi == {("battery bank", "orange box"), ("phone", "blue box")}


def test_goal_overlap_filtering_keeps_distinct_destination_variant() -> None:
    target = "Put the battery bank in the orange box"
    overlap = "Put the battery bank in the orange box and the phone in the blue box"
    distinct = "Put the battery in the orange box and the battery bank in the blue box"

    target_goals = dl._extract_goal_pairs(target)
    overlap_goals = dl._extract_goal_pairs(overlap)
    distinct_goals = dl._extract_goal_pairs(distinct)

    assert target_goals & overlap_goals
    assert not (target_goals & distinct_goals)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def tokenize(self, prompt: str, include_image_tag: bool = True):
        self.calls.append((prompt, include_image_tag))
        # Fixed-length mock output.
        return np.array([7, 8, 0, 0], dtype=np.int32), np.array([True, True, False, False], dtype=bool)


def test_collate_fn_tokenizes_and_pads_state_to_32() -> None:
    tokenizer = _DummyTokenizer()
    collate = dl.CollateFnWithTokenizer(tokenizer)

    item0 = {
        "prompt": "task one",
        "is_success": True,
        "state": np.arange(8, dtype=np.float32),
        "returns": np.array(-0.2, dtype=np.float32),
        "image": {
            "base_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
    }
    item1 = {
        "prompt": np.array("task two"),
        "is_success": False,
        "state": np.ones(8, dtype=np.float32),
        "returns": np.array(-0.7, dtype=np.float32),
        "image": {
            "base_0_rgb": np.ones((2, 2, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.ones((2, 2, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.ones((2, 2, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
    }

    batch = collate([item0, item1])

    assert batch["tokenized_prompt"].shape == (2, 4)
    assert batch["tokenized_prompt_mask"].shape == (2, 4)
    assert batch["state"].shape == (2, 32)
    assert np.allclose(batch["state"][0, :8], np.arange(8, dtype=np.float32))
    assert np.allclose(batch["state"][0, 8:], 0.0)
    assert np.allclose(batch["state"][1, :8], 1.0)
    assert "is_success" not in batch

    # Ensure collate calls tokenizer with include_image_tag=False for both prompts.
    assert tokenizer.calls == [("task one", False), ("task two", False)]


class _FakeEpisodeDataset:
    def __init__(self, sample: dict) -> None:
        self._sample = sample

    def __getitem__(self, idx: int) -> dict:
        return self._sample


class _FakeCostManager:
    def get_cost(self, prompt: str | None) -> float:
        return 7.0


class _FakeHFDatasetForIndex:
    def __init__(self, episode_index: list[int]) -> None:
        self._episode_index = episode_index

    def __getitem__(self, key: str) -> list[int]:
        if key != "episode_index":
            raise KeyError(key)
        return self._episode_index


class _FakeMeta:
    def __init__(self, episodes: list[dict]) -> None:
        self.episodes = episodes


class _FakeSubDatasetForIndex:
    def __init__(self, *, episodes: list[dict], requested_episodes: list[int], row_episode_index: list[int]) -> None:
        self.meta = _FakeMeta(episodes)
        self.episodes = requested_episodes
        self.hf_dataset = _FakeHFDatasetForIndex(row_episode_index)


class _FakeMultiDatasetForIndex:
    def __init__(self, datasets: list[_FakeSubDatasetForIndex]) -> None:
        self._datasets = datasets


def test_value_function_dataset_getitem_uses_episode_dataset_and_prompt_override() -> None:
    dataset = object.__new__(dl.ValueFunctionDataset)
    fake_ds = _FakeEpisodeDataset(_make_sample(task="Original Task"))

    dataset.success_episodes = []
    dataset.failure_episodes = [
        dl.EpisodeMetadata(
            episode_id=0,
            start_idx=0,
            end_idx=4,
            length=5,
            success=False,
            prompt="Target Task",
            dataset=fake_ds,
        )
    ]
    dataset.success_ratio = 0.0  # Always sample failure path.
    dataset.base_seed = 123
    dataset.rng = np.random.RandomState(0)
    dataset.value_min = -1.0
    dataset.value_max = 0.0
    dataset.target_task = "Forced Target Task"
    dataset.treat_other_tasks_as_failure = True
    dataset.cost_manager = _FakeCostManager()
    dataset.task_normalization = {"target task": (-20.0, 0.0)}
    dataset.global_raw_value_min = -20.0
    dataset.global_raw_value_max = 0.0

    sample = dataset[0]

    assert sample["is_success"] is False
    assert sample["prompt"] == "Forced Target Task"
    assert sample["returns"].dtype == np.float32
    assert -1.0 <= float(sample["returns"]) <= 0.0


def test_build_episode_index_uses_hf_dataset_rows_for_bounds() -> None:
    dataset = object.__new__(dl.ValueFunctionDataset)

    # Metadata lengths are from full dataset, but hf_dataset rows represent a compacted split.
    # Old cumulative-length logic would place episode 2 at start_idx=2000 (out of bounds).
    sub_ds = _FakeSubDatasetForIndex(
        episodes=[
            {"episode_index": 0, "tasks": ["task-0"], "length": 1000},
            {"episode_index": 1, "tasks": ["task-1"], "length": 1000},
            {"episode_index": 2, "tasks": ["task-2"], "length": 1000},
        ],
        requested_episodes=[0, 2],
        row_episode_index=[0] * 10 + [2] * 5,  # compacted split rows
    )
    multi_ds = _FakeMultiDatasetForIndex([sub_ds])

    episode_index = dataset._build_episode_index(multi_ds, success=True)

    assert len(episode_index) == 2
    assert episode_index[0].episode_id == 0
    assert episode_index[0].start_idx == 0
    assert episode_index[0].end_idx == 9
    assert episode_index[0].length == 10

    assert episode_index[1].episode_id == 2
    assert episode_index[1].start_idx == 10
    assert episode_index[1].end_idx == 14
    assert episode_index[1].length == 5
