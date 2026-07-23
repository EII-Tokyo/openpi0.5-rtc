from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from aloha_isaac_rebuild.scripts import probe_a21_runtime_target_readback_once as probe

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py"
CANONICAL_PATHS = [
    "/aloha/joints/left_waist",
    "/aloha/joints/left_shoulder",
    "/aloha/joints/left_elbow",
    "/aloha/joints/left_forearm_roll",
    "/aloha/joints/left_wrist_angle",
    "/aloha/joints/left_wrist_rotate",
    "/aloha/joints/left_left_finger",
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_waist",
    "/aloha/joints/right_shoulder",
    "/aloha/joints/right_elbow",
    "/aloha/joints/right_forearm_roll",
    "/aloha/joints/right_wrist_angle",
    "/aloha/joints/right_wrist_rotate",
    "/aloha/joints/right_left_finger",
    "/aloha/joints/right_right_finger",
]
RUNTIME_PATHS = [CANONICAL_PATHS[index] for index in (0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 7, 14, 15)]
POLICY_CANONICAL_GROUPS = [[index] for index in range(6)] + [[6, 7]] + [[index] for index in range(8, 14)] + [[14, 15]]


class FakeArticulationView:
    """Small target-buffer fake; every getter returns an independent ndarray."""

    def __init__(self, targets: object | None = None) -> None:
        self.targets = (
            np.asarray(targets, dtype=float).copy()
            if targets is not None
            else np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.03, 0.0, 0.0, 0.0, 0.0, 0.03, 0.03, 0.03, 0.03]])
        )
        self.write_history: list[tuple[np.ndarray, list[int]]] = []
        self.get_calls = 0
        self.raise_on_set_calls: set[int] = set()
        self.raise_after_write_on_set_calls: set[int] = set()
        self.ignore_set_calls: set[int] = set()
        self.mutate_other_on_set_calls: set[int] = set()
        self.nan_on_get_calls: set[int] = set()

    def get_dof_position_targets(self):
        self.get_calls += 1
        returned = self.targets.copy()
        if self.get_calls in self.nan_on_get_calls:
            returned = returned.astype(float, copy=False)
            returned[0, 0] = math.nan
        return returned

    def set_dof_position_targets(self, values, articulation_indices):
        copied = np.asarray(values).copy()
        indices = list(articulation_indices)
        self.write_history.append((copied, indices))
        call = len(self.write_history)
        if call in self.raise_on_set_calls:
            raise RuntimeError(f"fake setter failure {call}")
        if call not in self.ignore_set_calls:
            self.targets[indices, :] = copied
        if call in self.raise_after_write_on_set_calls:
            raise RuntimeError(f"fake setter post-write failure {call}")
        if call in self.mutate_other_on_set_calls:
            self.targets[0, 15] += 0.001


def _transform(path: str) -> dict[str, object]:
    return {
        "path": path,
        "sign": 1.0,
        "offset": 0.021 if "finger" in path else 0.0,
        "scale": 0.036 if "finger" in path else 1.0,
    }


def _adapter() -> dict[str, object]:
    runtime_by_path = {path: index for index, path in enumerate(RUNTIME_PATHS)}
    entries = []
    for policy_index, canonical_indices in enumerate(POLICY_CANONICAL_GROUPS):
        paths = [CANONICAL_PATHS[index] for index in canonical_indices]
        entries.append(
            {
                "openpi_index": policy_index,
                "runtime_indices": [runtime_by_path[path] for path in paths],
                "transforms": [_transform(path) for path in paths],
            }
        )
    return {
        "schema_version": "a20-policy-runtime-order-v1",
        "policy_dimension": 14,
        "runtime_dimension": 16,
        "canonical_order": list(CANONICAL_PATHS),
        "runtime_order": list(RUNTIME_PATHS),
        "canonical_to_runtime_indices": [runtime_by_path[path] for path in CANONICAL_PATHS],
        "runtime_to_canonical_indices": [CANONICAL_PATHS.index(path) for path in RUNTIME_PATHS],
        "policy_to_runtime": entries,
        "mapping_complete": True,
        "round_trip_check": {
            "status": "PASS",
            "sample_count": 3,
            "gripper_values": [0.0, 0.5, 1.0],
            "max_abs_error": 0.0,
            "error": None,
        },
    }


def _runtime_records() -> list[dict[str, object]]:
    records = []
    for index, path in enumerate(RUNTIME_PATHS):
        finger = "finger" in path
        records.append(
            {
                "index": index,
                "path": path,
                "joint_type": "PhysicsPrismaticJoint" if finger else "PhysicsRevoluteJoint",
                "lower_limit": 0.018 if finger else -180.0,
                "upper_limit": 0.058 if finger else 180.0,
            }
        )
    return records


def _spoof_arm_path(adapter: dict[str, object], records: list[dict[str, object]]) -> None:
    original = RUNTIME_PATHS[0]
    replacement = "/untrusted/joints/left_waist"
    adapter["canonical_order"][0] = replacement
    adapter["runtime_order"][0] = replacement
    adapter["policy_to_runtime"][0]["transforms"][0]["path"] = replacement
    records[0]["path"] = replacement
    assert original not in adapter["canonical_order"]


def _set_bool_adapter_runtime_index(
    adapter: dict[str, object], records: list[dict[str, object]], view: FakeArticulationView
) -> None:
    del records, view
    adapter["policy_to_runtime"][0]["runtime_indices"][0] = True


@pytest.mark.parametrize(("side", "expected"), [("left", list(range(7))), ("right", list(range(7, 14)))])
def test_batch_policy_indices_are_exact(side: str, expected: list[int]) -> None:
    assert probe.batch_policy_indices(side) == expected


def test_batch_policy_indices_reject_unknown_side() -> None:
    with pytest.raises(ValueError, match="side"):
        probe.batch_policy_indices("centre")


def test_choose_interior_delta_prefers_parity_and_switches_at_each_limit() -> None:
    assert probe.choose_interior_delta(0, 0.0, -1.0, 1.0, 0.1) == 0.1
    assert probe.choose_interior_delta(1, 0.0, -1.0, 1.0, 0.1) == -0.1
    assert probe.choose_interior_delta(0, 0.95, -1.0, 1.0, 0.1) == -0.1
    assert probe.choose_interior_delta(1, -0.95, -1.0, 1.0, 0.1) == 0.1
    with pytest.raises(ValueError, match="interior"):
        probe.choose_interior_delta(0, 0.0, -0.05, 0.05, 0.1)


def test_choose_interior_delta_accepts_exact_limit_room() -> None:
    assert probe.choose_interior_delta(0, 0.9, -1.0, 1.0, 0.1) == pytest.approx(0.1)
    assert probe.choose_interior_delta(1, -0.9, -1.0, 1.0, 0.1) == pytest.approx(-0.1)


@pytest.mark.parametrize(
    ("side", "expected_indices"),
    [("left", [0, 2, 4, 6, 8, 10, 12, 13]), ("right", [1, 3, 5, 7, 9, 11, 14, 15])],
)
def test_exercise_writes_only_selected_targets_and_restores_full_baseline(
    side: str, expected_indices: list[int]
) -> None:
    view = FakeArticulationView()
    adapter = _adapter()
    records = _runtime_records()
    adapter_before = deepcopy(adapter)
    records_before = deepcopy(records)
    baseline = view.targets.copy()

    result = probe.exercise_target_batch(view, adapter, records, side=side)

    assert result["ok"] is True
    assert result["runtime_indices"] == expected_indices
    assert result["safety"] == {
        "physics_stepped": False,
        "positions_written": False,
        "velocities_written": False,
        "efforts_written": False,
        "targets_write_attempted": True,
        "targets_written_or_may_have_written": True,
        "targets_written": True,
        "targets_restored": True,
        "target_only_no_step": True,
    }
    assert len(view.write_history) == 2
    assert all(values.shape == (1, 16) and indices == [0] for values, indices in view.write_history)
    intended = view.write_history[0][0]
    untouched = sorted(set(range(16)) - set(expected_indices))
    assert np.array_equal(intended[:, untouched], baseline[:, untouched])
    assert np.array_equal(view.targets, baseline)
    assert adapter == adapter_before
    assert records == records_before
    assert [entry["runtime_index"] for entry in result["deltas"]] == expected_indices
    assert all(entry["path"] == RUNTIME_PATHS[entry["runtime_index"]] for entry in result["deltas"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda adapter, records, view: records.__setitem__(0, {**records[0], "index": 1}),
        lambda adapter, records, view: records.__setitem__(1, {**records[1], "path": records[0]["path"]}),
        lambda adapter, records, view: adapter["policy_to_runtime"][6].pop("transforms"),
        _set_bool_adapter_runtime_index,
        lambda adapter, records, view: records[0].update(lower_limit=True),
        lambda adapter, records, view: records[0].update(upper_limit=math.inf),
        lambda adapter, records, view: adapter["policy_to_runtime"][0]["transforms"][0].update(path="/wrong/path"),
    ],
)
def test_exercise_fails_closed_for_invalid_adapter_or_record_contract(mutate) -> None:
    view = FakeArticulationView()
    adapter = _adapter()
    records = _runtime_records()
    mutate(adapter, records, view)

    result = probe.exercise_target_batch(view, adapter, records, side="left")

    assert result["ok"] is False
    assert result["errors"]
    assert view.write_history == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda adapter, records: _spoof_arm_path(adapter, records),
        lambda adapter, records: records[0].update(joint_type="PhysicsPrismaticJoint"),
        lambda adapter, records: adapter["round_trip_check"].update(status="FAIL"),
    ],
)
def test_exercise_fails_closed_for_spoofed_a20_contract(mutate) -> None:
    view = FakeArticulationView()
    adapter = _adapter()
    records = _runtime_records()
    mutate(adapter, records)

    result = probe.exercise_target_batch(view, adapter, records, side="left")

    assert result["ok"] is False
    assert result["errors"]
    assert view.write_history == []


@pytest.mark.parametrize("targets", [np.zeros((16,)), np.full((1, 16), math.nan)])
def test_exercise_rejects_wrong_or_nonfinite_baseline(targets: object) -> None:
    view = FakeArticulationView(targets)

    result = probe.exercise_target_batch(view, _adapter(), _runtime_records(), side="left")

    assert result["ok"] is False
    assert result["errors"]
    assert view.write_history == []


def test_exercise_rejects_non_ndarray_baseline() -> None:
    view = FakeArticulationView()
    view.get_dof_position_targets = lambda: [[0.0] * 16]  # type: ignore[method-assign]

    result = probe.exercise_target_batch(view, _adapter(), _runtime_records(), side="left")

    assert result["ok"] is False
    assert result["errors"]
    assert view.write_history == []


@pytest.mark.parametrize(
    "targets",
    [
        np.full((1, 16), 1 + 2j, dtype=complex),
        np.full((1, 16), "not-a-number", dtype=object),
        np.zeros((1, 16), dtype=bool),
    ],
)
def test_exercise_rejects_nonreal_numeric_baseline_arrays(targets: np.ndarray) -> None:
    view = FakeArticulationView()
    view.get_dof_position_targets = lambda: targets.copy()  # type: ignore[method-assign]

    result = probe.exercise_target_batch(view, _adapter(), _runtime_records(), side="left")

    assert result["ok"] is False
    assert result["errors"]
    assert view.write_history == []


def test_exercise_rejects_baseline_outside_live_limits() -> None:
    view = FakeArticulationView()
    view.targets[0, 0] = math.pi + 0.1

    result = probe.exercise_target_batch(view, _adapter(), _runtime_records(), side="left")

    assert result["ok"] is False
    assert result["errors"]
    assert view.write_history == []


@pytest.mark.parametrize(
    "configure",
    [
        lambda view: view.mutate_other_on_set_calls.add(1),
        lambda view: view.ignore_set_calls.add(1),
        lambda view: view.nan_on_get_calls.add(2),
        lambda view: view.raise_on_set_calls.add(1),
        lambda view: view.raise_on_set_calls.add(2),
    ],
)
def test_exercise_reports_write_or_readback_failure_and_still_attempts_restore(configure) -> None:
    view = FakeArticulationView()
    configure(view)

    result = probe.exercise_target_batch(view, _adapter(), _runtime_records(), side="right")

    assert result["ok"] is False
    assert result["errors"]
    assert len(view.write_history) == 2
    assert all(values.shape == (1, 16) and indices == [0] for values, indices in view.write_history)


def test_exercise_reports_restoration_readback_mismatch() -> None:
    view = FakeArticulationView()
    view.ignore_set_calls.add(2)

    result = probe.exercise_target_batch(view, _adapter(), _runtime_records(), side="left")

    assert result["ok"] is False
    assert result["safety"]["targets_written"] is True
    assert result["safety"]["targets_restored"] is False
    assert len(view.write_history) == 2


@pytest.mark.parametrize("failure", ["write_then_raise", "nan_readback"])
def test_exercise_restores_baseline_after_write_or_readback_failure(failure: str) -> None:
    view = FakeArticulationView()
    baseline = view.targets.copy()
    if failure == "write_then_raise":
        view.raise_after_write_on_set_calls.add(1)
    else:
        view.nan_on_get_calls.add(2)

    result = probe.exercise_target_batch(view, _adapter(), _runtime_records(), side="right")

    assert result["ok"] is False
    assert result["errors"]
    assert result["safety"]["targets_write_attempted"] is True
    assert result["safety"]["targets_written_or_may_have_written"] is True
    assert result["safety"]["targets_restored"] is True
    assert len(view.write_history) == 2
    assert np.array_equal(view.write_history[1][0], baseline)
    assert view.write_history[1][1] == [0]
    assert np.array_equal(view.targets, baseline)


def test_exact_a20_evidence_gate_requires_layer1_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, object]] = []
    evidence = {"layer": 2}
    layer1 = {"layer": 1}

    def exact(payload: object, trusted: object) -> bool:
        calls.append((payload, trusted))
        return True

    monkeypatch.setattr(probe, "is_exact_runtime_pass", exact)

    assert probe._require_exact_a20_evidence(evidence, layer1) is evidence  # noqa: SLF001
    assert calls == [(evidence, layer1)]


@pytest.mark.parametrize("arguments", [[], ["--invocation-id", "test", "--batch", "invalid"]])
def test_invalid_cli_arguments_emit_one_fail_marker(arguments: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, str(MODULE), *arguments],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        check=False,
        text=True,
    )

    markers = [line for line in completed.stdout.splitlines() if line.startswith(probe.MARKER)]
    assert completed.returncode == 1
    assert len(markers) == 1
    assert json.loads(markers[0][len(probe.MARKER) :])["status"] == probe.FAIL_STATUS


def test_runtime_script_source_has_one_marker_and_no_advance_or_position_apis() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert source.count('MARKER = "A21_RUNTIME_TARGET_READBACK_JSON="') == 1
    assert "from isaacsim import SimulationApp" not in source.split("def main", 1)[0]
    tree = ast.parse(source)
    forbidden = {
        "step",
        "play",
        "update",
        "set_dof_positions",
        "set_joint_positions",
        "set_dof_velocities",
        "set_dof_efforts",
        "set_joint_velocities",
        "set_joint_efforts",
    }
    called = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (called & forbidden)
    allowed_articulation = {
        "get_dof_limits",
        "get_dof_positions",
        "get_dof_position_targets",
        "set_dof_position_targets",
    }
    articulation_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "articulation_view"
    }
    assert articulation_calls <= allowed_articulation
