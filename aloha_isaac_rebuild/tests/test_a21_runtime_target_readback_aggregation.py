from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from aloha_isaac_rebuild.scripts import probe_a21_runtime_target_readback_once as probe
from aloha_isaac_rebuild.scripts import run_a21_runtime_target_readback as coordinator

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py"
HASH = "a" * 64
LEFT = [0, 2, 4, 6, 8, 10, 12, 13]
RIGHT = [1, 3, 5, 7, 9, 11, 14, 15]


def _preflight() -> dict[str, object]:
    return {
        "schema_version": "a21-policy-target-limit-v1",
        "status": "PASS_A21_POLICY_TARGET_LIMIT_PREFLIGHT",
        "ok": True,
        "sample_count": 4,
        "samples": [{"label": label} for label in range(4)],
        "mismatches": [],
        "errors": [],
        "inputs": {
            "config": {"path": "/inputs/config.yaml", "sha256": HASH},
            "layer1": {"path": "/inputs/layer1.json", "sha256": HASH},
            "layer2": {"path": "/inputs/layer2.json", "sha256": HASH},
        },
        "physics_stepped": False,
        "actions_applied": False,
        "targets_written": False,
        "targets_restored": False,
        "stage_saved": False,
    }


def _run(side: str, pid: int) -> dict[str, object]:
    indices = LEFT if side == "left" else RIGHT
    return {
        "status": probe.PASS_STATUS,
        "batch": side,
        "runtime_indices": list(indices),
        "invocation_id": f"{side}-invocation",
        "pid": pid,
        "marker_count": 1,
        "process_status": "completed",
        "returncode": 0,
        "timed_out": False,
        "output_limit_exceeded": False,
        "cleanup_verified": True,
        "physics_stepped": False,
        "actions_applied": False,
        "positions_written": False,
        "velocities_written": False,
        "efforts_written": False,
        "stage_saved": False,
        "inputs": {
            "config": {"path": "/inputs/config.yaml", "sha256": HASH},
            "stage": {"path": "/inputs/stage.usda", "sha256": HASH},
            "mapping": {"path": "/inputs/mapping.json", "sha256": HASH},
            "a20_evidence": {"path": "/inputs/layer2.json", "sha256": HASH},
            "a20_layer1": {"path": "/inputs/layer1.json", "sha256": HASH},
        },
        "provenance": {
            "git_head": "f" * 40,
            "git_dirty": False,
            "probe_sha256": HASH,
            "coordinator_sha256": HASH,
            "safety_checker": {"ok": True},
        },
        "result": {
            "ok": True,
            "runtime_indices": list(indices),
            "safety": {
                "physics_stepped": False,
                "positions_written": False,
                "velocities_written": False,
                "efforts_written": False,
                "targets_write_attempted": True,
                "targets_written_or_may_have_written": True,
                "targets_written": True,
                "targets_restored": True,
                "target_only_no_step": True,
            },
        },
        "safety": {
            "physics_stepped": False,
            "positions_written": False,
            "velocities_written": False,
            "efforts_written": False,
            "targets_write_attempted": True,
            "targets_written_or_may_have_written": True,
            "targets_written": True,
            "targets_restored": True,
            "target_only_no_step": True,
        },
    }


def test_check_probe_source_accepts_task5_probe_and_rejects_aliases() -> None:
    assert coordinator.check_probe_source(PROBE.read_text(encoding="utf-8"))["ok"] is True
    for source in (
        "runner = view.set_dof_position_targets\nrunner()\n",
        "import importlib\nimportlib.import_module('omni.usd')\n",
        "getattr(view, 'get_dof_position_targets')()\n",
        "interface.update_simulation(0.0, 0.0)\n",
    ):
        assert coordinator.check_probe_source(source)["ok"] is False


def test_aggregate_batches_accepts_exact_left_then_right_contract_without_mutation() -> None:
    preflight = _preflight()
    runs = [_run("left", 101), _run("right", 102)]
    before = deepcopy((preflight, runs))

    result = coordinator.aggregate_batches(preflight, runs)

    assert result["status"] == coordinator.PASS_STATUS
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["run_count"] == 2
    assert result["runtime_indices"] == list(range(16))
    assert result["targets_written"] is True
    assert result["targets_restored"] is True
    assert (preflight, runs) == before


def test_aggregate_batches_rejects_a_prohibited_top_level_action_flag() -> None:
    runs = [_run("left", 101), _run("right", 102)]
    runs[1]["actions_applied"] = True

    result = coordinator.aggregate_batches(_preflight(), runs)

    assert result["status"] == coordinator.FAIL_STATUS
    assert any(error["code"] == "prohibited_safety_flag" for error in result["errors"])


def test_left_nonrestored_contract_prevents_right_invocation(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    left = _run("left", 7001)
    left["safety"]["targets_restored"] = False  # type: ignore[index]
    left["result"]["safety"]["targets_restored"] = False  # type: ignore[index]

    def execute(argv: list[str], _cwd: Path, _timeout: float) -> dict[str, object]:
        calls.append(argv)
        payload = left if argv[-1] == "left" else _run("right", 7002)
        payload = deepcopy(payload)
        payload["invocation_id"] = argv[argv.index("--invocation-id") + 1]
        return {
            "process_status": "completed",
            "returncode": 0,
            "timed_out": False,
            "output_limit_exceeded": False,
            "cleanup_verified": True,
            "observed_pid": payload["pid"],
            "stdout": coordinator.MARKER + json.dumps(payload) + "\n",
            "stderr": "",
        }

    runs = coordinator.run_two_batches(
        tmp_path, Path("/bin/true"), PROBE, 1.0, execute=execute, invocation_ids=("left", "right")
    )

    assert len(calls) == 1
    assert len(runs) == 1


def test_aggregate_rejects_wrong_order_duplicate_ids_bad_hashes_and_target_failures() -> None:
    cases = []
    wrong_order = [_run("right", 101), _run("left", 102)]
    cases.append(wrong_order)
    duplicate = [_run("left", 101), _run("right", 101)]
    duplicate[1]["invocation_id"] = duplicate[0]["invocation_id"]
    cases.append(duplicate)
    bad_hash = [_run("left", 101), _run("right", 102)]
    bad_hash[1]["provenance"]["probe_sha256"] = "b" * 64  # type: ignore[index]
    cases.append(bad_hash)
    not_written = [_run("left", 101), _run("right", 102)]
    not_written[0]["safety"]["targets_written"] = False  # type: ignore[index]
    cases.append(not_written)
    overlap = [_run("left", 101), _run("right", 102)]
    overlap[1]["runtime_indices"] = list(LEFT)
    overlap[1]["result"]["runtime_indices"] = list(LEFT)  # type: ignore[index]
    cases.append(overlap)

    for runs in cases:
        assert coordinator.aggregate_batches(_preflight(), runs)["status"] == coordinator.FAIL_STATUS


def test_marker_protocol_and_output_limit_prevent_right_invocation(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def execute(argv: list[str], _cwd: Path, _timeout: float) -> dict[str, object]:
        calls.append(argv)
        return {
            "process_status": "output_limit_exceeded",
            "returncode": -1,
            "timed_out": False,
            "output_limit_exceeded": True,
            "cleanup_verified": True,
            "observed_pid": 7001,
            "stdout": coordinator.MARKER + "{}\n" + coordinator.MARKER + "{}\n",
            "stderr": "",
        }

    runs = coordinator.run_two_batches(
        tmp_path, Path("/bin/true"), PROBE, 1.0, execute=execute, invocation_ids=("left", "right")
    )

    assert len(calls) == len(runs) == 1
    assert runs[0]["status"] == probe.FAIL_STATUS
    assert runs[0]["marker_count"] == 2


def test_timeout_terminates_non_isaac_process_group(tmp_path: Path) -> None:
    """This uses only a temporary Python child; it never launches Isaac or a GUI."""
    execution = coordinator._execute_probe(  # noqa: SLF001 - exercises bounded process-group cleanup.
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 0.05
    )

    assert execution["timed_out"] is True
    assert execution["cleanup_verified"] is True


def test_missing_marker_is_a_fail_closed_protocol_error() -> None:
    run = coordinator._batch_payload(  # noqa: SLF001 - validates raw marker parser.
        {
            "process_status": "completed",
            "returncode": 0,
            "timed_out": False,
            "output_limit_exceeded": False,
            "cleanup_verified": True,
            "observed_pid": 123,
            "stdout": "ordinary output only\n",
            "stderr": "",
        },
        "expected-id",
        "left",
    )

    assert run["status"] == probe.FAIL_STATUS
    assert run["marker_count"] == 0


def test_batch_payload_promotes_task5_result_indices_without_aliasing() -> None:
    marker = _run("left", 123)
    marker.pop("runtime_indices")
    run = coordinator._batch_payload(  # noqa: SLF001 - validates raw Task5 marker normalization.
        {
            "process_status": "completed",
            "returncode": 0,
            "timed_out": False,
            "output_limit_exceeded": False,
            "cleanup_verified": True,
            "observed_pid": 123,
            "stdout": coordinator.MARKER + json.dumps(marker) + "\n",
            "stderr": "",
        },
        "left-invocation",
        "left",
    )

    assert run["runtime_indices"] == LEFT
    assert run["runtime_indices"] is not run["result"]["runtime_indices"]  # type: ignore[index]
    report = coordinator.format_report({"status": coordinator.PASS_STATUS, "ok": True, "runs": [run]})
    assert f"indices: {LEFT}" in report


def test_batch_payload_rejects_conflicting_top_level_and_task5_result_indices() -> None:
    marker = _run("left", 123)
    marker["runtime_indices"] = list(RIGHT)
    run = coordinator._batch_payload(  # noqa: SLF001 - validates raw Task5 marker normalization.
        {
            "process_status": "completed",
            "returncode": 0,
            "timed_out": False,
            "output_limit_exceeded": False,
            "cleanup_verified": True,
            "observed_pid": 123,
            "stdout": coordinator.MARKER + json.dumps(marker) + "\n",
            "stderr": "",
        },
        "left-invocation",
        "left",
    )

    assert run["status"] == probe.FAIL_STATUS
    assert any(error["code"] == "runtime_indices_protocol_mismatch" for error in run["errors"])


def test_code_provenance_rejects_dirty_probe_or_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(("deadbeef\n", " M probe.py\n"))
    monkeypatch.setattr(coordinator.subprocess, "check_output", lambda *args, **kwargs: next(responses))

    provenance = coordinator._code_provenance(  # noqa: SLF001 - direct provenance seam.
        ROOT, PROBE, ROOT / "aloha_isaac_rebuild/scripts/run_a21_runtime_target_readback.py"
    )

    assert provenance["git_head"] == "deadbeef"
    assert provenance["git_dirty"] is True


def test_aggregate_rejects_stale_recorded_git_head() -> None:
    runs = [_run("left", 101), _run("right", 102)]
    runs[1]["provenance"]["git_head"] = "e" * 40  # type: ignore[index]

    result = coordinator.aggregate_batches(_preflight(), runs)

    assert result["status"] == coordinator.FAIL_STATUS
    assert any(error["code"] == "stale_or_dirty_git_provenance" for error in result["errors"])


def _write_cli_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    stage = tmp_path / "stage.usda"
    mapping = tmp_path / "mapping.json"
    layer1 = tmp_path / "layer1.json"
    layer2 = tmp_path / "layer2.json"
    preflight = tmp_path / "preflight.json"
    for path in (stage, mapping, layer1, layer2, preflight):
        path.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.json"
    report = tmp_path / "report.md"
    config = tmp_path / "config.yaml"
    config.write_text(
        "outputs:\n"
        f"  a19_clean_articulation_candidate: {stage}\n"
        f"  a17_clean_articulation_mapping_plan_json: {mapping}\n"
        f"  a20_usd_dof_metadata_json: {layer1}\n"
        f"  a20_runtime_articulation_discovery_json: {layer2}\n"
        f"  a21_policy_target_limit_preflight_json: {preflight}\n"
        f"  a21_runtime_target_readback_json: {output}\n"
        f"  a21_target_limit_and_readback_md: {report}\n",
        encoding="utf-8",
    )
    return config, stage, report


def test_cli_fails_if_stage_hash_changes_during_coordinate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, stage, _report = _write_cli_paths(tmp_path)
    real_digest = coordinator._digest  # noqa: SLF001 - deterministic stage-race seam.
    calls = 0

    def digest(path: Path) -> str:
        nonlocal calls
        if path == stage:
            calls += 1
            return "a" * 64 if calls == 1 else "b" * 64
        return real_digest(path)

    monkeypatch.setattr(coordinator, "_digest", digest)
    monkeypatch.setattr(sys, "argv", ["coordinator", "--config", str(config)])

    assert coordinator.main() == 1

    result = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))
    assert result["status"] == coordinator.FAIL_STATUS
    assert any(error["code"] == "stage_hash_changed_during_coordinate" for error in result["errors"])


def test_cli_report_write_failure_removes_stale_ready_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _stage, report = _write_cli_paths(tmp_path)
    report.write_text("Overall: READY\n", encoding="utf-8")
    monkeypatch.setattr(coordinator, "_atomic_write_text", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(sys, "argv", ["coordinator", "--config", str(config)])

    assert coordinator.main() == 1

    assert not report.exists()
    result = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))
    assert result["status"] == coordinator.FAIL_STATUS
    assert any(error["code"] == "report_write_failed" for error in result["errors"])


def test_cli_preserves_absolute_venv_launcher_symlink_for_batch_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _stage, _report = _write_cli_paths(tmp_path)
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))
    captured: dict[str, Path] = {}

    def run_two_batches(_repo: Path, interpreter: Path, *_args: object, **_kwargs: object) -> list[dict[str, object]]:
        captured["interpreter"] = interpreter
        return []

    monkeypatch.setattr(
        coordinator, "_code_provenance", lambda *_args: {"git_dirty": False, "safety_checker": {"ok": True}}
    )
    monkeypatch.setattr(coordinator, "is_exact_runtime_pass", lambda *_args: True)
    monkeypatch.setattr(coordinator, "_exact_preflight", lambda *_args: True)
    monkeypatch.setattr(coordinator, "run_two_batches", run_two_batches)
    monkeypatch.setattr(sys, "argv", ["coordinator", "--config", str(config), "--interpreter", str(launcher)])

    assert coordinator.main() == 1

    assert captured["interpreter"] == launcher.absolute()
    assert captured["interpreter"] != launcher.resolve()
