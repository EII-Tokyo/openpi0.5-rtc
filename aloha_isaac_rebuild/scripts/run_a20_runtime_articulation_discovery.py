"""Pure aggregation contract for A20 runtime articulation discovery evidence."""

from __future__ import annotations

import argparse
from itertools import pairwise
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
import uuid

import yaml

from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import compare_dof_records
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import validate_dof_records
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import validate_safety_flags

_PASS = "PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP"
_FAIL = "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
_BLOCKED = "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION"
_RUN_PASS = "PASS_RUNTIME_PROBE"
_REQUIRED_RUN_FIELDS = (
    "status",
    "process_status",
    "returncode",
    "timed_out",
    "articulation_root",
    "articulation_count",
    "dof_count",
    "valid_handle",
    "records",
    "requires_unapproved_initialization",
)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _layer1_errors(layer1: object) -> list[dict[str, Any]]:
    details: list[str] = []
    if not isinstance(layer1, dict):
        return [{"code": "invalid_layer1_evidence", "details": ["not_a_dict"]}]

    if layer1.get("status") != "PASS_A20_USD_DOF_METADATA":
        details.append("status")
    if layer1.get("ok") is not True:
        details.append("ok")
    if layer1.get("mismatches") != []:
        details.append("mismatches")
    if layer1.get("errors") != []:
        details.append("errors")

    expected = layer1.get("expected")
    observed = layer1.get("observed")
    if not (
        isinstance(expected, list)
        and isinstance(observed, list)
        and all(isinstance(record, dict) for record in expected + observed)
    ):
        details.append("records_shape")
    else:
        if len(expected) != 16 or not validate_dof_records(expected)["ok"]:
            details.append("expected_records")
        comparison = compare_dof_records(expected, observed)
        if not comparison["ok"]:
            details.append("expected_observed_consistency")

    inputs = layer1.get("inputs")
    if not isinstance(inputs, dict):
        details.append("inputs")
    else:
        stage = inputs.get("stage")
        if not isinstance(stage, dict):
            details.append("stage_input")
        elif not (
            isinstance(stage.get("path"), str)
            and bool(stage["path"])
            and _valid_sha256(stage.get("pre_sha256"))
            and stage.get("pre_sha256") == stage.get("post_sha256")
            and stage.get("consistent_during_audit") is True
        ):
            details.append("stage_hash_consistency")
        for name in ("mapping", "config"):
            item = inputs.get(name)
            if not (
                isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and bool(item["path"])
                and _valid_sha256(item.get("sha256"))
            ):
                details.append(f"{name}_input")

    safety = validate_safety_flags(layer1)
    if not safety["ok"]:
        details.append("safety_flags")

    return [{"code": "invalid_layer1_evidence", "details": details}] if details else []


def _run_errors(
    run: object, run_index: int, expected: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    errors: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    if not isinstance(run, dict):
        return ([{"code": "invalid_run_shape", "run_index": run_index}], [], False)

    missing = [field for field in _REQUIRED_RUN_FIELDS if field not in run]
    errors.extend({"code": "missing_field", "run_index": run_index, "field": field} for field in missing)

    errors.extend(
        {
            "code": "invalid_field_type",
            "run_index": run_index,
            "field": field,
            "expected": "bool",
            "observed_type": type(run[field]).__name__,
        }
        for field in ("timed_out", "valid_handle", "requires_unapproved_initialization")
        if field in run and not isinstance(run[field], bool)
    )
    errors.extend(
        {
            "code": "invalid_field_type",
            "run_index": run_index,
            "field": field,
            "expected": "int",
            "observed_type": type(run[field]).__name__,
        }
        for field in ("returncode", "articulation_count", "dof_count")
        if field in run and type(run[field]) is not int
    )

    safety = validate_safety_flags(run)
    errors.extend({"run_index": run_index, **error} for error in safety["errors"])

    process_ok = (
        run.get("process_status") == "completed"
        and type(run.get("returncode")) is int
        and run["returncode"] == 0
        and run.get("timed_out") is False
    )
    if not process_ok:
        errors.append({"code": "subprocess_failure", "run_index": run_index})

    blocked = (
        run.get("status") == _BLOCKED
        and run.get("requires_unapproved_initialization") is True
        and run.get("valid_handle") is False
    )
    if run.get("status") not in (_RUN_PASS, _BLOCKED):
        errors.append({"code": "invalid_run_status", "run_index": run_index})
    elif run.get("status") == _BLOCKED and not blocked:
        errors.append({"code": "invalid_blocked_evidence", "run_index": run_index})
    elif run.get("status") == _RUN_PASS and run.get("requires_unapproved_initialization") is not False:
        errors.append({"code": "unexpected_initialization_requirement", "run_index": run_index})

    if run.get("articulation_root") != "/aloha/root_joint":
        errors.append({"code": "invalid_articulation_root", "run_index": run_index})
    if type(run.get("articulation_count")) is int and run["articulation_count"] != 1:
        errors.append({"code": "invalid_articulation_count", "run_index": run_index})
    if type(run.get("dof_count")) is int and run["dof_count"] != 16:
        errors.append({"code": "invalid_dof_count", "run_index": run_index})
    if not blocked and run.get("valid_handle") is not True:
        errors.append({"code": "invalid_handle", "run_index": run_index})

    records = run.get("records")
    if not (isinstance(records, list) and all(isinstance(record, dict) for record in records)):
        errors.append({"code": "invalid_records_shape", "run_index": run_index})
    else:
        comparison = compare_dof_records(expected, records)
        if not comparison["ok"]:
            errors.append({"code": "runtime_records_mismatch", "run_index": run_index})
            mismatches.extend({"run_index": run_index, **mismatch} for mismatch in comparison["mismatches"])
            mismatches.extend(
                {"run_index": run_index, "validation_error": validation_error}
                for validation_error in comparison.get("validation_errors", [])
            )

    return errors, mismatches, blocked


def aggregate_runtime_runs(layer1: object, runs: object) -> dict[str, Any]:
    """Aggregate exactly three saved runtime run dictionaries, fail-closed."""
    errors = _layer1_errors(layer1)
    mismatches: list[dict[str, Any]] = []
    run_count = len(runs) if isinstance(runs, list) else None
    if run_count != 3:
        errors.append({"code": "invalid_run_count", "expected": 3, "observed": run_count})

    blocked_runs: list[int] = []
    if not errors and isinstance(layer1, dict) and isinstance(runs, list):
        expected = layer1["expected"]
        for run_index, run in enumerate(runs):
            run_errors, run_mismatches, blocked = _run_errors(run, run_index, expected)
            errors.extend(run_errors)
            mismatches.extend(run_mismatches)
            if blocked and not run_errors:
                blocked_runs.append(run_index)

    if errors:
        status = _FAIL
    elif blocked_runs:
        status = _BLOCKED
    else:
        status = _PASS
    return {
        "status": status,
        "ok": status == _PASS,
        "run_count": run_count,
        "blocked_run_indices": blocked_runs,
        "errors": errors,
        "mismatches": mismatches,
    }


MARKER = "A20_RUNTIME_DISCOVERY_JSON="
DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")


def _summary(value: object, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    return text[-limit:]


def run_three_probes(
    layer1: dict[str, Any],
    repo_root: Path,
    interpreter: Path,
    probe_path: Path,
    timeout_seconds: float,
    run_command=subprocess.run,
    invocation_ids: list[str] | None = None,
) -> dict[str, Any]:
    ids = invocation_ids or [str(uuid.uuid4()) for _ in range(3)]
    errors: list[dict[str, Any]] = []
    if len(ids) != 3 or len(set(ids)) != 3:
        return {
            "status": _FAIL,
            "ok": False,
            "runs": [],
            "errors": [{"code": "invalid_invocation_ids"}],
            "mismatches": [],
        }
    runs: list[dict[str, Any]] = []
    for index, invocation_id in enumerate(ids):
        argv = [str(interpreter), "-u", str(probe_path), "--invocation-id", invocation_id]
        try:
            completed = run_command(
                argv, cwd=repo_root, timeout=timeout_seconds, capture_output=True, text=True, check=False
            )
            stdout, stderr = completed.stdout or "", completed.stderr or ""
            markers = [line[len(MARKER) :] for line in stdout.splitlines() if line.startswith(MARKER)]
            if len(markers) != 1:
                raise ValueError(f"marker_count:{len(markers)}")
            payload = json.loads(markers[0])
            if not isinstance(payload, dict):
                raise ValueError("payload_not_object")
            payload.update(
                process_status="completed" if completed.returncode == 0 else "failed",
                returncode=completed.returncode,
                timed_out=False,
                stdout_summary=_summary(stdout),
                stderr_summary=_summary(stderr),
            )
            if payload.get("invocation_id") != invocation_id:
                errors.append({"code": "invocation_mismatch", "run_index": index})
            runs.append(payload)
        except subprocess.TimeoutExpired as exc:
            runs.append(
                {
                    "status": _FAIL,
                    "process_status": "timeout",
                    "returncode": -1,
                    "timed_out": True,
                    "invocation_id": invocation_id,
                    "stdout_summary": _summary(exc.output),
                    "stderr_summary": _summary(exc.stderr),
                }
            )
            errors.append({"code": "subprocess_timeout", "run_index": index})
        except Exception as exc:
            runs.append(
                {
                    "status": _FAIL,
                    "process_status": "protocol_error",
                    "returncode": -1,
                    "timed_out": False,
                    "invocation_id": invocation_id,
                }
            )
            errors.append({"code": "probe_protocol_error", "run_index": index, "message": str(exc)})
    valid = [run for run in runs if "pid" in run]
    if len({run.get("pid") for run in valid}) != len(valid):
        errors.append({"code": "duplicate_pid"})
    if len({run.get("invocation_id") for run in valid}) != len(valid):
        errors.append({"code": "duplicate_invocation"})
    if valid:
        versions = {run.get("isaac_sim_version") for run in valid}
        fingerprints = {json.dumps(run.get("inputs"), sort_keys=True) for run in valid}
        if len(versions) != 1:
            errors.append({"code": "isaac_version_mismatch"})
        if len(fingerprints) != 1:
            errors.append({"code": "input_hash_mismatch"})
        layer_stage = layer1.get("inputs", {}).get("stage", {}).get("post_sha256")
        if any(run.get("inputs", {}).get("stage", {}).get("sha256") != layer_stage for run in valid):
            errors.append({"code": "layer1_stage_hash_mismatch"})
        for previous, current in pairwise(valid):
            if str(previous.get("finished_at")) > str(current.get("started_at")):
                errors.append({"code": "run_timestamp_overlap"})
    aggregate = aggregate_runtime_runs(layer1, runs)
    aggregate["runs"] = runs
    aggregate["invocation_ids"] = ids
    if errors:
        aggregate["status"], aggregate["ok"] = _FAIL, False
        aggregate["errors"] = errors + aggregate["errors"]
    return aggregate


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--interpreter", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    outputs = config["outputs"]
    layer1 = json.loads((repo / outputs["a20_usd_dof_metadata_json"]).read_text())
    output = (repo / outputs["a20_runtime_articulation_discovery_json"]).resolve()
    result = run_three_probes(
        layer1,
        repo,
        args.interpreter,
        repo / "aloha_isaac_rebuild/scripts/probe_a20_runtime_articulation_once.py",
        args.timeout,
    )
    _atomic_write(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in (_PASS, _BLOCKED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
