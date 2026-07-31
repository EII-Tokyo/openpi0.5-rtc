from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

SCRIPT = Path("tools/run_aloha1_grasp_tester_scripted_equivalent.py")

EXPECTED_DOF_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "grasp_tester_scripted_equivalent",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contact(
    body0: str,
    body1: str,
    *,
    collider0: str | None = None,
    collider1: str | None = None,
    impulse_ns: float = 0.1,
    separation_m: float = -0.001,
    physics_step: int = 7,
) -> dict[str, object]:
    return {
        "event_type": "CONTACT_PERSIST",
        "physics_step": physics_step,
        "sim_time_s": physics_step / 60.0,
        "body0_path": body0,
        "body1_path": body1,
        "collider0_path": collider0 or body0,
        "collider1_path": collider1 or body1,
        "impulse_ns": impulse_ns,
        "separation_m": separation_m,
    }


def _passing_record() -> dict[str, object]:
    bottle = "/World/ALOHA1GraspEditorSession/Bottle500"
    return {
        "frozen_hashes_verified": True,
        "actual_dof_order": list(EXPECTED_DOF_ORDER),
        "timeout_reasons": [],
        "tester_terminal_callbacks": 1,
        "tester_success": True,
        "successful_yields": 12,
        "hold_command_count": 12,
        "telemetry": {
            "arm_hold_error_rad": [0.001] * 6,
            "mimic_observation": {
                "left_position_m": 0.022,
                "right_position_m": -0.022,
            },
        },
        "contacts": [
            _contact(
                "/World/follower_left/vx300s_left/left_finger_link",
                bottle,
            ),
            _contact(
                "/World/follower_left/vx300s_left/right_finger_link",
                bottle,
            ),
        ],
    }


def test_frozen_hash_verifier_rejects_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    frozen = tmp_path / "frozen.usda"
    frozen.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module.verify_sha256(frozen, "0" * 64, label="stage")


def test_exact_dof_order_is_required() -> None:
    module = _load_module()
    module.validate_dof_order(EXPECTED_DOF_ORDER)

    wrong = list(EXPECTED_DOF_ORDER)
    wrong[-2:] = reversed(wrong[-2:])
    with pytest.raises(RuntimeError, match="DOF order mismatch"):
        module.validate_dof_order(wrong)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "right_active",
        "right_finger",
        "gripper",
        "left_and_gripper",
        "unknown",
    ],
)
def test_invalid_or_forbidden_variants_are_rejected(name: str) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="variant"):
        module.resolve_variant(name)


def test_variants_preserve_active_and_observer_semantics() -> None:
    module = _load_module()

    variant_a = module.resolve_variant("A")
    assert variant_a["name"] == "dual_active_exact_candidate"
    assert variant_a["active_joints"] == ("left_finger", "right_finger")
    assert variant_a["observer_joints"] == ()
    assert variant_a["mimic_commandability_risk"] is True
    assert variant_a["recommended"] is False

    variant_b = module.resolve_variant("left_active_mimic_observed")
    assert variant_b["name"] == "left_active_mimic_observed"
    assert variant_b["active_joints"] == ("left_finger",)
    assert variant_b["native_export_joints"] == ("left_finger",)
    assert variant_b["observer_joints"] == ("right_finger",)
    assert variant_b["recommended"] is True


def test_timeout_fails_closed() -> None:
    module = _load_module()
    record = _passing_record()
    record["timeout_reasons"] = ["wall_timeout"]

    assert module.evaluate_trial(record) == "FAIL_TIMEOUT"


def test_terminal_success_without_bilateral_contact_is_not_task_pass() -> None:
    module = _load_module()
    record = _passing_record()
    record["contacts"] = record["contacts"][:1]

    assert (
        module.evaluate_trial(record)
        == "INCONCLUSIVE_NO_BILATERAL_CONTACT"
    )


def test_terminal_success_with_bilateral_contacts_is_only_tester_pass() -> None:
    module = _load_module()

    assert (
        module.evaluate_trial(_passing_record())
        == "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"
    )


def test_tester_failure_is_an_explicit_failure() -> None:
    module = _load_module()
    record = _passing_record()
    record["tester_success"] = False

    assert module.evaluate_trial(record) == "FAIL_GRASP_TESTER"


def test_bilateral_contacts_must_share_a_step_and_have_positive_impulse() -> None:
    module = _load_module()
    record = _passing_record()
    record["contacts"][1]["physics_step"] = 8
    assert (
        module.evaluate_trial(record)
        == "INCONCLUSIVE_NO_BILATERAL_CONTACT"
    )

    record = _passing_record()
    record["contacts"][1]["impulse_ns"] = 0.0
    assert (
        module.evaluate_trial(record)
        == "INCONCLUSIVE_NO_BILATERAL_CONTACT"
    )


def test_forbidden_bottle_contact_fails_trial() -> None:
    module = _load_module()
    record = _passing_record()
    record["contacts"].append(
        _contact(
            "/World/follower_left/vx300s_left/follower_left_gripper_bar_link",
            "/World/ALOHA1GraspEditorSession/Bottle500",
        )
    )

    assert (
        module.evaluate_trial(record)
        == "FAIL_FORBIDDEN_BOTTLE_CONTACT"
    )


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_telemetry_fails_closed(bad_value: float) -> None:
    module = _load_module()
    record = _passing_record()
    record["telemetry"]["arm_hold_error_rad"][0] = bad_value

    assert module.evaluate_trial(record) == "FAIL_NONFINITE_TELEMETRY"


def test_missing_arm_hold_command_fails_closed() -> None:
    module = _load_module()
    record = _passing_record()
    record["hold_command_count"] = 0

    assert module.evaluate_trial(record) == "FAIL_MISSING_ARM_HOLD_COMMAND"


def test_bad_hash_and_wrong_dof_are_classified_before_tester_result() -> None:
    module = _load_module()
    record = _passing_record()
    record["frozen_hashes_verified"] = False
    assert module.evaluate_trial(record) == "FAIL_FROZEN_INPUT"

    record = _passing_record()
    record["actual_dof_order"] = list(reversed(EXPECTED_DOF_ORDER))
    assert module.evaluate_trial(record) == "FAIL_DOF_ORDER"


def test_tester_terminal_callback_is_distinct_from_successful_yields() -> None:
    module = _load_module()
    record = _passing_record()
    record["tester_terminal_callbacks"] = 0
    record["successful_yields"] = 50

    assert module.evaluate_trial(record) == "FAIL_NO_TERMINAL_RESULT"


def test_canonical_signature_is_stable_and_json_safes_nonfinite_data() -> None:
    module = _load_module()
    left = {
        "variant": "B",
        "nested": {"b": 2, "a": [1, 3]},
        "canonical_signature": "old-value-is-not-signed",
    }
    right = {
        "nested": {"a": [1, 3], "b": 2},
        "variant": "B",
    }

    assert module.canonical_signature(left) == module.canonical_signature(
        right
    )
    assert len(module.canonical_signature(left)) == 64
    assert len(module.canonical_signature({"bad": math.nan})) == 64


def test_deterministic_trial_signature_excludes_volatile_fields() -> None:
    module = _load_module()
    left = _passing_record()
    left["telemetry"] = [
        {
            "physics_step": 7,
            "sim_time_s": 7 / 60.0,
            "wall_time_s": 1.5,
            "joint_positions": [0.0] * 9,
        }
    ]
    left["report_path"] = "/tmp/first.json"
    left["traceback"] = "first traceback"
    right = json.loads(json.dumps(left))
    right["telemetry"][0]["wall_time_s"] = 99.0
    right["report_path"] = "/tmp/second.json"
    right["traceback"] = "second traceback"

    assert module.deterministic_trial_signature(
        left
    ) == module.deterministic_trial_signature(right)
    changed = json.loads(json.dumps(right))
    changed["telemetry"][0]["physics_step"] = 8
    assert module.deterministic_trial_signature(
        left
    ) != module.deterministic_trial_signature(changed)


def test_nonfinite_trial_remains_machine_readable_failure(
    tmp_path: Path,
) -> None:
    module = _load_module()
    record = _passing_record()
    record["telemetry"]["arm_hold_error_rad"][0] = math.nan
    classification = module.evaluate_trial(record)
    output = tmp_path / "failure.json"

    module._atomic_write_json(  # noqa: SLF001
        output,
        {
            "trial_classification": classification,
            "trial": record,
            "deterministic_trial_signature": (
                module.deterministic_trial_signature(record)
            ),
        },
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["trial_classification"] == "FAIL_NONFINITE_TELEMETRY"
    assert saved["trial"]["telemetry"]["arm_hold_error_rad"][0] == {
        "__nonfinite_float__": "NaN"
    }


@pytest.mark.parametrize(
    ("classification", "expected_exit"),
    [
        ("GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS", 0),
        ("FAIL_RUNTIME", 1),
    ],
)
def test_pre_shutdown_publication_precedes_nonreturning_close(
    tmp_path: Path,
    classification: str,
    expected_exit: int,
) -> None:
    module = _load_module()
    report_path = tmp_path / f"{expected_exit}-report.json"
    telemetry_path = tmp_path / f"{expected_exit}-telemetry.json"
    export_path = tmp_path / f"{expected_exit}-export.yaml"
    verification_states: list[tuple[bool, bool]] = []

    def verify_frozen() -> dict[str, object]:
        verification_states.append(
            (report_path.exists(), telemetry_path.exists())
        )
        return {"stage": {"sha256": "verified"}}

    class _NonReturningApp:
        def close(self) -> None:
            assert report_path.is_file()
            assert telemetry_path.is_file()
            raise SystemExit(99)

    report = {
        **module.TOP_LEVEL_EVIDENCE,
        "trial_classification": classification,
        "trial": {
            "telemetry": [{"physics_step": 1, "value": math.nan}],
            "contacts": [],
            "tester_status_messages": [],
        },
    }

    with pytest.raises(SystemExit, match="99"):
        module._publish_pre_shutdown(  # noqa: SLF001
            report,
            report_path=report_path,
            telemetry_path=telemetry_path,
            export_path=export_path,
            simulation_app=_NonReturningApp(),
            verified_frozen_manifest={"stage": {"sha256": "verified"}},
            frozen_verifier=verify_frozen,
        )

    assert verification_states == [(False, False), (False, True)]
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    saved_telemetry = json.loads(
        telemetry_path.read_text(encoding="utf-8")
    )
    assert saved_report["publication_phase"] == (
        "PRE_KIT_SHUTDOWN_AFTER_PHYSICS_CLEANUP"
    )
    assert saved_report["simulation_app_close_status"] == (
        "SCHEDULED_AS_FINAL_ACTION_NOT_POST_READABLE"
    )
    assert saved_report["shell_exit_code_is_not_authoritative"] is True
    assert saved_report["intended_exit_code"] == expected_exit
    assert saved_telemetry["intended_exit_code"] == expected_exit
    assert saved_telemetry["telemetry"][0]["value"] == {
        "__nonfinite_float__": "NaN"
    }


@pytest.mark.parametrize(
    "failure_point",
    ["final_verifier", "telemetry_write", "report_write"],
)
def test_publication_failure_closes_and_removes_partial_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure_point: str,
) -> None:
    module = _load_module()
    report_path = tmp_path / "report.json"
    telemetry_path = tmp_path / "telemetry.json"
    export_path = tmp_path / "export.yaml"
    export_path.write_text("stale export\n", encoding="utf-8")
    verifier_calls = 0

    def verify_frozen() -> dict[str, object]:
        nonlocal verifier_calls
        verifier_calls += 1
        if failure_point == "final_verifier" and verifier_calls == 2:
            raise RuntimeError("final verifier boom")
        return {"stage": {"sha256": "verified"}}

    def write_json(path: Path, payload: dict[str, object]) -> None:
        if path == telemetry_path and failure_point == "telemetry_write":
            path.write_text("{partial telemetry", encoding="utf-8")
            raise RuntimeError("telemetry write boom")
        if path == report_path and failure_point == "report_write":
            path.write_text("{partial report", encoding="utf-8")
            raise RuntimeError("report write boom")
        module._atomic_write_json(path, payload)  # noqa: SLF001

    class _ClosingApp:
        closed = False

        def close(self) -> None:
            self.closed = True

    app = _ClosingApp()
    expected_message = {
        "final_verifier": "final verifier boom",
        "telemetry_write": "telemetry write boom",
        "report_write": "report write boom",
    }[failure_point]

    with pytest.raises(RuntimeError, match=expected_message):
        module._publish_pre_shutdown(  # noqa: SLF001
            {
                **module.TOP_LEVEL_EVIDENCE,
                "trial_classification": "FAIL_RUNTIME",
                "trial": {
                    "telemetry": [],
                    "contacts": [],
                    "tester_status_messages": [],
                },
            },
            report_path=report_path,
            telemetry_path=telemetry_path,
            export_path=export_path,
            simulation_app=app,
            verified_frozen_manifest={"stage": {"sha256": "verified"}},
            frozen_verifier=verify_frozen,
            json_writer=write_json,
        )

    assert app.closed is True
    assert not report_path.exists()
    assert not telemetry_path.exists()
    assert not export_path.exists()
    stderr = capsys.readouterr().err
    assert "PUBLICATION_FAILURE_BEFORE_CLOSE" in stderr
    assert expected_message in stderr


def test_output_validation_rejects_frozen_and_output_aliases(
    tmp_path: Path,
) -> None:
    module = _load_module()
    frozen = tmp_path / "frozen.usda"
    frozen.write_text("#usda 1.0\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="frozen input"):
        module.validate_output_paths(
            frozen,
            tmp_path / "export.yaml",
            tmp_path / "telemetry.json",
            artifact_dir,
            frozen_paths=(frozen,),
        )

    with pytest.raises(ValueError, match="frozen input"):
        module.validate_output_paths(
            module.STAGE_PATH,
            tmp_path / "export.yaml",
            tmp_path / "telemetry.json",
            artifact_dir,
        )

    with pytest.raises(ValueError, match="must be unique"):
        module.validate_output_paths(
            tmp_path / "same.json",
            tmp_path / "same.json",
            tmp_path / "telemetry.json",
            artifact_dir,
            frozen_paths=(frozen,),
        )


def test_native_export_gate_requires_full_diagnostic_pass() -> None:
    module = _load_module()

    assert module.native_export_status(
        "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"
    ) == "WRITE_NATIVE_EXPORT"
    for classification in (
        "FAIL_GRASP_TESTER",
        "INCONCLUSIVE_NO_BILATERAL_CONTACT",
        "FAIL_FORBIDDEN_BOTTLE_CONTACT",
        "FAIL_NONFINITE_TELEMETRY",
    ):
        assert module.native_export_status(classification) == (
            f"NOT_WRITTEN_{classification}"
        )
        assert module.trial_exit_code(classification) == 1
    assert (
        module.trial_exit_code(
            "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"
        )
        == 0
    )


def test_required_stage_prims_can_appear_after_update() -> None:
    module = _load_module()

    class _Prim:
        def __init__(self, *, valid: bool) -> None:
            self.valid = valid

        def IsValid(self) -> bool:  # noqa: N802 - mirrors USD API.
            return self.valid

    class _Stage:
        def __init__(self, present: set[str]) -> None:
            self.present = present

        def GetPrimAtPath(self, path: str) -> _Prim:  # noqa: N802
            return _Prim(valid=path in self.present)

    required = ("/World/robot", "/World/table")
    context = type("Context", (), {"get_stage": lambda self: None})()

    class _App:
        def is_running(self) -> bool:
            return True

        def update(self) -> None:
            context.get_stage = lambda: _Stage(set(required))

    stage = module.wait_for_required_stage_prims(
        context,
        _App(),
        required,
        timeout_s=1.0,
        monotonic=lambda: 0.0,
    )

    assert all(stage.GetPrimAtPath(path).IsValid() for path in required)


def test_required_stage_prim_wait_fails_on_stop_or_timeout() -> None:
    module = _load_module()
    context = type("Context", (), {"get_stage": lambda self: None})()

    class _StoppedApp:
        def is_running(self) -> bool:
            return False

        def update(self) -> None:
            raise AssertionError("update must not run after app stops")

    with pytest.raises(RuntimeError, match="stopped"):
        module.wait_for_required_stage_prims(
            context,
            _StoppedApp(),
            ("/World/required",),
            timeout_s=1.0,
            monotonic=lambda: 0.0,
        )

    class _RunningApp:
        updates = 0

        def is_running(self) -> bool:
            return True

        def update(self) -> None:
            self.updates += 1

    times = iter((0.0, 0.2, 1.1))
    app = _RunningApp()
    with pytest.raises(RuntimeError, match="stage_load_timeout"):
        module.wait_for_required_stage_prims(
            context,
            app,
            ("/World/required",),
            timeout_s=1.0,
            monotonic=lambda: next(times),
        )
    assert app.updates == 1


def test_stage_timeout_reports_last_missing_prim_state() -> None:
    module = _load_module()

    class _Prim:
        def __init__(self, *, valid: bool) -> None:
            self.valid = valid

        def IsValid(self) -> bool:  # noqa: N802 - mirrors USD API.
            return self.valid

    class _Stage:
        def GetPrimAtPath(self, path: str) -> _Prim:  # noqa: N802
            return _Prim(valid=path == "/World/present")

    context = type("Context", (), {"get_stage": lambda self: _Stage()})()

    class _App:
        updates = 0

        def is_running(self) -> bool:
            return True

        def update(self) -> None:
            self.updates += 1

    app = _App()
    times = iter((0.0, 0.2, 1.1))
    with pytest.raises(RuntimeError) as caught:
        module.wait_for_required_stage_prims(
            context,
            app,
            ("/World/z_missing", "/World/present", "/World/a_missing"),
            timeout_s=1.0,
            monotonic=lambda: next(times),
        )

    message = str(caught.value)
    assert "stage_load_timeout" in message
    assert "stage_was_none=False" in message
    assert (
        'missing_required_prim_paths=["/World/a_missing", '
        '"/World/z_missing"]'
    ) in message
    assert "elapsed_s=1.100000" in message
    assert "update_count=1" in message


def test_stage_readiness_uses_required_prims_not_unsupported_api() -> None:
    module = _load_module()
    source = SCRIPT.read_text(encoding="utf-8")

    unsupported_loading_probe = "is_stage" + "_loading"
    assert unsupported_loading_probe not in source
    assert "wait_for_required_stage_prims" in source
    assert (
        module.TABLE_PATH
        == "/World/environment/worldBody/user_confirmed_table"
    )
    assert module.TABLE_PATH in module.REQUIRED_STAGE_PRIM_PATHS


def test_required_finger_paths_are_rigid_link_prims_not_dof_short_names() -> None:
    module = _load_module()

    assert module.LEFT_FINGER_PATH == (
        "/World/follower_left/vx300s_left/"
        "follower_left_left_finger_link"
    )
    assert module.RIGHT_FINGER_PATH == (
        "/World/follower_left/vx300s_left/"
        "follower_left_right_finger_link"
    )
    assert module.LEFT_FINGER_PATH in module.REQUIRED_STAGE_PRIM_PATHS
    assert module.RIGHT_FINGER_PATH in module.REQUIRED_STAGE_PRIM_PATHS
    assert module.EXPECTED_DOF_ORDER[-2:] == (
        "left_finger",
        "right_finger",
    )


def test_physics_material_binding_uses_local_usd_string_purpose() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls_with_physics_purpose: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        purpose = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "materialPurpose"
            ),
            None,
        )
        if (
            isinstance(purpose, ast.Constant)
            and purpose.value == "physics"
        ):
            calls_with_physics_purpose.add(function.attr)

    assert {
        "Bind",
        "GetDirectBinding",
        "ComputeBoundMaterial",
    } <= calls_with_physics_purpose
    assert "materialPurpose=UsdShade.Tokens.physics" not in source
    assert "materialPurpose=UsdPhysics.Tokens.physics" not in source


def test_only_bottle_rigid_prim_enables_canonical_xform_reset() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    reset_values: dict[str, list[object]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            function_name = function.id
        else:
            continue
        if function_name not in {"RigidPrim", "SingleArticulation"}:
            continue
        reset_keyword = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "reset_xform_properties"
            ),
            None,
        )
        assert isinstance(reset_keyword, ast.Constant)
        reset_values.setdefault(function_name, []).append(
            reset_keyword.value,
        )

    assert reset_values["RigidPrim"] == [True]
    assert reset_values["SingleArticulation"] == [False]


def test_cleanup_restores_only_observed_world_root_metadata() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "restore_root_runtime_metadata" in source
    assert "ClearTimeCodesPerSecond" in source
    assert "ClearStartTimeCode" in source
    assert "ClearEndTimeCode" in source
    assert "ClearCustomLayerData" in source
    assert "root layer dirty state changed" in source
    assert ".Save(" not in source
    assert ".ImportFromString(" not in source


class _FakeRootLayer:
    def __init__(self, *, dirty: bool, reload_state: dict[str, object]):
        self.dirty = dirty
        self._authored = {
            "time": True,
            "start": True,
            "end": True,
            "custom": True,
        }
        self.timeCodesPerSecond = 60.0
        self.startTimeCode = -1.0
        self.endTimeCode = 0.0
        self.customLayerData = {"physicsSettings": {"minFrameRate": 60}}
        self.unknown_root_mutation = "must_remain_visible"
        self.reload_calls: list[bool] = []
        self._reload_state = reload_state

    def HasTimeCodesPerSecond(self) -> bool:  # noqa: N802
        return self._authored["time"]

    def HasStartTimeCode(self) -> bool:  # noqa: N802
        return self._authored["start"]

    def HasEndTimeCode(self) -> bool:  # noqa: N802
        return self._authored["end"]

    def HasCustomLayerData(self) -> bool:  # noqa: N802
        return self._authored["custom"]

    def ClearTimeCodesPerSecond(self) -> None:  # noqa: N802
        self._authored["time"] = False
        self.timeCodesPerSecond = None

    def ClearStartTimeCode(self) -> None:  # noqa: N802
        self._authored["start"] = False
        self.startTimeCode = None

    def ClearEndTimeCode(self) -> None:  # noqa: N802
        self._authored["end"] = False
        self.endTimeCode = None

    def ClearCustomLayerData(self) -> None:  # noqa: N802
        self._authored["custom"] = False
        self.customLayerData = None

    def Reload(self, *, force: bool) -> bool:  # noqa: N802
        self.reload_calls.append(force)
        for name, value in self._reload_state.items():
            setattr(self, name, copy.deepcopy(value))
        return True


def _root_before(*, dirty: bool) -> dict[str, object]:
    return {
        "dirty": dirty,
        "runtime_metadata": {
            "time_codes_per_second": {
                "authored": False,
                "value": None,
            },
            "start_time_code": {"authored": False, "value": None},
            "end_time_code": {"authored": False, "value": None},
            "custom_layer_data": {"authored": False, "value": None},
        },
    }


def test_dirty_root_metadata_restore_is_targeted_and_preserves_unknowns() -> None:
    module = _load_module()
    root = _FakeRootLayer(dirty=True, reload_state={})

    module.restore_root_runtime_metadata(root, _root_before(dirty=True))

    assert root.reload_calls == []
    assert root.HasTimeCodesPerSecond() is False
    assert root.HasStartTimeCode() is False
    assert root.HasEndTimeCode() is False
    assert root.HasCustomLayerData() is False
    assert root.unknown_root_mutation == "must_remain_visible"
    assert root.dirty is True


def test_clean_root_metadata_restore_reloads_to_recover_clean_state() -> None:
    module = _load_module()
    root = _FakeRootLayer(
        dirty=True,
        reload_state={
            "dirty": False,
            "unknown_root_mutation": "disk_baseline",
        },
    )

    module.restore_root_runtime_metadata(root, _root_before(dirty=False))

    assert root.reload_calls == [True]
    assert root.dirty is False
    assert root.unknown_root_mutation == "disk_baseline"


def test_output_validation_rejects_existing_hardlink_to_frozen(
    tmp_path: Path,
) -> None:
    module = _load_module()
    frozen = tmp_path / "frozen.usda"
    frozen.write_text("#usda 1.0\n", encoding="utf-8")
    alias = tmp_path / "report.json"
    os.link(frozen, alias)

    with pytest.raises(ValueError, match="same file as frozen"):
        module.validate_output_paths(
            alias,
            tmp_path / "export.yaml",
            tmp_path / "telemetry.json",
            tmp_path / "artifacts",
            frozen_paths=(frozen,),
        )


def test_transform_chain_uses_world_gripper_and_inverse_object_gripper() -> None:
    module = _load_module()
    report = {
        "matrices": {
            "world_from_gripper_reference": [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "object_from_gripper": [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    result = module.compute_world_from_object(report)

    expected = [
        [1.0, 0.0, 0.0, 9.0],
        [0.0, 1.0, 0.0, 18.0],
        [0.0, 0.0, 1.0, 27.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for actual_row, expected_row in zip(result, expected, strict=True):
        assert actual_row == pytest.approx(expected_row)


def test_source_contract_is_scripted_equivalent_without_ik_or_attachment() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    lower = source.lower()
    tree = ast.parse(source)

    assert "from isaacsim import SimulationApp" in source
    assert source.index("from isaacsim import SimulationApp") < source.index(
        "from omni"
    )
    assert "GraspTester" in source
    assert "GraspTestSettings" in source
    assert "DataWriter" in source
    assert "SingleArticulation" in source
    assert "RigidPrim(" in source
    assert 'Sdf.Path("/Bottle500")' in source
    assert "Layer.CreateAnonymous" in source
    assert "stage.SetEditTarget" in source
    assert "world.reset()" in source
    assert "world.add_physics_callback" in source
    assert source.index("world.reset()") < source.index(
        "world.add_physics_callback"
    )
    assert "tester.update_grasp_test(dt)" in source
    assert "world.step(" in source
    assert "World.clear_instance()" in source
    assert "set_solve_articulation_contact_last(True)" in source
    assert "get_solve_articulation_contact_last()" in source
    assert "stage_load_timeout" in source
    assert "simulation_app.is_running()" in source
    assert "native_export_status(trial_classification)" in source
    assert '"GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"' in source
    simulation_import = source.index("from isaacsim import SimulationApp")
    protected_try = source.index("try:", simulation_import)
    omni_import = source.index("from omni", simulation_import)
    assert simulation_import < protected_try < omni_import

    for forbidden in (
        "rmpflow",
        "lula",
        "compute_inverse_kinematics",
        "articulationkinematicssolver",
        "kinematicssolver",
        "motion_generation",
        "surfacegripper",
        "fixedjoint",
        "parent_attachment",
        "world.clear()",
        ".save(",
        "save_as_stage",
        "convert_prim_to_collidable_rigid_body",
    ):
        assert forbidden not in lower

    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id.lower())
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr.lower())

    assert not any(
        "motion_generation" in module.lower()
        for module in imported_modules
    )
    assert called_names.isdisjoint(
        {
            "articulationkinematicssolver",
            "kinematicssolver",
            "compute_inverse_kinematics",
        }
    )


def test_native_export_validation_reads_file_and_hashes_content(
    tmp_path: Path,
) -> None:
    module = _load_module()
    export = tmp_path / "grasp.yaml"
    export.write_text(
        "\n".join(
            [
                "format: isaac_grasp",
                "format_version: 1.0",
                f"object_frame: {module.BOTTLE_SESSION_PATH}",
                f"gripper_frame: {module.GRIPPER_FRAME_PATH}",
                "grasps:",
                "  grasp_0:",
                "    confidence: 1.0",
                "    position: [0.1, 0.2, 0.3]",
                "    orientation: {w: 1.0, xyz: [0.0, 0.0, 0.0]}",
                "    cspace_position: {left_finger: 0.021}",
                "    pregrasp_cspace_position: {left_finger: 0.057}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.validate_native_export(
        export,
        ("left_finger",),
    )

    assert result["sha256"] == module.sha256_file(export)
    assert result["size_bytes"] == export.stat().st_size
    assert result["active_joints"] == ["left_finger"]
    assert result["grasp_count"] == 1


def test_native_export_validation_rejects_wrong_or_nonfinite_content(
    tmp_path: Path,
) -> None:
    module = _load_module()
    export = tmp_path / "bad.yaml"
    export.write_text(
        "\n".join(
            [
                "format: isaac_grasp",
                "format_version: 1.0",
                f"object_frame: {module.BOTTLE_SESSION_PATH}",
                f"gripper_frame: {module.GRIPPER_FRAME_PATH}",
                "grasps:",
                "  grasp_0:",
                "    confidence: .nan",
                "    position: [0.1, 0.2, 0.3]",
                "    orientation: {w: 1.0, xyz: [0.0, 0.0, 0.0]}",
                "    cspace_position: {right_finger: -0.021}",
                "    pregrasp_cspace_position: {right_finger: -0.057}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        module.validate_native_export(export, ("left_finger",))


def test_deterministic_run_signature_covers_cleanup_export_and_variant() -> None:
    module = _load_module()
    report = {
        "variant": {"name": "B"},
        "trial_classification": "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS",
        "frozen_inputs": {"stage": {"sha256": "a" * 64}},
        "cleanup": {
            "errors": [],
            "post_cleanup_hash_errors": [],
            "no_persistent_stage_write": True,
        },
        "native_export_validation": {"sha256": "b" * 64},
        "trial": {
            "tester_success": True,
            "telemetry": [{"wall_time_s": 1.0, "physics_step": 1}],
        },
        "native_export_path": "/volatile/run1/grasp.yaml",
    }
    equivalent = copy.deepcopy(report)
    equivalent["native_export_path"] = "/volatile/run2/grasp.yaml"
    equivalent["trial"]["telemetry"][0]["wall_time_s"] = 9.0

    signature = module.deterministic_run_signature(report)

    assert signature == module.deterministic_run_signature(equivalent)
    changed = copy.deepcopy(report)
    changed["cleanup"]["no_persistent_stage_write"] = False
    assert signature != module.deterministic_run_signature(changed)
    changed = copy.deepcopy(report)
    changed["native_export_validation"]["sha256"] = "c" * 64
    assert signature != module.deterministic_run_signature(changed)


def test_full_experience_resolves_from_local_installed_package() -> None:
    module = _load_module()
    package_files = list(
        (module.REPO_ROOT / ".venv_issac/lib").glob(
            "python*/site-packages/isaacsim/__init__.py"
        )
    )
    assert len(package_files) == 1

    experience = module.resolve_full_experience(package_files[0])

    assert experience.is_absolute()
    assert experience.is_file()
    assert experience.name == "isaacsim.exp.full.kit"
    assert experience.parent == package_files[0].resolve().parent / "apps"


def test_simulation_app_uses_computed_absolute_full_experience() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SimulationApp"
    ]

    assert len(calls) == 1
    call = calls[0]
    experience_keywords = [
        keyword for keyword in call.keywords if keyword.arg == "experience"
    ]
    assert len(experience_keywords) == 1
    assert ast.unparse(experience_keywords[0].value) == (
        "str(full_experience_path)"
    )
    assert len(call.args) == 1
    assert ast.literal_eval(call.args[0]) == {
        "fast_shutdown": False,
        "headless": True,
        "sync_loads": True,
    }


def test_import_exception_is_logged_before_pre_shutdown_publication() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    simulation_import = source.index("from isaacsim import SimulationApp")
    import_handler = source.index(
        "except Exception as exc:",
        simulation_import,
    )
    log_call = source.index(
        "_emit_bounded_import_error(exc)",
        import_handler,
    )
    publication_call = source.index(
        "return _publish_pre_shutdown(",
        import_handler,
    )

    assert import_handler < log_call < publication_call


def test_runtime_exception_is_logged_before_finally_and_publication() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    runtime_handler = source.index(
        "\n    except Exception as exc:",
        source.index("artifact_dir.mkdir"),
    )
    log_call = source.index(
        '_emit_bounded_error("RUNTIME_FAILURE_BEFORE_CLOSE", exc)',
        runtime_handler,
    )
    finally_block = source.index("\n    finally:", runtime_handler)
    publication_call = source.index(
        "return _publish_pre_shutdown(",
        finally_block,
    )

    assert runtime_handler < log_call < finally_block < publication_call


def test_authoritative_writes_and_hash_checks_precede_only_close() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    helper_start = source.index("def _publish_pre_shutdown(")
    helper_end = source.index("\ndef ", helper_start + 1)
    helper_source = source[helper_start:helper_end]
    close_call = helper_source.index("simulation_app.close()")
    telemetry_write = helper_source.index(
        "json_writer(telemetry_path, telemetry_payload)"
    )
    final_hash_gate = helper_source.rindex("frozen_verifier()")
    report_commit = helper_source.index("json_writer(report_path, report)")

    assert helper_source.count("simulation_app.close()") == 1
    assert helper_source.count("frozen_verifier()") == 2
    assert telemetry_write < final_hash_gate < report_commit < close_call
    after_close = helper_source[close_call:]
    assert "frozen_verifier()" not in after_close
    assert "json_writer(" not in after_close


def test_source_pins_inputs_versions_and_partial_classification() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for expected in (
        "5.1.0.0",
        "107.3.3",
        "107.3.26",
        "2.0.20",
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf",
        "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e",
        "b3307c86a44101eadd6ed2151722e7668bb7d644422378765d98eac906835cca",
        "37d36dcbb4bfd7a9fdc39f96565c796bdc0d9b8d571172bf4639251a23b3f329",
        '"status": "PARTIAL"',
        '"gui_evidence": "GUI_PENDING"',
        '"ik": "NOT_RUN"',
        '"classification": "DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI"',
        "NOT_TABLE_TASK/NOT_IK",
        "INCONCLUSIVE_NO_APPROVED_ARM_HOLD_TOLERANCE",
        "INCONCLUSIVE_NO_APPROVED_MIMIC_TOLERANCE",
    ):
        assert expected in source


def test_source_uses_exact_initial_frame_and_physics_parameters() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for expected in (
        "-0.16720470786094666",
        "0.5324101448059082",
        "-0.017540352419018745",
        "-0.3624092638492584",
        "0.9591664671897888",
        "-0.11042828112840652",
        "OPEN_LEFT_M = 0.057",
        "OPEN_RIGHT_M = -0.057",
        "CLOSE_LEFT_M = 0.021",
        "CLOSE_RIGHT_M = -0.021",
        "CLOSE_SPEED_M_S = 0.02",
        "PHYSICS_DT_S = 1.0 / 60.0",
        "BOTTLE_MASS_KG = 0.020",
        "FRICTION = 0.7",
        "RESTITUTION = 0.0",
        "EXPECTED_BOTTLE_COLLISIONS = 41",
    ):
        assert expected in source
