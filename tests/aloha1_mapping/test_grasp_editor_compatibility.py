from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROBE = Path("tools/probe_aloha1_grasp_editor_compatibility.py")
LAUNCHER = Path("tools/open_aloha1_grasp_editor_diagnostic.py")
REPORT = Path(
    "reports/aloha1_mapping/aloha1_grasp_editor_compatibility.json"
)


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("grasp_editor_probe", PROBE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_launcher_module():
    spec = importlib.util.spec_from_file_location(
        "grasp_editor_launcher",
        LAUNCHER,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEditTarget:
    def __init__(self, layer: object) -> None:
        self._layer = layer

    def GetLayer(self) -> object:  # noqa: N802 - mirrors USD API.
        return self._layer


class _FakeStage:
    def __init__(
        self,
        target_layer: object,
        events: list[str] | None = None,
        *,
        restore_failure: bool = False,
        silent_restore_failure: bool = False,
    ) -> None:
        self.target_layer = target_layer
        self.events = events
        self.restore_error = (
            RuntimeError("restore failed") if restore_failure else None
        )
        self.silent_restore_failure = silent_restore_failure

    def GetEditTarget(self) -> _FakeEditTarget:  # noqa: N802
        return _FakeEditTarget(self.target_layer)

    def SetEditTarget(self, edit_target: object) -> None:  # noqa: N802
        if self.events is not None:
            self.events.append("restore")
        if self.restore_error is not None:
            raise self.restore_error
        if self.silent_restore_failure:
            return
        if isinstance(edit_target, _FakeEditTarget):
            self.target_layer = edit_target.GetLayer()
        else:
            self.target_layer = edit_target


class _FakeRootLayer:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    @property
    def dirty(self) -> bool:
        self._events.append("root_dirty")
        return False

    def ExportToString(self) -> str:  # noqa: N802
        self._events.append("root_specs")
        return "frozen root specs"


class _FakeSubLayerPaths(list[str]):
    def __init__(
        self,
        identifier: str,
        events: list[str],
        *,
        removal_failure: bool = False,
    ) -> None:
        super().__init__([identifier])
        self._events = events
        self._removal_failure = removal_failure

    def remove(self, value: str) -> None:
        self._events.append("remove")
        if self._removal_failure:
            raise RuntimeError("remove failed")
        super().remove(value)


class _FakeSessionLayer:
    def __init__(self, sub_layer_paths: _FakeSubLayerPaths) -> None:
        self.subLayerPaths = sub_layer_paths


class _FakeApp:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def close(self) -> None:
        self._events.append("close")


class _FakeExtensionManager:
    def __init__(
        self,
        *,
        extension_id: str,
        version: str,
        extension_path: Path,
    ) -> None:
        self.extension_id = extension_id
        self.version = version
        self.extension_path = extension_path
        self.enabled = False
        self.paths: list[str] = []
        self.enable_calls: list[tuple[str, bool]] = []

    def add_path(self, path: str) -> None:
        self.paths.append(path)

    def is_extension_enabled(self, extension_id: str) -> bool:
        assert extension_id == self.extension_id
        return self.enabled

    def set_extension_enabled_immediate(
        self,
        extension_id: str,
        enabled: bool,  # noqa: FBT001 - mirrors local Kit API.
    ) -> bool:
        self.enable_calls.append((extension_id, enabled))
        self.enabled = enabled
        return True

    def get_enabled_extension_id(self, extension_id: str) -> str | None:
        assert extension_id == self.extension_id
        return extension_id if self.enabled else None

    def get_extension_dict(self, enabled_id: str) -> dict[str, object]:
        assert enabled_id == self.extension_id
        return {"package": {"version": self.version}}

    def get_extension_path(self, enabled_id: str) -> str:
        assert enabled_id == self.extension_id
        return str(self.extension_path)


def test_probe_uses_local_grasp_editor_and_frozen_stage() -> None:
    source = PROBE.read_text(encoding="utf-8")
    assert "isaacsim.robot_setup.grasp_editor" in source
    assert "2.0.20" in source
    assert "aloha1_signal_correspondence_workcell.usda" in source
    assert (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
        in source
    )
    assert '"left_finger"' in source
    assert '"right_finger"' in source
    assert '"waist"' in source
    assert '"GUI_PENDING"' in source
    assert '"NOT_RUN"' in source
    assert '"structural_api_classification"' in source
    assert '"structural_api_probe_status"' in source
    assert '"synthetic_serializer_parse_probe"' in source


def test_generated_report_keeps_structural_and_behavioral_gates_separate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["status"] == "PARTIAL"
    assert report["classification"] == "INCONCLUSIVE"
    assert (
        report["structural_api_classification"]
        == "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"
    )
    assert report["structural_api_probe_status"] == "PASS"
    assert report["grasp_tester_execution_status"] == "NOT_RUN"
    assert report["timeline_physics_execution_status"] == "NOT_RUN"
    assert report["session_bottle500_composition_status"] == "NOT_RUN"
    assert report["arm_hold_during_grasp_test_status"] == "NOT_RUN"
    assert report["actual_isaac_grasp_export_status"] == "NOT_RUN"
    assert report["mimic_commandability_status"] == "GUI_PENDING"
    assert report["gui_evidence_status"] == "GUI_PENDING"
    assert (
        report["classification"]
        != report["structural_api_classification"]
    )
    serializer = report["probe"]["synthetic_serializer_parse_probe"]
    assert serializer["synthetic"] is True
    assert serializer["uses_grasp_tester_output"] is False
    assert serializer["exercises_gui_import_remap"] is False


def test_launcher_never_saves_the_source_stage() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "session_layer" in source
    assert "save_as_stage" not in source
    assert "save_stage" not in source


def test_launcher_uses_user_confirmed_aligned_stage_and_world_zero() -> None:
    module = _load_launcher_module()

    assert module.STAGE_PATH.resolve() == Path(
        "assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/"
        "aloha1_table_support_aligned_workcell.usda"
    ).resolve()
    assert (
        module.EXPECTED_STAGE_SHA256
        == "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
    )
    assert module.EXPECTED_ROOT_PRIM == "/World"
    assert module.EXPECTED_SUBLAYERS == (
        "configuration/aloha1_tabletop_world_zero.usda",
        "../../signal_correspondence/1.0/"
        "aloha1_signal_correspondence_workcell.usda",
    )
    assert module.REQUIRED_PRIM_PATHS == (
        "/World/follower_left/vx300s_left/root_joint",
        "/World/follower_right/vx300s_right/root_joint",
        "/World/environment/worldBody/user_confirmed_table",
    )
    assert module.TASK_FRAME_TRANSLATION_WORLD_M == (0.0, 0.0, 0.0)
    assert "-0.0909000015258789" not in LAUNCHER.read_text(encoding="utf-8")


def test_launcher_rejects_loaded_stage_contract_mismatch() -> None:
    module = _load_launcher_module()

    class _FakePathPrim:
        def __init__(self, path: str, *, valid: bool = True) -> None:
            self._path = path
            self._valid = valid

        def GetPath(self) -> str:  # noqa: N802 - mirrors USD API.
            return self._path

        def IsValid(self) -> bool:  # noqa: N802 - mirrors USD API.
            return self._valid

    class _FakeContractRootLayer:
        def __init__(self) -> None:
            self.subLayerPaths = list(module.EXPECTED_SUBLAYERS)

    class _FakeContractStage:
        def GetDefaultPrim(self) -> _FakePathPrim:  # noqa: N802
            return _FakePathPrim(module.EXPECTED_ROOT_PRIM)

        def GetRootLayer(self) -> _FakeContractRootLayer:  # noqa: N802
            return _FakeContractRootLayer()

        def GetPrimAtPath(self, path: str) -> _FakePathPrim:  # noqa: N802
            return _FakePathPrim(
                path,
                valid=path != module.REQUIRED_PRIM_PATHS[-1],
            )

    with pytest.raises(RuntimeError, match="required prim"):
        module._validate_loaded_stage_contract(  # noqa: SLF001
            _FakeContractStage()
        )


def test_launcher_enables_exact_runtime_extension_path_and_version(
    tmp_path: Path,
) -> None:
    module = _load_launcher_module()
    extension_parent = tmp_path / "extensions"
    extension_path = extension_parent / "isaac.sim.mcp_extension"
    extension_path.mkdir(parents=True)
    manager = _FakeExtensionManager(
        extension_id="isaac.sim.mcp_extension",
        version="0.4.1",
        extension_path=extension_path,
    )

    enabled_id, version = module._enable_extension_exact(  # noqa: SLF001
        manager,
        extension_id="isaac.sim.mcp_extension",
        expected_version="0.4.1",
        extension_parent=extension_parent,
        expected_extension_path=extension_path,
    )

    assert enabled_id == "isaac.sim.mcp_extension"
    assert version == "0.4.1"
    assert manager.paths == [str(extension_parent.resolve())]
    assert manager.enable_calls == [("isaac.sim.mcp_extension", True)]


def test_launcher_rejects_runtime_extension_version_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_launcher_module()
    extension_path = tmp_path / "isaac.sim.mcp_extension"
    extension_path.mkdir()
    manager = _FakeExtensionManager(
        extension_id="isaac.sim.mcp_extension",
        version="9.9.9",
        extension_path=extension_path,
    )

    with pytest.raises(RuntimeError, match="version mismatch"):
        module._enable_extension_exact(  # noqa: SLF001
            manager,
            extension_id="isaac.sim.mcp_extension",
            expected_version="0.4.1",
            expected_extension_path=extension_path,
        )


def test_launcher_adds_external_reference_with_local_usd_signature(
    tmp_path: Path,
) -> None:
    module = _load_launcher_module()
    calls: list[tuple[str, object]] = []

    class _FakeReferences:
        def AddReference(  # noqa: N802 - mirrors USD API.
            self,
            asset_path: str,
            prim_path: object,
        ) -> None:
            calls.append((asset_path, prim_path))

    class _FakePrim:
        def GetReferences(self) -> _FakeReferences:  # noqa: N802
            return _FakeReferences()

    asset_path = tmp_path / "bottle.usd"
    prim_path = object()
    module._add_external_reference(  # noqa: SLF001
        _FakePrim(),
        asset_path,
        prim_path,
    )

    assert calls == [(str(asset_path.resolve()), prim_path)]
    assert isinstance(calls[0][0], str)


def test_stage_guard_rejects_edit_target_escape() -> None:
    module = _load_launcher_module()
    diagnostic_layer = object()
    frozen_stage = _FakeStage(object())

    with pytest.raises(RuntimeError, match="edit target"):
        module._assert_frozen_diagnostic_stage(  # noqa: SLF001
            lambda: frozen_stage,
            frozen_stage,
            diagnostic_layer,
        )


def test_guarded_update_rejects_current_stage_replacement_after_final_update() -> None:
    module = _load_launcher_module()
    diagnostic_layer = object()
    frozen_stage = _FakeStage(diagnostic_layer)
    replacement_stage = _FakeStage(diagnostic_layer)
    context = {"stage": frozen_stage}

    class _ReplacingApp:
        def __init__(self) -> None:
            self.running = True

        def update(self) -> None:
            context["stage"] = replacement_stage
            self.running = False

        def is_running(self) -> bool:
            return self.running

    with pytest.raises(RuntimeError, match="current USD Stage"):
        module._guarded_app_update(  # noqa: SLF001
            _ReplacingApp(),
            lambda: context["stage"],
            frozen_stage,
            diagnostic_layer,
        )


def test_guarded_final_update_allows_current_stage_to_clear() -> None:
    module = _load_launcher_module()
    diagnostic_layer = object()
    frozen_stage = _FakeStage(diagnostic_layer)
    context = {"stage": frozen_stage}

    class _ClosingApp:
        def __init__(self) -> None:
            self.running = True

        def update(self) -> None:
            context["stage"] = None
            self.running = False

        def is_running(self) -> bool:
            return self.running

    module._guarded_app_update(  # noqa: SLF001
        _ClosingApp(),
        lambda: context["stage"],
        frozen_stage,
        diagnostic_layer,
    )


def test_restore_verifies_exact_previous_edit_target_layer() -> None:
    module = _load_launcher_module()
    diagnostic_layer = object()
    previous_layer = object()
    stage = _FakeStage(diagnostic_layer)

    module._restore_previous_edit_target(  # noqa: SLF001
        stage,
        _FakeEditTarget(previous_layer),
    )

    assert stage.GetEditTarget().GetLayer() is previous_layer


def test_restore_rejects_silent_failure_to_change_target() -> None:
    module = _load_launcher_module()
    stage = _FakeStage(
        object(),
        silent_restore_failure=True,
    )

    with pytest.raises(RuntimeError, match="exact previous edit target layer"):
        module._restore_previous_edit_target(  # noqa: SLF001
            stage,
            _FakeEditTarget(object()),
        )


def test_cleanup_attempts_all_gates_after_restore_and_removal_failures(
    tmp_path: Path,
) -> None:
    module = _load_launcher_module()
    events: list[str] = []
    diagnostic_identifier = "anon:ALOHA1GraspEditorDiagnostic"
    stage_path = tmp_path / "stage.usda"
    bottle_path = tmp_path / "bottle.usd"
    hashes = {
        stage_path: "stage hash",
        bottle_path: "bottle hash",
    }
    stage = _FakeStage(
        object(),
        events,
        restore_failure=True,
    )

    def fake_sha256(path: Path) -> str:
        events.append(f"hash:{path.name}")
        return hashes[path]

    with pytest.raises(BaseExceptionGroup) as caught:
        module._cleanup_diagnostic_session(  # noqa: SLF001
            stage=stage,
            previous_edit_target=_FakeEditTarget(object()),
            session_layer=_FakeSessionLayer(
                _FakeSubLayerPaths(
                    diagnostic_identifier,
                    events,
                    removal_failure=True,
                )
            ),
            diagnostic_layer_identifier=diagnostic_identifier,
            root_layer=_FakeRootLayer(events),
            root_dirty_before=False,
            root_specs_before="frozen root specs",
            app=_FakeApp(events),
            stage_path=stage_path,
            source_stage_sha256_before="stage hash",
            bottle_path=bottle_path,
            bottle_sha256_before="bottle hash",
            sha256=fake_sha256,
        )

    assert events == [
        "restore",
        "remove",
        "root_dirty",
        "root_specs",
        "close",
        "hash:stage.usda",
        "hash:bottle.usd",
    ]
    assert len(caught.value.exceptions) == 2
    assert caught.value.exceptions[0] is stage.restore_error
    assert type(caught.value.exceptions[0]) is RuntimeError
    assert "restore previous edit target" in "\n".join(
        caught.value.exceptions[0].__notes__
    )
    assert "remove anonymous session sublayer" in "\n".join(
        caught.value.exceptions[1].__notes__
    )


def test_cleanup_preserves_primary_exception_and_chains_cleanup_errors(
    tmp_path: Path,
) -> None:
    module = _load_launcher_module()
    events: list[str] = []
    diagnostic_identifier = "anon:ALOHA1GraspEditorDiagnostic"
    stage_path = tmp_path / "stage.usda"
    bottle_path = tmp_path / "bottle.usd"
    primary_error = RuntimeError("primary runtime failure")

    def fake_sha256(path: Path) -> str:
        events.append(f"hash:{path.name}")
        return "frozen"

    with pytest.raises(RuntimeError, match="primary runtime failure") as caught:
        module._cleanup_diagnostic_session(  # noqa: SLF001
            stage=_FakeStage(
                object(),
                events,
                restore_failure=True,
            ),
            previous_edit_target=_FakeEditTarget(object()),
            session_layer=_FakeSessionLayer(
                _FakeSubLayerPaths(diagnostic_identifier, events)
            ),
            diagnostic_layer_identifier=diagnostic_identifier,
            root_layer=_FakeRootLayer(events),
            root_dirty_before=False,
            root_specs_before="frozen root specs",
            app=_FakeApp(events),
            stage_path=stage_path,
            source_stage_sha256_before="frozen",
            bottle_path=bottle_path,
            bottle_sha256_before="frozen",
            sha256=fake_sha256,
            primary_error=primary_error,
            primary_traceback=primary_error.__traceback__,
        )

    assert caught.value is primary_error
    assert isinstance(caught.value.__cause__, BaseExceptionGroup)
    assert events == [
        "restore",
        "remove",
        "root_dirty",
        "root_specs",
        "close",
        "hash:stage.usda",
        "hash:bottle.usd",
    ]


def test_cleanup_preserves_existing_primary_cause_and_retains_group(
    tmp_path: Path,
) -> None:
    module = _load_launcher_module()
    events: list[str] = []
    diagnostic_identifier = "anon:ALOHA1GraspEditorDiagnostic"
    stage_path = tmp_path / "stage.usda"
    bottle_path = tmp_path / "bottle.usd"
    original_cause = ValueError("pre-existing cause")
    primary_error = RuntimeError("primary runtime failure")
    primary_error.__cause__ = original_cause
    stage = _FakeStage(object(), events, restore_failure=True)

    def fake_sha256(path: Path) -> str:
        events.append(f"hash:{path.name}")
        return "frozen"

    with pytest.raises(RuntimeError) as caught:
        module._cleanup_diagnostic_session(  # noqa: SLF001
            stage=stage,
            previous_edit_target=_FakeEditTarget(object()),
            session_layer=_FakeSessionLayer(
                _FakeSubLayerPaths(diagnostic_identifier, events)
            ),
            diagnostic_layer_identifier=diagnostic_identifier,
            root_layer=_FakeRootLayer(events),
            root_dirty_before=False,
            root_specs_before="frozen root specs",
            app=_FakeApp(events),
            stage_path=stage_path,
            source_stage_sha256_before="frozen",
            bottle_path=bottle_path,
            bottle_sha256_before="frozen",
            sha256=fake_sha256,
            primary_error=primary_error,
            primary_traceback=primary_error.__traceback__,
        )

    assert caught.value is primary_error
    assert caught.value.__cause__ is original_cause
    cleanup_group = caught.value.grasp_editor_cleanup_errors
    assert isinstance(cleanup_group, ExceptionGroup)
    assert cleanup_group.exceptions == (stage.restore_error,)
    assert any(
        "grasp_editor_cleanup_errors" in note
        for note in caught.value.__notes__
    )


@pytest.mark.parametrize(
    "exit_error",
    [KeyboardInterrupt(), SystemExit(7)],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_cleanup_does_not_convert_base_exit(
    exit_error: BaseException,
) -> None:
    module = _load_launcher_module()
    events: list[str] = []

    def interrupting_step() -> None:
        events.append("interrupt")
        raise exit_error

    with pytest.raises(type(exit_error)) as caught:
        module._run_cleanup_steps(  # noqa: SLF001
            [
                ("interrupt", interrupting_step),
                ("later", lambda: events.append("later")),
            ]
        )

    assert caught.value is exit_error
    assert events == ["interrupt"]


def test_launcher_keeps_grasp_editor_authored_opinions_in_anonymous_layer() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Sdf.Layer.CreateAnonymous(" in source
    assert '"ALOHA1GraspEditorDiagnostic"' in source
    assert "previous_edit_target = stage.GetEditTarget()" in source
    assert (
        "session_layer.subLayerPaths.append(diagnostic_layer_identifier)"
        in source
    )
    assert "stage.SetEditTarget(diagnostic_layer)" in source
    assert "root_dirty_before = root_layer.dirty" in source
    assert "root_specs_before = root_layer.ExportToString()" in source
    assert "while app.is_running():\n            _guarded_app_update(" in source
    assert "bottle_sha256_before = _sha256(bottle_path)" in source
    assert "diagnostic_layer.Save(" not in source
    assert "_add_external_reference(" in source
    assert "str(asset_path.resolve())" in source
    assert 'Sdf.Path("/Bottle500")' in source


def test_structural_classification_accepts_only_exact_contract() -> None:
    module = _load_probe_module()
    expected = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ]
    assert (
        module.classify_structural_api_compatibility(
            extension_version="2.0.20",
            dof_names=expected,
            active_joint_names=["left_finger", "right_finger"],
            structural_setup_arm_joint_mutation=False,
            synthetic_serializer_parse_pass=True,
            stage_immutable=True,
        )
        == "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"extension_version": "2.2.0"}, "INCOMPATIBLE"),
        ({"dof_names": ["left_finger", "right_finger"]}, "INCOMPATIBLE"),
        ({"active_joint_names": ["left_finger"]}, "INCOMPATIBLE"),
        (
            {"structural_setup_arm_joint_mutation": True},
            "REQUIRES_DIAGNOSTIC_GRIPPER_ONLY",
        ),
        ({"synthetic_serializer_parse_pass": False}, "INCOMPATIBLE"),
        ({"stage_immutable": False}, "INCOMPATIBLE"),
    ],
)
def test_classification_fails_closed(
    overrides: dict[str, object],
    expected: str,
) -> None:
    module = _load_probe_module()
    values: dict[str, object] = {
        "extension_version": "2.0.20",
        "dof_names": list(module.EXPECTED_DOF_NAMES),
        "active_joint_names": ["left_finger", "right_finger"],
        "structural_setup_arm_joint_mutation": False,
        "synthetic_serializer_parse_pass": True,
        "stage_immutable": True,
    }
    values.update(overrides)
    assert module.classify_structural_api_compatibility(**values) == expected
