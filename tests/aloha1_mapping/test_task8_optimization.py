from tools.aloha1_mapping.task8_optimization import build_inventory_summary
from tools.aloha1_mapping.task8_optimization import build_material_dedup_plan
from tools.aloha1_mapping.task8_optimization import build_model_first_gate
from tools.aloha1_mapping.task8_optimization import build_protected_signature
from tools.aloha1_mapping.task8_optimization import build_task8_progression_gate
from tools.aloha1_mapping.task8_optimization import compare_lower_is_better
from tools.aloha1_mapping.task8_optimization import failure_evidence_contract
from tools.aloha1_mapping.task8_optimization import rank_optimization_opportunities
from tools.aloha1_mapping.task8_optimization import summarize_numeric_samples
from tools.audit_aloha1_task8_baseline import canonical_layer_identifier
from tools.audit_aloha1_task8_baseline import start_usd_runtime_if_needed
from tools.benchmark_aloha1_task8_stage import _measurement_dict
from tools.build_aloha1_task8_progression_authorization import build_report
from tools.build_aloha1_task8_visual_material_candidate import _bound_visual_material_count


def _mesh(
    path: str,
    geometry: str,
    *,
    collision: bool = False,
    points: int = 10,
    faces: int = 6,
) -> dict[str, object]:
    return {
        "path": path,
        "geometry_signature": geometry,
        "is_collision": collision,
        "point_count": points,
        "face_count": faces,
        "material_path": None,
        "is_instance_proxy": False,
        "is_instanceable": False,
    }


def test_inventory_separates_visual_and_collision_duplicates() -> None:
    records = [
        _mesh("/left/visual", "finger-left"),
        _mesh("/right/visual", "finger-left"),
        _mesh("/left/collision", "arm", collision=True, points=20, faces=12),
        _mesh("/right/collision", "arm", collision=True, points=20, faces=12),
    ]

    summary = build_inventory_summary(
        mesh_records=records,
        material_records=[],
        prim_type_counts={"Mesh": 4, "Xform": 2},
        composition_records=[],
    )

    assert summary["mesh_count"] == 4
    assert summary["visual_mesh_count"] == 2
    assert summary["collision_mesh_count"] == 2
    assert summary["point_count"] == 60
    assert summary["face_count"] == 36
    assert summary["repeated_visual_geometry_groups"] == 1
    assert summary["repeated_collision_geometry_groups"] == 1
    assert summary["repeated_visual_mesh_instances"] == 2
    assert summary["repeated_collision_mesh_instances"] == 2
    assert summary["distinct_bound_visual_material_count"] == 0


def test_inventory_counts_distinct_effective_visual_material_bindings() -> None:
    records = [
        {**_mesh("/left/visual", "left"), "material_path": "/Looks/shared"},
        {**_mesh("/right/visual", "right"), "material_path": "/Looks/shared"},
        {**_mesh("/bottle/visual", "bottle"), "material_path": "/Looks/bottle"},
    ]

    summary = build_inventory_summary(
        mesh_records=records,
        material_records=[],
        prim_type_counts={"Mesh": 3},
        composition_records=[],
    )

    assert summary["distinct_bound_visual_material_count"] == 2
    assert summary["bound_visual_material_paths"] == ["/Looks/bottle", "/Looks/shared"]


def test_material_dedup_plan_targets_robot_visual_instance_roots_only() -> None:
    meshes = [
        {
            **_mesh("/World/follower_left/link/visuals/base/mesh", "base"),
            "material_path": "/World/follower_left/link/visuals/base/Looks/Default",
        },
        {
            **_mesh("/World/follower_right/link/visuals/base/mesh", "base"),
            "material_path": "/World/follower_right/link/visuals/base/Looks/Default",
        },
        {
            **_mesh("/World/environment/table/mesh", "table"),
            "material_path": "/World/environment/Looks/Default",
        },
    ]
    groups = [
        {
            "material_signature": "same",
            "paths": [
                "/World/environment/Looks/Default",
                "/World/follower_left/link/visuals/base/Looks/Default",
                "/World/follower_right/link/visuals/base/Looks/Default",
            ],
        }
    ]

    plan = build_material_dedup_plan(mesh_records=meshes, duplicate_material_groups=groups)

    assert plan == [
        {
            "visual_root": "/World/follower_left/link/visuals",
            "canonical_material": "/World/environment/Looks/Default",
            "source_material": "/World/follower_left/link/visuals/base/Looks/Default",
            "material_signature": "same",
            "representative_mesh": "/World/follower_left/link/visuals/base/mesh",
        },
        {
            "visual_root": "/World/follower_right/link/visuals",
            "canonical_material": "/World/environment/Looks/Default",
            "source_material": "/World/follower_right/link/visuals/base/Looks/Default",
            "material_signature": "same",
            "representative_mesh": "/World/follower_right/link/visuals/base/mesh",
        },
    ]


def test_opportunity_ranking_prefers_visual_only_and_defers_collision() -> None:
    summary = {
        "repeated_visual_geometry_groups": 2,
        "repeated_visual_mesh_instances": 4,
        "repeated_collision_geometry_groups": 18,
        "repeated_collision_mesh_instances": 36,
        "duplicate_material_groups": 0,
        "instanceable_prim_count": 44,
        "payload_prim_count": 3,
    }

    opportunities = rank_optimization_opportunities(
        summary,
        known_hydra_instance_regression=True,
    )

    assert opportunities[0]["id"] == "deduplicate_repeated_visual_geometry"
    assert opportunities[0]["risk"] == "MEDIUM_HYDRA_REGRESSION_KNOWN"
    collision = next(
        item for item in opportunities if item["id"] == "deduplicate_collision_geometry"
    )
    assert collision["decision"] == "DEFER_UNTIL_VISUAL_CANDIDATE_EVALUATED"
    assert collision["changes_physics_composition"] is True
    payload = next(item for item in opportunities if item["id"] == "add_payloads")
    assert payload["decision"] == "NO_ACTION_ALREADY_PRESENT"


def test_protected_signature_ignores_visual_records_but_detects_physics_change() -> None:
    baseline = {
        "joints": [{"path": "/robot/joint", "lower": -1.0, "upper": 1.0}],
        "colliders": [{"path": "/robot/collider", "geometry": "abc"}],
        "rigid_bodies": [{"path": "/robot/link", "mass": 1.0}],
        "articulations": [{"path": "/robot"}],
        "visuals": [{"path": "/robot/visual", "geometry": "before"}],
    }
    visual_candidate = {
        **baseline,
        "visuals": [{"path": "/robot/visual_task8", "geometry": "after"}],
    }
    physics_candidate = {
        **baseline,
        "colliders": [{"path": "/robot/collider", "geometry": "changed"}],
    }

    assert build_protected_signature(baseline) == build_protected_signature(
        visual_candidate
    )
    assert build_protected_signature(baseline) != build_protected_signature(
        physics_candidate
    )


def test_protected_signature_preserves_nonfinite_usd_values_as_tokens() -> None:
    inventory = {
        "joints": [{"path": "/joint", "upper": float("inf")}],
        "colliders": [],
        "rigid_bodies": [],
        "articulations": [],
    }

    signature = build_protected_signature(inventory)

    assert len(signature) == 64


def test_every_reproducible_task8_failure_requires_images_and_video() -> None:
    contract = failure_evidence_contract(reproducible=True)

    assert contract["raw_screenshots"] == [
        "before_anomaly",
        "first_anomalous_frame",
        "final_failure",
    ]
    assert contract["annotated_screenshots"] == contract["raw_screenshots"]
    assert contract["full_arm_collision_enabled_video_required"] is True
    assert contract["visual_review_required"] is True
    assert contract["machine_telemetry_required"] is True


def test_model_first_gate_requires_every_mathematical_and_runtime_contract() -> None:
    complete = {
        gate_id: {"status": "PASS"}
        for gate_id in (
            "source_audit",
            "parameter_matrix",
            "kinematic_contract",
            "dynamics_contract",
            "gripper_geometry_contract",
            "collider_geometry_contract",
            "runtime_contract",
        )
    }

    assert build_model_first_gate(complete)["candidate_authoring_allowed"] is True
    complete["dynamics_contract"] = {"status": "PARTIAL"}
    gate = build_model_first_gate(complete)
    assert gate["status"] == "BLOCKED"
    assert gate["candidate_authoring_allowed"] is False
    assert gate["blocking_gates"][0]["id"] == "dynamics_contract"


def test_task8_progression_keeps_calibration_gaps_as_nonblocking_reminders() -> None:
    gate = build_task8_progression_gate(
        runtime_grasp_status="PASS",
        finger_safety_status="PASS",
        model_first_status="PARTIAL_MODEL_PROOF",
        known_issues=[
            {
                "id": "finite_contact_patch_miss",
                "status": "KNOWN_LIMITATION",
                "summary": "Bottle500 central tangent is outside one rejected diagnostic patch.",
            },
            {
                "id": "uncalibrated_material_pair",
                "status": "TEMPORARY_UNCALIBRATED",
                "summary": "Physical finger/bottle material coefficients are not measured.",
            },
        ],
    )

    assert gate["status"] == "AUTHORIZED_IN_PROGRESS"
    assert gate["isolated_candidate_authoring_allowed"] is True
    assert gate["final_default_promotion_allowed"] is False
    assert gate["sim_to_real_calibrated_claim_allowed"] is False
    assert len(gate["known_issue_reminders"]) == 2
    assert all(not item["blocking_task8"] for item in gate["known_issue_reminders"])


def test_task8_progression_still_blocks_when_runtime_functional_gate_failed() -> None:
    gate = build_task8_progression_gate(
        runtime_grasp_status="FAIL",
        finger_safety_status="PASS",
        model_first_status="PARTIAL_MODEL_PROOF",
        known_issues=[],
    )

    assert gate["status"] == "BLOCKED_BY_FUNCTIONAL_BASELINE"
    assert gate["isolated_candidate_authoring_allowed"] is False


def test_task8_progression_report_preserves_history_and_skips_extra_evidence() -> None:
    report = build_report()

    assert report["status"] == "AUTHORIZED_IN_PROGRESS"
    assert report["policy"]["additional_contact_patch_screenshots"] == "NOT_REQUESTED"
    assert report["policy"]["repeat_five_grasp_videos"] == "NOT_REQUIRED_BY_DEFAULT"
    assert report["history_preserved"] == {
        "strict_model_first_report_rewritten": False,
        "rejected_compound_candidate_promoted": False,
        "final_or_default_asset_modified": False,
    }


def test_task8_static_audit_starts_headless_runtime_only_when_pxr_is_missing() -> None:
    calls: list[dict[str, object]] = []

    class FakeApp:
        pass

    def factory(config: dict[str, object]) -> FakeApp:
        calls.append(config)
        return FakeApp()

    app = start_usd_runtime_if_needed(pxr_available=False, app_factory=factory)
    assert isinstance(app, FakeApp)
    assert calls == [{"headless": True, "create_new_stage": False}]

    calls.clear()
    assert start_usd_runtime_if_needed(pxr_available=True, app_factory=factory) is None
    assert calls == []


def test_task8_inventory_canonicalizes_process_local_session_layer_address() -> None:
    identifier = "anon:0x601bd15f9a00:frozen-stage-session.usda"

    assert canonical_layer_identifier(identifier) == "anon:<session>:frozen-stage-session.usda"
    assert canonical_layer_identifier("/tmp/frozen-stage.usda") == "/tmp/frozen-stage.usda"


def test_material_candidate_supports_baseline_inventory_before_bound_count_field() -> None:
    inventory = {
        "summary": {},
        "meshes": [
            {"is_collision": False, "material_path": "/Looks/shared"},
            {"is_collision": False, "material_path": "/Looks/shared"},
            {"is_collision": True, "material_path": "/Looks/physics"},
        ],
    }

    assert _bound_visual_material_count(inventory) == 1


def test_task8_benchmark_sample_summary_records_variance() -> None:
    summary = summarize_numeric_samples([10.0, 12.0, 14.0])

    assert summary == {
        "count": 3,
        "mean": 12.0,
        "stdev": 2.0,
        "min": 10.0,
        "max": 14.0,
    }


def test_task8_benchmark_serializes_unitless_list_measurements() -> None:
    measurement = type("ListMeasurement", (), {"name": "samples", "value": [1.0]})()
    data = type("MeasurementData", (), {"measurements": [measurement]})()

    assert _measurement_dict(data) == {"samples": {"value": [1.0], "unit": ""}}


def test_task8_benchmark_comparison_uses_nonoverlapping_fresh_process_ranges() -> None:
    improved = compare_lower_is_better([10.0, 11.0], [7.0, 8.0])
    worsened = compare_lower_is_better([10.0, 11.0], [12.0, 13.0])
    overlap = compare_lower_is_better([10.0, 12.0], [11.0, 13.0])

    assert improved["classification"] == "IMPROVES_NONOVERLAPPING_RANGE"
    assert worsened["classification"] == "WORSENS_NONOVERLAPPING_RANGE"
    assert overlap["classification"] == "INCONCLUSIVE_OVERLAPPING_RANGE"
