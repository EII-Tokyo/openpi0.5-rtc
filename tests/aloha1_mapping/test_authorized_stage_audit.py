from __future__ import annotations

import sys
from types import SimpleNamespace

from tools.aloha1_mapping.authorized_stage_audit import _mesh_record
from tools.aloha1_mapping.authorized_stage_audit import (
    EXPECTED_REVIEW_STAGE_SHA256,
)
from tools.aloha1_mapping.authorized_stage_audit import (
    REQUIRED_REVIEW_STAGE_PRIMS,
)
from tools.aloha1_mapping.authorized_stage_audit import evaluate_stage_snapshot


def _valid_snapshot() -> dict:
    return {
        "absolute_path": (
            "/home/eii/project/openpi0.5-rtc-reward-learning/"
            "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
        ),
        "source_sha256_before": EXPECTED_REVIEW_STAGE_SHA256,
        "source_sha256_after": EXPECTED_REVIEW_STAGE_SHA256,
        "default_prim": "/workcell",
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "required_prims": {
            path: {"valid": True, "type_name": "Xform"}
            for path in REQUIRED_REVIEW_STAGE_PRIMS
        },
        "used_layers": [
            {
                "absolute_path": (
                    "/tmp/aloha_viperx/configuration/"
                    "aloha_viperx_physics.usd"
                ),
                "exists": True,
                "sha256": "a" * 64,
            },
            {
                "absolute_path": (
                    "/tmp/aloha_viperx/configuration/"
                    "aloha_viperx_base.usd"
                ),
                "exists": True,
                "sha256": "b" * 64,
            },
            {
                "absolute_path": (
                    "/tmp/aloha_viperx/configuration/"
                    "aloha_viperx_robot.usd"
                ),
                "exists": True,
                "sha256": "c" * 64,
            },
        ],
        "finger_branches": {
            "left": {
                "visuals_instanceable": True,
                "visual_mesh_is_instance_proxy": True,
                "collisions_instanceable": True,
                "collision_mesh_is_instance_proxy": True,
            },
            "right": {
                "visuals_instanceable": True,
                "visual_mesh_is_instance_proxy": True,
                "collisions_instanceable": True,
                "collision_mesh_is_instance_proxy": True,
            },
        },
    }


def test_authorized_stage_snapshot_passes_only_when_source_is_unchanged() -> None:
    report = evaluate_stage_snapshot(_valid_snapshot())

    assert report["status"] == "PASS"
    assert report["source_immutable_gate"] == "PASS"
    assert report["root_prim_gate"] == "PASS"
    assert report["required_key_prims_status"] == "PASS"
    assert report["layer_stack_status"] == "PASS"
    assert report["instance_proxy_strategy"] == (
        "DEINSTANCE_VISUAL_BRANCH_IN_DIAGNOSTIC_LAYER_ONLY"
    )


def test_authorized_stage_snapshot_fails_on_missing_finger_joint() -> None:
    snapshot = _valid_snapshot()
    missing_path = "/workcell/joints/vx300s_left_right_finger"
    snapshot["required_prims"][missing_path]["valid"] = False

    report = evaluate_stage_snapshot(snapshot)

    assert report["status"] == "FAIL"
    assert report["required_key_prims_status"] == "FAIL"
    assert missing_path in report["missing_required_prims"]


def test_authorized_stage_snapshot_fails_on_source_hash_change() -> None:
    snapshot = _valid_snapshot()
    snapshot["source_sha256_after"] = "0" * 64

    report = evaluate_stage_snapshot(snapshot)

    assert report["status"] == "FAIL"
    assert report["source_immutable_gate"] == "FAIL"


def test_mesh_record_uses_local_usd_prim_range(monkeypatch) -> None:
    calls = []

    class FakeUsd:
        @staticmethod
        def TraverseInstanceProxies():
            return "TRAVERSE_INSTANCE_PROXIES"

        @staticmethod
        def PrimRange(branch, predicate=None):
            calls.append((branch, predicate))
            return []

    fake_pxr = SimpleNamespace(
        Usd=FakeUsd,
        UsdGeom=SimpleNamespace(Mesh=object),
    )
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    branch = object()
    stage = SimpleNamespace(GetPrimAtPath=lambda _: branch)

    record = _mesh_record(stage, "/finger/visuals")

    assert record["mesh_count"] == 0
    assert calls == [(branch, "TRAVERSE_INSTANCE_PROXIES")]
