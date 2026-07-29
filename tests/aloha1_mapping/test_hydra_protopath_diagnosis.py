from tools.aloha1_mapping.hydra_protopath_diagnosis import PROTOPATH_SETTINGS
from tools.aloha1_mapping.hydra_protopath_diagnosis import build_variant_matrix
from tools.aloha1_mapping.hydra_protopath_diagnosis import classify_diagnosis
from tools.aloha1_mapping.hydra_protopath_diagnosis import parse_protopath_errors


def test_variant_matrix_only_uses_runtime_supported_settings() -> None:
    supported = {
        path: path
        not in {
            PROTOPATH_SETTINGS["intermediate_instance_proxy"],
            PROTOPATH_SETTINGS["renderer_instancing"],
        }
        for path in PROTOPATH_SETTINGS.values()
    }

    variants = build_variant_matrix(supported)

    assert [item["id"] for item in variants] == ["A", "B", "C1", "C2", "C3", "D"]
    assert variants[0]["setting_overrides"] == {}
    assert variants[-1]["materialize_visual_instances"] is True
    assert all(len(item["setting_overrides"]) <= 1 for item in variants if item["id"] not in {"A", "D"})
    assert all(
        path not in item["setting_overrides"]
        for item in variants
        for path in {
            PROTOPATH_SETTINGS["intermediate_instance_proxy"],
            PROTOPATH_SETTINGS["renderer_instancing"],
        }
    )


def test_parse_protopath_errors_counts_total_and_unique_pairs() -> None:
    text = "\n".join(
        [
            "[Error] Instance /World/A cannot find protoPath /__Prototype_1",
            "[Error] Instance /World/A cannot find protoPath /__Prototype_1",
            "[Error] Instance /World/B cannot find protoPath /__Prototype_2",
        ]
    )

    parsed = parse_protopath_errors(text)

    assert parsed["total_count"] == 3
    assert parsed["unique_pair_count"] == 2
    assert parsed["unique_pairs"] == [
        {
            "instance_path": "/World/A",
            "prototype_path": "/__Prototype_1",
        },
        {
            "instance_path": "/World/B",
            "prototype_path": "/__Prototype_2",
        },
    ]


def test_classification_prefers_omnihydra_when_b_alone_repairs_native_render() -> None:
    variants = [
        {"id": "A", "native_render_complete": False, "proto_error_count": 29},
        {"id": "B", "native_render_complete": True, "proto_error_count": 0},
        {"id": "D", "native_render_complete": True, "proto_error_count": 0},
    ]

    assert classify_diagnosis(variants) == {
        "classification": "FSD_7_5_1_PRIMARY",
        "effective_variant": "B",
        "effective_setting": PROTOPATH_SETTINGS["use_fabric_scene_delegate"],
    }


def test_classification_names_the_only_effective_population_option() -> None:
    variants = [
        {"id": "A", "native_render_complete": False, "proto_error_count": 29},
        {"id": "B", "native_render_complete": False, "proto_error_count": 29},
        {
            "id": "C1",
            "native_render_complete": True,
            "proto_error_count": 0,
            "setting_overrides": {PROTOPATH_SETTINGS["single_threaded"]: True},
        },
        {"id": "C2", "native_render_complete": False, "proto_error_count": 29},
        {"id": "D", "native_render_complete": True, "proto_error_count": 0},
    ]

    assert classify_diagnosis(variants) == {
        "classification": "USD_TO_FABRIC_POPULATION_OPTION",
        "effective_variant": "C1",
        "effective_setting": PROTOPATH_SETTINGS["single_threaded"],
    }


def test_classification_uses_instance_authoring_only_when_d_repairs_render() -> None:
    variants = [
        {"id": "A", "native_render_complete": False, "proto_error_count": 29},
        {"id": "B", "native_render_complete": False, "proto_error_count": 29},
        {"id": "C1", "native_render_complete": False, "proto_error_count": 29},
        {"id": "D", "native_render_complete": True, "proto_error_count": 0},
    ]

    assert classify_diagnosis(variants) == {
        "classification": "INSTANCE_AUTHORING_STRUCTURE",
        "effective_variant": "D",
        "effective_setting": None,
    }


def test_classification_detects_reset_repopulation_boundary() -> None:
    variants = [
        {
            "id": "A",
            "native_render_complete": False,
            "proto_error_count": 29,
            "native_render_complete_without_reopen": True,
            "proto_error_count_without_reopen": 0,
        },
        {"id": "D", "native_render_complete": False, "proto_error_count": 29},
    ]

    assert classify_diagnosis(variants) == {
        "classification": "RESET_REPOPULATION_RACE",
        "effective_variant": "A_WITHOUT_REOPEN",
        "effective_setting": None,
    }
