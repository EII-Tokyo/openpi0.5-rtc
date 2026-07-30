from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL = Path(
    "tools/annotate_aloha1_grasp_editor_external_skip_sim.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "grasp_editor_external_skip_sim_screenshots",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visual_scope_never_promotes_context_image_to_numeric_pass() -> None:
    module = _load_module()

    assert module.visual_scope_for_phase(
        "CONFIGURED_BEFORE_SIMULATE"
    ) == {
        "visual_scope": "FULL_ARM_CONTEXT_OPEN",
        "acceptance": "PASS_CONTEXT_ONLY",
    }
    assert module.visual_scope_for_phase(
        "EXTERNAL_CONTACT_SKIP_SIM_RESULT_CLOSEUP"
    ) == {
        "visual_scope": "BILATERAL_CONTACT_CLOSEUP",
        "acceptance": "PASS_VISUAL_CONTACT_STATE_NUMERIC_MIMIC_FAIL",
    }
