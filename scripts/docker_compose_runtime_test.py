from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    with (ROOT / "docker-compose.yml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_webrtc_media_service_starts_with_default_robot_stack():
    services = _compose()["services"]
    service = services["eii_pilot_webrtc_media"]

    assert "profiles" not in service
    assert "eii_pilot_webrtc_media" in services["eii_pilot_frontend"]["depends_on"]


def test_operator_core_start_command_includes_webrtc_media_service():
    workflow = (ROOT / "docs/rlt_online_operator_workflow.md").read_text(encoding="utf-8")
    start_section = workflow.split("## 2. Start Warmup/Online Runtime", maxsplit=1)[0]

    assert "eii_pilot_webrtc_media" in start_section


def test_openpi_server_skips_non_rtc_warmup_by_default():
    service = _compose()["services"]["openpi_server"]
    command = " ".join(service["command"])

    assert "--no-warmup-non-rtc" in command


def test_openpi_server_uses_project_local_jax_compilation_cache():
    service = _compose()["services"]["openpi_server"]

    assert "${OPENPI_JAX_CACHE_DIR:-./.jax_cache/openpi_server}:/app/.jax_cache" in service["volumes"]
    assert "JAX_COMPILATION_CACHE_DIR=/app/.jax_cache" in service["environment"]
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0" in service["environment"]
    assert "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1" in service["environment"]
    assert "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir" in service["environment"]
