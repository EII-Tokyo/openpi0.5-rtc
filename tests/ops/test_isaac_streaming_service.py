from __future__ import annotations

import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
UNIT = PROJECT_ROOT / "ops" / "isaac-frp" / "isaac-sim-streaming.service"


def test_streaming_service_is_full_headless_on_demand() -> None:
    text = UNIT.read_text(encoding="utf-8")

    assert "WorkingDirectory=/home/eii/Applications/isaacsim-5.1.0" in text
    assert (
        "ExecStart=/home/eii/Applications/isaacsim-5.1.0/isaac-sim.streaming.sh "
        "--/app/livestream/publicEndpointAddress=127.0.0.1 "
        "--/app/livestream/port=49100"
    ) in text
    assert "Restart=on-failure" in text
    assert "Restart=always" not in text
    assert "WantedBy=default.target" in text
    assert "isaac-sim.sh " not in text
