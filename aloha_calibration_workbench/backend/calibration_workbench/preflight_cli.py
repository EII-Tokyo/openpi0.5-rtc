from __future__ import annotations

import argparse
from pathlib import Path

from .preflight import PreflightService
from .registry import load_candidate_registry
from .rs_cli_probe import RsEnumerateCliProbe

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "third_party/aloha_collection/config/robot/aloha_stationary.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only ALOHA D405 identity preflight")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    args = parser.parse_args()
    registry = load_candidate_registry(args.robot_config)
    report = PreflightService(registry=registry, probe=RsEnumerateCliProbe(registry.profile)).run()
    print(report.model_dump_json(indent=2))
