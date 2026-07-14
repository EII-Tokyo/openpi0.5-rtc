from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MJCF = REPO_ROOT / "local_eval_assets/aloha_workcell_minimal/model/workcell.xml"
DEFAULT_OUTPUT = REPO_ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"


def convert_mjcf_to_usd(mjcf_path: Path, output_usd: Path, force: bool) -> Path:
    """Convert a MJCF file to USD using Isaac Lab's official converter wrapper."""
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF file does not exist: {mjcf_path}")

    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise SystemExit(
            "Isaac Lab AppLauncher is not importable in this Python environment. "
            "Install Isaac Sim and Isaac Lab, then rerun this script."
        ) from exc

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        from isaacsim.core.utils.extensions import enable_extension
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

        enable_extension("isaacsim.asset.importer.mjcf")
        cfg = MjcfConverterCfg(
            asset_path=str(mjcf_path.resolve()),
            usd_dir=str(output_usd.parent.resolve()),
            usd_file_name=output_usd.name,
            force_usd_conversion=force,
            make_instanceable=False,
            fix_base=True,
            import_sites=True,
            self_collision=False,
        )
        converter = MjcfConverter(cfg)
        return Path(converter.usd_path)
    finally:
        simulation_app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the local ALOHA MJCF workcell to USD with Isaac Lab.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output-usd", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = convert_mjcf_to_usd(args.mjcf, args.output_usd, args.force)
    print(f"usd={output}")


if __name__ == "__main__":
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    main()
