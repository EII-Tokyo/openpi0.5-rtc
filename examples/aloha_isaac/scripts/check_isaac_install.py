from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import sys


def _module_status(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "MISSING"
    return f"OK ({spec.origin})"


def _simulation_app_status() -> tuple[bool, str]:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
        try:
            import omni.usd
            from pxr import Usd  # noqa: F401

            if omni.usd.get_context() is None:
                return False, "SimulationApp started, but omni.usd context is unavailable."
            return True, "SimulationApp, omni.usd, and pxr are available after Kit startup."
        finally:
            app.close()
    except Exception as exc:
        return False, f"SimulationApp startup failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Isaac Sim / Isaac Lab availability in the active Python.")
    parser.add_argument("--launch-kit", action="store_true", help="Start headless SimulationApp for a real Kit import check.")
    args = parser.parse_args()

    print("Python:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    try:
        glibc = subprocess.check_output(["ldd", "--version"], text=True).splitlines()[0]
        print("glibc:", glibc)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print("glibc: unavailable", exc)

    modules = [
        "isaacsim",
        "isaaclab",
        "isaaclab_assets",
        "torch",
    ]
    missing = []
    for module in modules:
        status = _module_status(module)
        print(f"{module}: {status}")
        if status == "MISSING":
            missing.append(module)

    if args.launch_kit:
        ok, message = _simulation_app_status()
        print(f"kit_startup: {'OK' if ok else 'MISSING'} ({message})")
        if not ok:
            missing.append("kit_startup")
    else:
        print("kit_startup: SKIPPED (use --launch-kit to verify omni.usd and pxr after SimulationApp startup)")

    if missing:
        print("\nIsaac is not ready in this Python environment.")
        print("Recommended next step:")
        print("  python3.11 -m venv .venv_isaac")
        print("  source .venv_isaac/bin/activate")
        print("  pip install --upgrade pip")
        print("  pip install 'isaacsim[all,extscache]' --extra-index-url https://pypi.nvidia.com")
        print("  # Then install Isaac Lab from source following its official guide.")
        return 1

    print("\nIsaac Sim / Isaac Lab modules are importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
