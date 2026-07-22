#!/usr/bin/env python3
"""Open a clean ALOHA1 skeleton USD for visual inspection only."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_USD = Path("aloha_isaac_rebuild/scenes/aloha_skeleton.usda")
DEFAULT_WINDOW_SIZE = (1600, 980)
DEFAULT_WORKSPACE_NUMBER = 2
DEFAULT_MIN_ALIVE_SECONDS = 3.0


def _viewer_should_continue(app: object, *, elapsed_s: float, min_alive_s: float) -> bool:
    """Return whether the lightweight viewer should keep stepping Kit.

    `SimulationApp.is_running()` may briefly report false during startup for
    these inspection-only stages.  A short grace period prevents immediate
    teardown, but after that the viewer must respect Kit/window shutdown state
    so the desktop close button can end the process.
    """

    if elapsed_s < min_alive_s:
        return True
    is_running = getattr(app, "is_running", None)
    if is_running is None:
        return True
    return bool(is_running())


def _workspace_index_from_number(workspace_number: int) -> int:
    if workspace_number < 1:
        raise ValueError(f"workspace number must be >= 1, got {workspace_number}")
    return workspace_number - 1


def _move_window_to_workspace(
    workspace_number: int,
    pid: int | None = None,
    attempts: int = 40,
    sleep_s: float = 0.25,
) -> bool:
    if "DISPLAY" not in os.environ:
        print("Skipping workspace move: DISPLAY is not set.", flush=True)
        return False
    if shutil.which("xdotool") is None:
        print("Skipping workspace move: xdotool is not installed.", flush=True)
        return False

    target_index = _workspace_index_from_number(workspace_number)
    process_id = str(pid if pid is not None else os.getpid())

    try:
        desktops = subprocess.run(
            ["xdotool", "get_num_desktops"],
            check=True,
            capture_output=True,
            text=True,
        )
        desktop_count = int(desktops.stdout.strip())
        if desktop_count <= target_index:
            subprocess.run(["xdotool", "set_num_desktops", str(workspace_number)], check=True)
    except Exception as exc:
        print(f"Skipping workspace move: failed to inspect desktops: {exc}", flush=True)
        return False

    window_id = None
    for _ in range(max(1, attempts)):
        result = subprocess.run(
            ["xdotool", "search", "--pid", process_id],
            check=False,
            capture_output=True,
            text=True,
        )
        window_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if window_ids:
            window_id = window_ids[-1]
            break
        time.sleep(sleep_s)

    if window_id is None:
        print(f"Skipping workspace move: no X11 window found for pid={process_id}.", flush=True)
        return False

    subprocess.run(["xdotool", "set_desktop_for_window", window_id, str(target_index)], check=True)
    actual = subprocess.run(
        ["xdotool", "get_desktop_for_window", window_id],
        check=False,
        capture_output=True,
        text=True,
    )
    actual_index = actual.stdout.strip()
    print(
        f"Moved skeleton viewer window {window_id} for pid={process_id} "
        f"to workspace {workspace_number} (X11 desktop index {actual_index}).",
        flush=True,
    )
    return actual_index == str(target_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD, help="Skeleton USD to inspect.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a GUI window.")
    parser.add_argument(
        "--workspace",
        type=int,
        default=DEFAULT_WORKSPACE_NUMBER,
        help="Move the GUI window to this 1-based desktop/workspace.",
    )
    parser.add_argument(
        "--no-move-to-workspace",
        action="store_true",
        help="Keep the GUI window on the current desktop/workspace.",
    )
    parser.add_argument(
        "--min-alive-seconds",
        type=float,
        default=DEFAULT_MIN_ALIVE_SECONDS,
        help=(
            "Minimum time to keep the viewer alive after opening the stage. "
            "After this grace period the viewer follows SimulationApp.is_running()."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usd_path = args.usd.resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"Skeleton USD does not exist: {usd_path}")

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": bool(args.headless),
            "window_title": "Isaac Sim - ALOHA1 Skeleton Viewer",
            "window_width": DEFAULT_WINDOW_SIZE[0],
            "window_height": DEFAULT_WINDOW_SIZE[1],
        }
    )
    if not args.headless and not args.no_move_to_workspace:
        _move_window_to_workspace(args.workspace)

    try:
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(str(usd_path)):
            raise RuntimeError(f"Isaac failed to open skeleton stage: {usd_path}")

        print(f"Opened skeleton stage: {usd_path}", flush=True)
        for _ in range(5):
            app.update()
        started_at = time.monotonic()
        while _viewer_should_continue(
            app,
            elapsed_s=time.monotonic() - started_at,
            min_alive_s=max(0.0, float(args.min_alive_seconds)),
        ):
            app.update()
            time.sleep(1.0 / 120.0)
    finally:
        app.close()


if __name__ == "__main__":
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    main()
