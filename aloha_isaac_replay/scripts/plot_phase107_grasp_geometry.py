from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_bottle_rect(ax, bbox: dict[str, Any], dims: tuple[int, int], *, color: str, label: str) -> None:
    min_v = np.asarray(bbox["min"], dtype=float)
    max_v = np.asarray(bbox["max"], dtype=float)
    x0, y0 = min_v[list(dims)]
    x1, y1 = max_v[list(dims)]
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2.2, color=color, label=label)
    ax.add_patch(rect)


def _axis_label(index: int) -> str:
    return ("X", "Y", "Z")[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase107 BottleUSD rear-quarter grasp geometry from metrics JSON.")
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metrics_path = _resolve(args.metrics_json)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "phase107_rear_quarter_grasp_geometry.png"
    report_path = output_dir / "phase107_rear_quarter_grasp_geometry.md"

    payload = _load(metrics_path)
    bottle_gate = payload["bottle_runtime_composition_gate"]
    bbox = bottle_gate["bbox"]
    semantics = payload["bottle_grasp_semantics_gate"]
    placement = payload["object_placement"]
    finger_center = np.asarray(semantics["finger_contact_center_world_m"], dtype=float)
    bottle_min = np.asarray(bbox["min"], dtype=float)
    bottle_max = np.asarray(bbox["max"], dtype=float)
    bottle_center = np.asarray(bbox["center"], dtype=float)
    axis_index = int(semantics["object_axis_index"])
    target_fraction = float(semantics["rear_fraction_target"])
    target_axis_value = bottle_min[axis_index] + target_fraction * (bottle_max[axis_index] - bottle_min[axis_index])
    closing_axis = semantics["finger_gap_axis"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    fig.suptitle("Phase107 BottleUSD grasp geometry: rear-quarter body grasp", fontsize=14, fontweight="bold")

    top = axes[0]
    _plot_bottle_rect(top, bbox, (0, 1), color="#1f77b4", label="Bottle body bbox")
    top.axvline(target_axis_value, color="#2ca02c", linestyle="--", linewidth=2, label="Rear-quarter target")
    top.scatter([finger_center[0]], [finger_center[1]], s=90, color="#d62728", zorder=5, label="Finger gap center")
    top.annotate(
        "finger gap center\nat rear 1/4",
        xy=(finger_center[0], finger_center[1]),
        xytext=(finger_center[0] + 0.015, finger_center[1] + 0.02),
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        fontsize=10,
    )
    top.arrow(
        bottle_min[0],
        bottle_center[1],
        bottle_max[0] - bottle_min[0],
        0,
        width=0.001,
        head_width=0.008,
        color="#1f77b4",
        length_includes_head=True,
    )
    top.text(bottle_center[0], bottle_center[1] - 0.045, "Bottle long axis = world X", ha="center", fontsize=10)
    top.set_title("Top view: bottle length and rear-quarter position")
    top.set_xlabel("world X (m)")
    top.set_ylabel("world Y (m)")
    top.axis("equal")
    top.grid(True, alpha=0.25)
    top.legend(loc="upper right")

    side = axes[1]
    _plot_bottle_rect(side, bbox, (0, 2), color="#1f77b4", label="Bottle body bbox")
    side.axvline(target_axis_value, color="#2ca02c", linestyle="--", linewidth=2, label="Rear-quarter target")
    side.scatter([finger_center[0]], [finger_center[2]], s=90, color="#d62728", zorder=5, label="Finger gap center")
    side.annotate(
        "closing axis is Z\nperpendicular to bottle X",
        xy=(finger_center[0], finger_center[2]),
        xytext=(finger_center[0] + 0.012, finger_center[2] + 0.055),
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        fontsize=10,
    )
    side.arrow(
        finger_center[0],
        finger_center[2] - 0.035,
        0,
        0.07,
        width=0.0009,
        head_width=0.006,
        color="#d62728",
        length_includes_head=True,
    )
    side.arrow(
        bottle_min[0],
        bottle_center[2],
        bottle_max[0] - bottle_min[0],
        0,
        width=0.001,
        head_width=0.008,
        color="#1f77b4",
        length_includes_head=True,
    )
    side.set_title("Side view: closing direction is perpendicular")
    side.set_xlabel("world X (m)")
    side.set_ylabel("world Z (m)")
    side.axis("equal")
    side.grid(True, alpha=0.25)
    side.legend(loc="upper right")

    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    lines = [
        "# Phase107 Rear-Quarter Bottle Grasp Geometry",
        "",
        f"- Source metrics: `{_rel(metrics_path)}`",
        f"- Figure: `{_rel(png_path)}`",
        f"- Runtime bottle path: `{payload.get('visible_bottle_runtime_path')}`",
        f"- Bottle runtime composition: `{bottle_gate.get('status')}`",
        f"- Bottle visual mesh count: `{bottle_gate.get('visual_mesh_count')}`",
        f"- Bottle collision prim count: `{bottle_gate.get('collision_prim_count')}`",
        f"- Bottle bbox size: `{bbox.get('size')}` m",
        f"- Finger gap center: `{semantics.get('finger_contact_center_world_m')}` m",
        f"- Object axis: `{semantics.get('object_axis')}` / index `{axis_index}`",
        f"- Finger gap axis: `{closing_axis}`",
        f"- Rear-quarter fraction: `{semantics.get('fraction_from_axis_min')}`",
        f"- Closing axis dot long axis abs: `{semantics.get('closing_long_axis_dot_abs')}`",
        f"- Contact status: `{payload.get('contact_trace_status')}`",
        f"- Target finger contacts: `{payload.get('target_contact_finger_hits')}`",
        f"- Controller tracking: `{payload.get('controller_tracking_gate', {}).get('status')}`",
        "",
        "## Interpretation",
        "",
        "This plot is generated from the Isaac runtime metric JSON, not from a hand-drawn estimate. "
        "The green dashed line is the desired rear-quarter body point. The red point is the live finger "
        "gap center. Passing this gate means the replay starts with the BottleUSD body placed so the left "
        "gripper holds the bottle body around the rear 1/4, and the finger closing direction is perpendicular "
        "to the bottle long axis.",
        "",
        "## Current Limitation",
        "",
        "This remains an already-in-contact replay gate. It proves runtime placement/contact semantics for the "
        "HDF5 replay start, but it does not yet prove a full free-space approach, active grasp, lift, or RL training loop.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "figure": _rel(png_path),
                "report": _rel(report_path),
                "fraction_from_axis_min": semantics.get("fraction_from_axis_min"),
                "closing_long_axis_dot_abs": semantics.get("closing_long_axis_dot_abs"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
