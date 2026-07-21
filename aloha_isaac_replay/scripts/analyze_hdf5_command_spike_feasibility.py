from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def _percentile_summary(values: list[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _load_qpos(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "observations/qpos" not in h5:
            raise KeyError(f"{path} does not contain observations/qpos")
        return np.asarray(h5["observations/qpos"], dtype=np.float64)


def _load_mapping(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("dof_mapping"), list):
        raise ValueError(f"{path} must contain a dof_mapping list")
    entries = []
    for row in data["dof_mapping"]:
        if not isinstance(row, dict):
            continue
        if row.get("dataset_index") is None or not row.get("canonical_name"):
            continue
        entries.append(
            {
                "canonical_name": str(row["canonical_name"]),
                "dataset_index": int(row["dataset_index"]),
                "sign": float(row.get("sign", 1.0)),
                "offset": float(row.get("offset", 0.0)),
                "scale": float(row.get("scale", 1.0)),
                "unit": str(row.get("unit", "")),
            }
        )
    if not entries:
        raise ValueError(f"{path} has no usable mapped DOFs")
    return entries


def _mapped_values(qpos: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(qpos[:, int(entry["dataset_index"])], dtype=np.float64)
    return raw * float(entry["sign"]) * float(entry["scale"]) + float(entry["offset"])


def _cluster_steps(steps: list[int], *, max_gap: int) -> list[dict[str, int]]:
    if not steps:
        return []
    ordered = sorted(int(s) for s in steps)
    clusters: list[dict[str, int]] = []
    start = prev = ordered[0]
    for step in ordered[1:]:
        if step - prev <= max_gap:
            prev = step
            continue
        clusters.append({"start_step": start, "end_step": prev, "length_steps": prev - start + 1})
        start = prev = step
    clusters.append({"start_step": start, "end_step": prev, "length_steps": prev - start + 1})
    return clusters


def _classify_failure(
    *,
    failure_joint: str | None,
    failure_step: int | None,
    clusters_by_joint: dict[str, list[dict[str, Any]]],
    metrics: dict[str, Any],
    spike_threshold_rad_s: float,
) -> str:
    if failure_joint is None or failure_step is None:
        return "NO_FAILURE_STEP_IN_METRICS"
    joint_clusters = clusters_by_joint.get(failure_joint) or []
    matching = [
        c for c in joint_clusters if int(c["cluster_start_step"]) <= failure_step <= int(c["cluster_end_step"])
    ]
    if not matching:
        return "NOT_COMMAND_SPIKE_DOMINATED"
    contact_categories = (metrics.get("tracking_spike_packet") or {}).get("contact_categories_at_step") or []
    if contact_categories and any(str(c) not in {"workcell_or_environment"} for c in contact_categories):
        return "CONTACT_LOADED_TRACKING_RESIDUAL"
    cluster = matching[0]
    if int(cluster["cluster_count"]) <= 1:
        return "SINGLE_SPIKE_RESIDUAL"
    total_steps = max(1, int((metrics.get("hdf5_gripper_replay_steps") or 0) - 1))
    joint_spike_count = sum(int(c["cluster_count"]) for c in joint_clusters)
    if joint_spike_count / total_steps > 0.1:
        return "GLOBAL_HIGH_RATE_COMMAND_MISMATCH"
    return "REPEATED_SPIKE_CLUSTER"


def analyze_command_spikes(
    *,
    hdf5_path: Path,
    mapping_path: Path,
    metrics_path: Path,
    output_dir: Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    hdf5_rate_hz: float | None = None,
    spike_threshold_rad_s: float = 2.0,
    cluster_gap_steps: int = 3,
) -> dict[str, Any]:
    qpos = _load_qpos(hdf5_path)
    mapping = _load_mapping(mapping_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    inputs = metrics.get("inputs") or {}

    resolved_start = int(start_frame if start_frame is not None else inputs.get("hdf5_gripper_start_frame", 0))
    resolved_end = int(end_frame if end_frame is not None else inputs.get("hdf5_gripper_end_frame", qpos.shape[0]))
    resolved_rate = float(hdf5_rate_hz if hdf5_rate_hz is not None else metrics.get("hdf5_replay_rate_hz", 50.0))
    if resolved_start < 0 or resolved_end > qpos.shape[0] or resolved_end <= resolved_start + 1:
        raise ValueError(
            f"invalid frame window start={resolved_start}, end={resolved_end}, qpos_frames={qpos.shape[0]}"
        )
    expected_steps = resolved_end - resolved_start - 1
    actual_steps = int(metrics.get("hdf5_gripper_replay_steps") or expected_steps)
    if actual_steps != expected_steps:
        raise ValueError(
            f"metrics/HDF5 timing mismatch: metrics hdf5_gripper_replay_steps={actual_steps}, "
            f"but frame window implies {expected_steps}"
        )

    target_frame_indices = np.arange(resolved_start + 1, resolved_end, dtype=np.int64)
    target_steps = np.arange(len(target_frame_indices), dtype=np.int64)
    dt = 1.0 / resolved_rate
    per_joint: dict[str, Any] = {}
    clusters_by_joint: dict[str, list[dict[str, Any]]] = {}

    for entry in mapping:
        name = entry["canonical_name"]
        values = _mapped_values(qpos, entry)
        current = values[target_frame_indices]
        previous = values[target_frame_indices - 1]
        next_indices = np.minimum(target_frame_indices + 1, resolved_end - 1)
        next_values = values[next_indices]
        prev_prev_indices = np.maximum(target_frame_indices - 2, 0)
        previous_delta = previous - values[prev_prev_indices]
        delta = current - previous
        next_delta = next_values - current
        velocity = delta / dt
        next_velocity = next_delta / dt
        acceleration = (delta - previous_delta) / (dt * dt)
        abs_velocity = np.abs(velocity)
        spike_mask = abs_velocity >= float(spike_threshold_rad_s)
        spike_steps = [int(s) for s in target_steps[spike_mask]]
        clusters = []
        for cluster in _cluster_steps(spike_steps, max_gap=cluster_gap_steps):
            mask = (target_steps >= cluster["start_step"]) & (target_steps <= cluster["end_step"]) & spike_mask
            cluster_vel = abs_velocity[mask]
            cluster_accel = np.abs(acceleration[mask])
            cluster_steps = target_steps[mask]
            peak_local = int(np.argmax(cluster_vel)) if cluster_vel.size else 0
            clusters.append(
                {
                    "joint_name": name,
                    "cluster_start_step": int(cluster["start_step"]),
                    "cluster_end_step": int(cluster["end_step"]),
                    "cluster_count": int(cluster_steps.size),
                    "cluster_span_steps": int(cluster["end_step"] - cluster["start_step"] + 1),
                    "cluster_span_seconds": float((cluster["end_step"] - cluster["start_step"] + 1) * dt),
                    "peak_step": int(cluster_steps[peak_local]) if cluster_steps.size else None,
                    "peak_hdf5_frame": int(resolved_start + 1 + int(cluster_steps[peak_local]))
                    if cluster_steps.size
                    else None,
                    "peak_abs_target_velocity": float(cluster_vel[peak_local]) if cluster_vel.size else None,
                    "peak_abs_target_acceleration": float(cluster_accel[peak_local]) if cluster_accel.size else None,
                }
            )
        clusters_by_joint[name] = clusters
        per_joint[name] = {
            "dataset_index": int(entry["dataset_index"]),
            "unit": entry["unit"],
            "target_velocity": _percentile_summary(abs_velocity),
            "target_acceleration": _percentile_summary(np.abs(acceleration)),
            "target_delta": _percentile_summary(np.abs(delta)),
            "spike_threshold_rad_s": float(spike_threshold_rad_s),
            "spike_count": len(spike_steps),
            "spike_fraction": float(len(spike_steps) / max(1, len(target_steps))),
            "clusters": clusters,
        }

    tracking_summary = metrics.get("tracking_summary", {}).get("groups", {}).get("controlled", {})
    failure_joint = tracking_summary.get("max_abs_error_dof_name")
    failure_step = tracking_summary.get("max_abs_error_step")
    failure_step_int = None if failure_step is None else int(failure_step)
    failure_frame = None if failure_step_int is None else int(resolved_start + 1 + failure_step_int)
    failure_joint_report = per_joint.get(str(failure_joint), {})
    failure_row: dict[str, Any] = {}
    if failure_joint in per_joint and failure_step_int is not None:
        entry = next(row for row in mapping if row["canonical_name"] == failure_joint)
        values = _mapped_values(qpos, entry)
        frame = int(resolved_start + 1 + failure_step_int)
        if 1 <= frame < qpos.shape[0]:
            delta = float(values[frame] - values[frame - 1])
            next_delta = float(values[min(frame + 1, resolved_end - 1)] - values[frame])
            prev_delta = float(values[frame - 1] - values[max(frame - 2, 0)])
            failure_row = {
                "joint_name": failure_joint,
                "dataset_index": int(entry["dataset_index"]),
                "failure_step": failure_step_int,
                "hdf5_frame_index": frame,
                "time_from_window_start_s": float((frame - resolved_start) * dt),
                "target_prev": float(values[frame - 1]),
                "target_current": float(values[frame]),
                "target_next": float(values[min(frame + 1, resolved_end - 1)]),
                "target_delta_prev_to_current": delta,
                "target_delta_current_to_next": next_delta,
                "target_velocity_prev_to_current": float(delta / dt),
                "target_velocity_current_to_next": float(next_delta / dt),
                "target_accel_estimate": float((delta - prev_delta) / (dt * dt)),
                "is_spike": bool(abs(delta / dt) >= spike_threshold_rad_s),
            }
    spike_packet = metrics.get("tracking_spike_packet") or {}
    failure_row.update(
        {
            "actual_qpos_pre": spike_packet.get("pre_step_qpos"),
            "actual_qpos_post": spike_packet.get("post_step_qpos"),
            "actual_delta": spike_packet.get("actual_delta_during_hold"),
            "actual_velocity": spike_packet.get("estimated_actual_velocity_during_hold"),
            "tracking_error_post": spike_packet.get("max_abs_error_signed"),
            "tracking_ratio": spike_packet.get("tracking_ratio"),
            "contact_state_at_failure": spike_packet.get("contact_categories_at_step"),
            "effort_clipped_at_failure": (metrics.get("drive_authority_audit") or {}).get("estimated_effort_clipped"),
        }
    )

    classification = _classify_failure(
        failure_joint=str(failure_joint) if failure_joint is not None else None,
        failure_step=failure_step_int,
        clusters_by_joint=clusters_by_joint,
        metrics=metrics,
        spike_threshold_rad_s=spike_threshold_rad_s,
    )
    output = {
        "script_name": Path(__file__).name,
        "read_only": True,
        "episode_path": str(hdf5_path),
        "metrics_path": str(metrics_path),
        "mapping_path": str(mapping_path),
        "frame_window": {"start": resolved_start, "end": resolved_end, "target_steps": expected_steps},
        "hdf5_rate_hz": resolved_rate,
        "physics_dt": metrics.get("physics_dt"),
        "target_hold_steps": metrics.get("hdf5_replay_target_hold_steps"),
        "formal_replay_mode": metrics.get("hdf5_replay_substep_mode")
        or inputs.get("hdf5_replay_substep_mode"),
        "spike_threshold_rad_s": float(spike_threshold_rad_s),
        "cluster_gap_steps": int(cluster_gap_steps),
        "failure_classification": classification,
        "failure_step": failure_row,
        "physical_grasp_gate": metrics.get("physical_grasp_gate"),
        "target_limit_gate": {"target_limit_gate_ok": metrics.get("target_limit_gate_ok")},
        "controller_fidelity_gate": metrics.get("controller_replay_fidelity_gate"),
        "per_joint": per_joint,
        "top_joint_spike_counts": sorted(
            ((name, row["spike_count"]) for name, row in per_joint.items()), key=lambda item: item[1], reverse=True
        )[:10],
        "notes": [
            "This report does not modify replay targets, drive gains, contact policy, USD, or physics settings.",
            "A command-rate spike is evidence about replay target feasibility, not proof that the real ALOHA command was invalid.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "command_spike_feasibility.json"
    md_path = output_dir / "command_spike_feasibility.md"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(md_path, output)
    output["json"] = str(json_path)
    output["markdown"] = str(md_path)
    return output


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    failure = report.get("failure_step") or {}
    lines = [
        "# HDF5 Command Spike Feasibility Report",
        "",
        "This is a read-only diagnostic. It does not change replay targets, drive gains, contact policy, USD, or physics settings.",
        "",
        "## Replay Context",
        "",
        f"- episode: `{report.get('episode_path')}`",
        f"- metrics: `{report.get('metrics_path')}`",
        f"- frame window: `{report.get('frame_window')}`",
        f"- HDF5 rate: `{report.get('hdf5_rate_hz')}` Hz",
        f"- physics dt: `{report.get('physics_dt')}`",
        f"- target hold steps: `{report.get('target_hold_steps')}`",
        f"- formal replay mode: `{report.get('formal_replay_mode')}`",
        f"- spike threshold: `{report.get('spike_threshold_rad_s')}` rad/s",
        "",
        "## Failure Classification",
        "",
        f"- classification: `{report.get('failure_classification')}`",
        f"- failure joint: `{failure.get('joint_name')}`",
        f"- failure close step: `{failure.get('failure_step')}`",
        f"- failure HDF5 frame: `{failure.get('hdf5_frame_index')}`",
        f"- target velocity into failure: `{_fmt(failure.get('target_velocity_prev_to_current'))}` rad/s",
        f"- target acceleration estimate: `{_fmt(failure.get('target_accel_estimate'))}` rad/s^2",
        f"- actual velocity during hold: `{_fmt(failure.get('actual_velocity'))}` rad/s",
        f"- tracking ratio: `{_fmt(failure.get('tracking_ratio'))}`",
        f"- contact state at failure: `{failure.get('contact_state_at_failure')}`",
        f"- effort clipped at failure: `{failure.get('effort_clipped_at_failure')}`",
        "",
        "## Per-Joint Target Velocity Summary",
        "",
        "| joint | p95 velocity | p99 velocity | max velocity | spike count | clusters |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report.get("per_joint", {}).items():
        vel = row.get("target_velocity") or {}
        lines.append(
            "| {name} | {p95} | {p99} | {maxv} | {count} | {clusters} |".format(
                name=name,
                p95=_fmt(vel.get("p95")),
                p99=_fmt(vel.get("p99")),
                maxv=_fmt(vel.get("max")),
                count=row.get("spike_count"),
                clusters=len(row.get("clusters") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Failure Joint Clusters",
            "",
            "| cluster | steps | count | seconds | peak step | peak HDF5 frame | peak velocity | peak acceleration |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    failure_joint = failure.get("joint_name")
    joint_row = (report.get("per_joint") or {}).get(failure_joint) or {}
    for i, cluster in enumerate(joint_row.get("clusters") or [], start=1):
        lines.append(
            "| {i} | {start}-{end} | {count} | {seconds} | {peak_step} | {frame} | {vel} | {accel} |".format(
                i=i,
                start=cluster.get("cluster_start_step"),
                end=cluster.get("cluster_end_step"),
                count=cluster.get("cluster_count"),
                seconds=_fmt(cluster.get("cluster_span_seconds")),
                peak_step=cluster.get("peak_step"),
                frame=cluster.get("peak_hdf5_frame"),
                vel=_fmt(cluster.get("peak_abs_target_velocity")),
                accel=_fmt(cluster.get("peak_abs_target_acceleration")),
            )
        )
    lines.extend(
        [
            "",
            "## Gate Snapshot",
            "",
            f"- physical grasp gate: `{(report.get('physical_grasp_gate') or {}).get('status')}`",
            f"- target limit gate ok: `{(report.get('target_limit_gate') or {}).get('target_limit_gate_ok')}`",
            f"- controller fidelity gate: `{(report.get('controller_fidelity_gate') or {}).get('status')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze HDF5 replay command-rate spikes against existing metrics.")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--hdf5-rate-hz", type=float)
    parser.add_argument("--spike-threshold-rad-s", type=float, default=2.0)
    parser.add_argument("--cluster-gap-steps", type=int, default=3)
    args = parser.parse_args()
    report = analyze_command_spikes(
        hdf5_path=args.hdf5,
        mapping_path=args.mapping,
        metrics_path=args.metrics_json,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        hdf5_rate_hz=args.hdf5_rate_hz,
        spike_threshold_rad_s=args.spike_threshold_rad_s,
        cluster_gap_steps=args.cluster_gap_steps,
    )
    print(json.dumps({"json": report["json"], "markdown": report["markdown"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
