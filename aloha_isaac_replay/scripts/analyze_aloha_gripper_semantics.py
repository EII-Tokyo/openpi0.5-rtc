from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aloha_isaac_replay.data.gripper_semantics import analyze_episode_grippers


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    left = payload["left"]
    right = payload["right"]
    lines = [
        "# ALOHA Gripper qpos/action Semantics",
        "",
        f"- Episode: `{payload['episode']}`",
        f"- qpos/action must remain separate: `{payload['interpretation']['qpos_action_must_remain_separate']}`",
        "",
        "## Summary",
        "",
        "| side | qpos index | qpos min/max | action min/max | rmse qpos-action | linear scale | linear offset | linear rmse |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for item in (left, right):
        lines.append(
            "| {side} | {idx} | {qmin:.6f}..{qmax:.6f} | {amin:.6f}..{amax:.6f} | {rmse:.6f} | {scale:.6f} | {offset:.6f} | {lrmse:.6f} |".format(
                side=item["side"],
                idx=item["qpos_index"],
                qmin=item["qpos_stats"]["min"],
                qmax=item["qpos_stats"]["max"],
                amin=item["action_stats"]["min"],
                amax=item["action_stats"]["max"],
                rmse=item["rmse_qpos_action"],
                scale=item["linear_fit_scale"],
                offset=item["linear_fit_offset"],
                lrmse=item["linear_fit_rmse"],
            )
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"- Timeseries: `{payload['plots'].get('timeseries', '')}`",
            f"- Scatter: `{payload['plots'].get('scatter', '')}`",
            f"- Preview frames: `{payload['plots'].get('preview_frames', '')}`",
            "",
            "## Interpretation",
            "",
            "The diagnostics deliberately do not choose a final action mapping. Kinematic qpos replay must use HDF5 `qpos[6]` and `qpos[13]`; action replay must be treated as a later separate problem.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze qpos/action gripper semantics for one ALOHA HDF5 episode.")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payload = analyze_episode_grippers(args.episode, args.output_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_markdown(Path(args.output_md), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

