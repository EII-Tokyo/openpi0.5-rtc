from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def select_layers(values: np.ndarray, last_layers: int | None) -> np.ndarray:
    if last_layers is None:
        return values
    if not 1 <= last_layers <= values.shape[0]:
        raise ValueError(f"--last-layers must be between 1 and {values.shape[0]}")
    return values[-last_layers:]


def load_map(sample_dir: Path, camera: str, last_layers: int | None) -> np.ndarray:
    with np.load(sample_dir / "attention.npz") as archive:
        values = archive[f"attention__{camera}"]
    # Stored shape: [18 layers, 1 batch, 50 action queries, 16, 16].
    values = select_layers(values, last_layers)
    return values.mean(axis=(0, 1, 2))


def add_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    normalized = np.clip((heatmap - low) / max(high - low, 1e-12), 0, 1)
    colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image, 0.58, colored, 0.42, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--last-layers", type=int)
    args = parser.parse_args()

    samples = sorted(args.run_dir.glob("sample_*"))
    if not samples:
        raise SystemExit("No samples found")

    timestamps = [json.loads((sample / "metadata.json").read_text())["unix_time"] for sample in samples]

    maps: dict[tuple[int, str], np.ndarray] = {}
    all_values = []
    for index, sample in enumerate(samples):
        for camera in CAMERAS:
            heatmap = load_map(sample, camera, args.last_layers)
            maps[index, camera] = heatmap
            all_values.append(heatmap.ravel())

    # One fixed scale across every frame and camera makes red comparable.
    global_values = np.concatenate(all_values)
    low, high = np.percentile(global_values, [5, 99.5])

    layer_count = args.last_layers or 18
    frames_dir = args.run_dir / f"video_frames_last_{layer_count}L_50T"
    frames_dir.mkdir(exist_ok=True)
    start = timestamps[0]
    concat_lines = []
    for index, sample in enumerate(samples):
        panels = []
        masses = {}
        with np.load(sample / "attention.npz") as archive:
            for camera in CAMERAS:
                values = archive[f"attention__{camera}"]
                values = select_layers(values, args.last_layers)
                masses[camera] = float(values[:, 0].sum(axis=(-2, -1)).mean())
        total_mass = sum(masses.values()) or 1.0

        for camera in CAMERAS:
            image = cv2.imread(str(sample / f"{camera}.jpg"))
            panel = add_heatmap(image, maps[index, camera], low, high)
            share = 100 * masses[camera] / total_mass
            cv2.putText(
                panel,
                f"{camera}  visual-attn share {share:.1f}%",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)

        frame = np.concatenate(panels, axis=1)
        header = np.full((54, frame.shape[1], 3), (23, 23, 23), dtype=np.uint8)
        elapsed = timestamps[index] - start
        text = (
            f"{sample.name}   elapsed {elapsed:7.2f}s   "
            f"MEAN: last {layer_count} transformer layers x 50 action tokens x all attention heads"
        )
        cv2.putText(
            header,
            text,
            (14, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frame = np.concatenate([header, frame], axis=0)
        frame_path = frames_dir / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if index + 1 < len(samples):
            duration = max(0.05, min(10.0, timestamps[index + 1] - timestamps[index]))
        else:
            duration = 1.0
        concat_lines.extend([f"file '{frame_path.resolve()}'", f"duration {duration:.6f}"])

    # Repeating the last file makes ffmpeg honor its final duration.
    concat_lines.append(f"file '{(frames_dir / f'frame_{len(samples) - 1:06d}.jpg').resolve()}'")
    concat_path = frames_dir / "concat.txt"
    concat_path.write_text("\n".join(concat_lines) + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            f"fps={args.fps},format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(samples),
                "duration_s": timestamps[-1] - timestamps[0] + 1.0,
                "global_scale_p05": float(low),
                "global_scale_p995": float(high),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
