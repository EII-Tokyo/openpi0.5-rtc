from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import warnings

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib

_BACKEND_ALIASES = {
    "agg": "Agg",
    "qtagg": "QtAgg",
    "tkagg": "TkAgg",
    "webagg": "WebAgg",
}


def _cli_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _configure_backend(argv: list[str]) -> None:
    raw_backend = _cli_value(argv, "--backend")
    # Let explicit CLI --backend override MPLBACKEND.
    if raw_backend is None and os.environ.get("MPLBACKEND"):
        return

    backend = (raw_backend or "auto").strip().lower()

    if backend == "auto":
        is_headless = any(flag in argv for flag in ("--no_show", "--no-show", "--mp4_path", "--mp4-path"))
        selected = "Agg" if is_headless else "WebAgg"
        matplotlib.use(selected)
        return

    if backend not in _BACKEND_ALIASES:
        choices = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(f"Unsupported --backend '{raw_backend}'. Choose one of: {choices}, auto")

    matplotlib.use(_BACKEND_ALIASES[backend])


_configure_backend(sys.argv)

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402
from matplotlib.widgets import Slider  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def _add_local_paths() -> None:
    this_file = Path(__file__).resolve()
    pkg_src = this_file.parents[1] / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))


_add_local_paths()

from topreward.data.lerobot_loader import load_episode_frames  # noqa: E402


def _load_results(path: str) -> list[dict[str, Any]]:
    results_path = Path(path).expanduser().resolve()
    data = json.loads(results_path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported results format in {results_path}")


def _select_result(results: list[dict[str, Any]], result_index: int, episode_idx: int | None) -> dict[str, Any]:
    if not results:
        raise ValueError("results is empty")

    if episode_idx is not None:
        for item in results:
            if int(item.get("episode_idx", -1)) == episode_idx:
                return item
        raise ValueError(f"No result entry found for episode_idx={episode_idx}")

    if result_index < 0 or result_index >= len(results):
        raise IndexError(f"result_index={result_index} out of range [0, {len(results) - 1}]")
    return results[result_index]


def _metadata_subsample_fps(metadata: dict[str, Any]) -> int | None:
    fps = metadata.get("dataset_fps")
    stride = metadata.get("stride")
    if not isinstance(fps, (int, float)) or not isinstance(stride, (int, float)):
        return None
    if stride <= 1:
        return None
    return max(1, int(round(float(fps) / float(stride))))


def _interpolate_progress(timestamps_1based: list[int], values: list[float], num_frames: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    if not timestamps_1based or not values:
        return np.zeros((num_frames,), dtype=np.float32)

    n = min(len(timestamps_1based), len(values))
    pairs = sorted((int(timestamps_1based[i]) - 1, float(values[i])) for i in range(n))

    xs: list[int] = []
    ys: list[float] = []
    for x_raw, y in pairs:
        x = int(min(max(x_raw, 0), num_frames - 1))
        if xs and x == xs[-1]:
            ys[-1] = y
        else:
            xs.append(x)
            ys.append(y)

    if len(xs) == 1:
        return np.full((num_frames,), ys[0], dtype=np.float32)

    xq = np.arange(num_frames, dtype=np.int32)
    interp = np.interp(xq, np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32))
    return interp.astype(np.float32)


def _resolve_current_index(current_timestep: int, num_frames: int) -> int:
    idx = current_timestep
    if idx < 0:
        idx = num_frames + idx
    if idx < 0 or idx >= num_frames:
        raise IndexError(f"current_timestep={current_timestep} resolves to {idx}, out of range [0, {num_frames - 1}]")
    return idx


def _to_display_arrays(frames: list[Image.Image], max_side: int | None) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for frame in frames:
        image = frame
        if max_side is not None and max_side > 0:
            width, height = image.size
            longest = max(width, height)
            if longest > max_side:
                scale = float(max_side) / float(longest)
                new_width = max(1, int(round(width * scale)))
                new_height = max(1, int(round(height * scale)))
                image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        out.append(np.ascontiguousarray(np.asarray(image)))
    return out


def _to_pil_rgb(frame: np.ndarray) -> Image.Image:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {array.shape}")

    if array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] != 3:
        raise ValueError(f"Unsupported channel dimension in frame: {array.shape}")

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if float(np.max(array)) <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array, mode="RGB")


def _decode_video_frames_cv2(video_path: Path, stride: int, max_frames: int | None) -> tuple[list[Image.Image], float | None]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video via OpenCV: {video_path}")

    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps_raw if fps_raw > 0 else None

    frames: list[Image.Image] = []
    decoded = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        if decoded % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(_to_pil_rgb(rgb))
            if max_frames is not None and len(frames) >= max_frames:
                break
        decoded += 1

    cap.release()
    return frames, fps


def _decode_video_frames_imageio(
    video_path: Path,
    stride: int,
    max_frames: int | None,
) -> tuple[list[Image.Image], float | None]:
    import imageio.v3 as iio

    metadata = iio.immeta(str(video_path))
    fps_raw = metadata.get("fps")
    fps = float(fps_raw) if isinstance(fps_raw, (int, float)) and fps_raw > 0 else None

    frames: list[Image.Image] = []
    decoded = 0
    for frame in iio.imiter(str(video_path)):
        if decoded % stride == 0:
            frames.append(_to_pil_rgb(frame))
            if max_frames is not None and len(frames) >= max_frames:
                break
        decoded += 1

    return frames, fps


def _decode_video_frames(
    video_path: Path,
    stride: int,
    max_frames: int | None,
) -> tuple[list[Image.Image], float | None, str]:
    errors: list[str] = []

    try:
        frames, fps = _decode_video_frames_cv2(video_path, stride=stride, max_frames=max_frames)
        return frames, fps, "cv2"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cv2: {exc}")

    try:
        frames, fps = _decode_video_frames_imageio(video_path, stride=stride, max_frames=max_frames)
        return frames, fps, "imageio"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"imageio: {exc}")

    raise RuntimeError(f"Failed to decode video '{video_path}'. Attempts: {' ; '.join(errors)}")


def _render_static(
    frame_images: list[np.ndarray],
    sampled_x: np.ndarray,
    sampled_y: np.ndarray,
    progress_full: np.ndarray,
    current_idx: int,
    info_text: str,
    save_path: str | None,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2])

    ax_curve = fig.add_subplot(grid[0, :])
    ax_frame = fig.add_subplot(grid[1, :2])
    ax_info = fig.add_subplot(grid[1, 2])

    t = np.arange(len(progress_full), dtype=np.int32)
    ax_curve.plot(t, progress_full, color="tab:blue", linewidth=2, label="interpolated progress")
    ax_curve.scatter(sampled_x, sampled_y, s=40, color="tab:orange", label="sampled prefixes", zorder=3)
    ax_curve.scatter([current_idx], [progress_full[current_idx]], s=80, color="tab:red", zorder=4, label="current")
    ax_curve.axvline(current_idx, color="tab:red", linestyle="--", linewidth=1)
    ax_curve.set_title("TOPReward Progress")
    ax_curve.set_xlabel("Frame Index")
    ax_curve.set_ylabel("Normalized Progress")
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(loc="upper right")

    ax_frame.imshow(frame_images[current_idx])
    ax_frame.set_title(f"Robot View @ frame {current_idx}")
    ax_frame.axis("off")

    ax_info.axis("off")
    ax_info.text(0.0, 1.0, info_text, va="top", ha="left", fontsize=10)

    if save_path:
        out = Path(save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _render_loop(
    frame_images: list[np.ndarray],
    sampled_x: np.ndarray,
    sampled_y: np.ndarray,
    progress_full: np.ndarray,
    start_idx: int,
    info_lines: list[str],
    fps: int,
    playback_step: int,
    ui_update_hz: int,
    mp4_path: str | None,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.2],
        left=0.04,
        right=0.98,
        top=0.96,
        bottom=0.18,
        wspace=0.2,
        hspace=0.25,
    )

    ax_curve = fig.add_subplot(grid[0, :])
    ax_frame = fig.add_subplot(grid[1, :2])
    ax_info = fig.add_subplot(grid[1, 2])

    ax_slider = fig.add_axes([0.10, 0.08, 0.65, 0.04])
    ax_button = fig.add_axes([0.78, 0.075, 0.10, 0.05])
    ax_speed = fig.add_axes([0.90, 0.08, 0.08, 0.04])

    t = np.arange(len(progress_full), dtype=np.int32)
    ax_curve.plot(t, progress_full, color="tab:blue", linewidth=2, label="interpolated progress")
    ax_curve.scatter(sampled_x, sampled_y, s=35, color="tab:orange", label="sampled prefixes", zorder=3)
    cursor = ax_curve.scatter([start_idx], [progress_full[start_idx]], s=80, color="tab:red", zorder=4)
    vline = ax_curve.axvline(start_idx, color="tab:red", linestyle="--", linewidth=1)
    ax_curve.set_title("TOPReward Progress")
    ax_curve.set_xlabel("Frame Index")
    ax_curve.set_ylabel("Normalized Progress")
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(loc="upper right")

    image_handle = ax_frame.imshow(frame_images[start_idx])
    ax_frame.set_title(f"Robot View @ frame {start_idx}")
    ax_frame.axis("off")

    ax_info.axis("off")
    text_handle = ax_info.text(0.0, 1.0, "", va="top", ha="left", fontsize=10, wrap=True)

    max_idx = len(frame_images) - 1
    frame_slider = Slider(ax_slider, "Frame", 0, max_idx, valinit=start_idx, valstep=1)
    play_button = Button(ax_button, "Pause")
    speed_slider = Slider(ax_speed, "xSpeed", 0.25, 4.0, valinit=1.0, valstep=0.25)

    state: dict[str, Any] = {
        "idx": int(start_idx),
        "playing": True,
        "fps": float(max(fps, 1)),
        "playback_step": max(1, int(playback_step)),
        "ui_update_hz": max(1, int(ui_update_hz)),
        "updating_from_slider": False,
    }

    anim_ref: dict[str, animation.FuncAnimation | None] = {"anim": None}

    def _timer() -> Any:
        anim_obj = anim_ref["anim"]
        if anim_obj is None:
            return None
        return anim_obj.event_source

    def _set_playing(is_playing: bool) -> None:
        state["playing"] = is_playing
        play_button.label.set_text("Pause" if is_playing else "Play")
        timer = _timer()
        if timer is None:
            return
        if is_playing:
            timer.start()
        else:
            timer.stop()

    def _set_interval() -> None:
        base_interval = 1000.0 / max(state["fps"], 1.0)
        speed = float(speed_slider.val)
        interval = max(10, int(base_interval / max(speed, 0.25)))
        timer = _timer()
        if timer is not None:
            timer.interval = interval

    def _should_refresh_ui(local_t: int) -> bool:
        if not state["playing"]:
            return True
        effective_fps = max(1.0, state["fps"] * float(speed_slider.val))
        period = max(1, int(round(effective_fps / float(state["ui_update_hz"]))))
        return local_t % period == 0

    def _draw(local_t: int) -> tuple[Any, ...]:
        cursor.set_offsets(np.array([[local_t, progress_full[local_t]]], dtype=np.float32))
        vline.set_xdata([local_t, local_t])
        image_handle.set_data(frame_images[local_t])
        ax_frame.set_title(f"Robot View @ frame {local_t}")

        if _should_refresh_ui(local_t):
            info = [
                *info_lines,
                f"current_frame: {local_t}",
                f"current_progress: {float(progress_full[local_t]):.4f}",
                "controls: space(play/pause), left/right(step), up/down(speed)",
            ]
            text_handle.set_text("\n".join(info))
        return cursor, vline, image_handle, text_handle

    def _set_frame(local_t: int, from_slider: bool = False) -> tuple[Any, ...]:
        wrapped = int(local_t) % len(frame_images)
        state["idx"] = wrapped
        artists = _draw(wrapped)
        if not from_slider and _should_refresh_ui(wrapped):
            state["updating_from_slider"] = True
            frame_slider.set_val(wrapped)
            state["updating_from_slider"] = False
        return artists

    def _on_slider_change(value: float) -> None:
        if state["updating_from_slider"]:
            return
        _set_playing(False)
        _set_frame(int(round(value)), from_slider=True)
        fig.canvas.draw_idle()

    def _on_play_click(_event: Any) -> None:
        _set_playing(not state["playing"])
        fig.canvas.draw_idle()

    def _on_speed_change(_value: float) -> None:
        _set_interval()

    def _on_key(event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key in {" ", "space", "k"}:
            _on_play_click(event)
            return
        if key in {"right", "d", "l"}:
            _set_playing(False)
            _set_frame(state["idx"] + 1)
            fig.canvas.draw_idle()
            return
        if key in {"left", "a", "h"}:
            _set_playing(False)
            _set_frame(state["idx"] - 1)
            fig.canvas.draw_idle()
            return
        if key in {"up", "+"}:
            speed_slider.set_val(min(4.0, float(speed_slider.val) + 0.25))
            fig.canvas.draw_idle()
            return
        if key in {"down", "-"}:
            speed_slider.set_val(max(0.25, float(speed_slider.val) - 0.25))
            fig.canvas.draw_idle()
            return

    frame_slider.on_changed(_on_slider_change)
    play_button.on_clicked(_on_play_click)
    speed_slider.on_changed(_on_speed_change)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    _set_frame(start_idx)

    interval_ms = int(1000 / max(fps, 1))

    def _update(_frame: int) -> tuple[Any, ...]:
        if not state["playing"]:
            return ()
        return _set_frame(state["idx"] + state["playback_step"])

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_images),
        interval=interval_ms,
        repeat=True,
        blit=False,
        cache_frame_data=False,
    )
    anim_ref["anim"] = anim
    _set_interval()
    _set_playing(True)

    if mp4_path:
        out = Path(mp4_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.FFMpegWriter(fps=max(fps, 1))
        anim.save(str(out), writer=writer, dpi=120)
        print(f"Saved loop MP4 to: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TOPReward progress with robot video frames")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "webagg", "tkagg", "qtagg", "agg"],
        help="Matplotlib backend. auto uses WebAgg for interactive mode and Agg for --no_show/--mp4_path.",
    )
    parser.add_argument("--results", type=str, required=True, help="Path to evaluate_offline.py output JSON")
    parser.add_argument("--dataset_repo", type=str, default=None, help="LeRobot dataset repo id (offline eval mode).")
    parser.add_argument(
        "--video_path",
        "--video-path",
        dest="video_path",
        type=str,
        default=None,
        help="Local video path for evaluate_video.py outputs. If omitted, uses `video_path` from results JSON when present.",
    )
    parser.add_argument(
        "--video_stride",
        "--video-stride",
        dest="video_stride",
        type=int,
        default=None,
        help="Frame stride when loading local video for visualization. Defaults to metadata stride from results JSON.",
    )
    parser.add_argument("--result_index", type=int, default=0)
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--camera_key", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subsample_fps", type=int, default=None)
    parser.add_argument(
        "--display_max_side",
        "--display-max-side",
        dest="display_max_side",
        type=int,
        default=720,
        help="Resize display frames so longest side <= this many pixels for faster playback. <=0 disables resizing.",
    )
    parser.add_argument("--current_timestep", type=int, default=-1)
    parser.add_argument("--loop", action="store_true", help="Animate through the episode")
    parser.add_argument("--fps", type=int, default=None, help="Playback FPS for --loop")
    parser.add_argument(
        "--ui_update_hz",
        "--ui-update-hz",
        dest="ui_update_hz",
        type=int,
        default=10,
        help="Max UI refresh rate for slider/info text during playback. Lower values improve high-FPS performance.",
    )
    parser.add_argument(
        "--playback_step",
        "--playback-step",
        dest="playback_step",
        type=int,
        default=1,
        help="Advance this many frames per animation tick in --loop mode.",
    )
    parser.add_argument("--save_path", type=str, default=None, help="PNG path for static visualization")
    parser.add_argument("--mp4_path", "--mp4-path", dest="mp4_path", type=str, default=None, help="MP4 output path for --loop")
    parser.add_argument("--no_show", "--no-show", dest="no_show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = _load_results(args.results)
    selected = _select_result(results, result_index=args.result_index, episode_idx=args.episode_index)

    episode_idx = int(selected.get("episode_idx", 0) if args.episode_index is None else args.episode_index)
    metadata = selected.get("metadata", {}) if isinstance(selected.get("metadata"), dict) else {}
    selected_video_path = selected.get("video_path")
    local_video_path = args.video_path or (selected_video_path if isinstance(selected_video_path, str) else None)
    local_video_mode = local_video_path is not None

    instruction_from_source = ""
    if local_video_mode:
        if args.dataset_repo is not None:
            warnings.warn("--dataset_repo is ignored in local video mode.", stacklevel=2)
        if args.camera_key is not None:
            warnings.warn("--camera_key is ignored in local video mode.", stacklevel=2)
        if args.subsample_fps is not None:
            warnings.warn("--subsample_fps is ignored in local video mode.", stacklevel=2)

        video_path = Path(local_video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")

        default_stride = 1
        meta_stride = metadata.get("stride")
        if isinstance(meta_stride, (int, float)) and int(meta_stride) > 0:
            default_stride = int(meta_stride)
        stride = args.video_stride if args.video_stride is not None else default_stride
        if stride < 1:
            raise ValueError("--video_stride must be >= 1")

        max_frames_arg = args.max_frames if args.max_frames is not None else metadata.get("loaded_frames")
        max_frames: int | None = None
        if isinstance(max_frames_arg, (int, float)) and int(max_frames_arg) > 0:
            max_frames = int(max_frames_arg)

        frames, detected_fps, decoder = _decode_video_frames(video_path, stride=stride, max_frames=max_frames)
        if not frames:
            raise ValueError(f"No frames decoded for visualization from {video_path}")

        effective_fps: float | None = None
        meta_effective_fps = metadata.get("effective_fps")
        if isinstance(meta_effective_fps, (int, float)) and float(meta_effective_fps) > 0:
            effective_fps = float(meta_effective_fps)
        elif isinstance(detected_fps, (int, float)) and float(detected_fps) > 0:
            effective_fps = float(detected_fps) / float(max(stride, 1))
        else:
            source_fps = metadata.get("source_fps")
            if isinstance(source_fps, (int, float)) and float(source_fps) > 0:
                effective_fps = float(source_fps) / float(max(stride, 1))

        load_meta = {
            "video_path": str(video_path),
            "decoder": decoder,
            "camera_key": "local_video",
            "dataset_fps": effective_fps,
            "stride": int(stride),
            "loaded_frames": len(frames),
        }
    else:
        if args.dataset_repo is None:
            raise ValueError("Provide --dataset_repo for LeRobot results, or --video_path for local video results.")

        camera_key = args.camera_key or selected.get("camera_key") or metadata.get("camera_key") or "observation.images.top"
        max_frames = args.max_frames if args.max_frames is not None else metadata.get("loaded_frames")
        subsample_fps = args.subsample_fps if args.subsample_fps is not None else _metadata_subsample_fps(metadata)

        dataset = LeRobotDataset(args.dataset_repo)
        frames, instruction_from_source, load_meta = load_episode_frames(
            dataset,
            episode_idx,
            camera_key=camera_key,
            max_frames=max_frames,
            subsample_fps=subsample_fps,
        )

        if not frames:
            raise ValueError("No frames loaded for visualization")

    progress = selected.get("progress", {}) if isinstance(selected.get("progress"), dict) else {}
    timestamps = [int(v) for v in progress.get("timestamps", [])]
    normalized = [float(v) for v in progress.get("normalized", [])]

    progress_full = _interpolate_progress(timestamps, normalized, len(frames))

    sampled_n = min(len(timestamps), len(normalized))
    sampled_x = np.asarray([min(max(int(t) - 1, 0), len(frames) - 1) for t in timestamps[:sampled_n]], dtype=np.int32)
    sampled_y = np.asarray(normalized[:sampled_n], dtype=np.float32)

    current_idx = _resolve_current_index(args.current_timestep, len(frames))
    frame_images = _to_display_arrays(frames, args.display_max_side)

    instruction = selected.get("instruction") or instruction_from_source or ""
    voc = selected.get("voc")
    success = selected.get("success", {}) if isinstance(selected.get("success"), dict) else {}

    if local_video_mode:
        info_lines = [
            f"video: {load_meta.get('video_path')}",
            f"decoder: {load_meta.get('decoder')}",
            f"frames: {len(frames)}",
            f"stride: {load_meta.get('stride')}",
            (
                f"effective_fps: {float(load_meta.get('dataset_fps')):.3f}"
                if isinstance(load_meta.get("dataset_fps"), (int, float))
                else "effective_fps: n/a"
            ),
            f"sampled_points: {len(sampled_x)}",
            f"VOC: {float(voc):.4f}" if isinstance(voc, (int, float)) else "VOC: n/a",
            (
                f"success_score: {float(success.get('score')):.4f}"
                if isinstance(success.get("score"), (int, float))
                else "success_score: n/a"
            ),
            f"instruction: {instruction}",
        ]
    else:
        info_lines = [
            f"repo: {args.dataset_repo}",
            f"episode: {episode_idx}",
            f"camera_key: {load_meta.get('camera_key')}",
            f"frames: {len(frames)}",
            f"sampled_points: {len(sampled_x)}",
            f"VOC: {float(voc):.4f}" if isinstance(voc, (int, float)) else "VOC: n/a",
            (
                f"success_score: {float(success.get('score')):.4f}"
                if isinstance(success.get("score"), (int, float))
                else "success_score: n/a"
            ),
            f"instruction: {instruction}",
        ]

    if args.loop:
        if args.no_show and not args.mp4_path:
            raise ValueError("--loop with --no_show requires --mp4_path")

        if args.fps is not None:
            loop_fps = args.fps
        elif isinstance(load_meta.get("dataset_fps"), (int, float)) and float(load_meta["dataset_fps"]) > 0:
            loop_fps = int(round(float(load_meta["dataset_fps"])))
        else:
            loop_fps = 10
        _render_loop(
            frame_images=frame_images,
            sampled_x=sampled_x,
            sampled_y=sampled_y,
            progress_full=progress_full,
            start_idx=current_idx,
            info_lines=info_lines,
            fps=loop_fps,
            playback_step=args.playback_step,
            ui_update_hz=args.ui_update_hz,
            mp4_path=args.mp4_path,
            show=not args.no_show,
        )
    else:
        info_text = "\n".join(info_lines + [f"current_frame: {current_idx}", f"current_progress: {float(progress_full[current_idx]):.4f}"])
        _render_static(
            frame_images=frame_images,
            sampled_x=sampled_x,
            sampled_y=sampled_y,
            progress_full=progress_full,
            current_idx=current_idx,
            info_text=info_text,
            save_path=args.save_path,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
