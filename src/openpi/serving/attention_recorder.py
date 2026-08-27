from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AttentionRecorder:
    """Persist compact action-to-camera attention captures and quick-look images."""

    def __init__(self, output_dir: str | Path, *, every_n: int = 1):
        if every_n < 1:
            raise ValueError("attention capture interval must be >= 1")
        run_name = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(output_dir).expanduser() / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.chmod(0o777)
        self.every_n = every_n
        self._infer_index = 0
        self._sample_index = 0
        logger.info("Attention diagnostics will be written to %s", self.run_dir)

    def should_record(self) -> bool:
        should_record = self._infer_index % self.every_n == 0
        self._infer_index += 1
        return should_record

    def record(
        self,
        raw_images: dict[str, np.ndarray],
        camera_maps: dict[str, np.ndarray],
        *,
        chunking_mode: str,
        capture_ms: float,
    ) -> Path:
        sample_name = f"sample_{self._sample_index:06d}"
        self._sample_index += 1
        sample_dir = self.run_dir / sample_name
        sample_dir.mkdir()
        sample_dir.chmod(0o777)

        images = {name: self._latest_rgb(image) for name, image in raw_images.items() if name in camera_maps}
        maps = {name: np.asarray(value, dtype=np.float32) for name, value in camera_maps.items()}
        np.savez_compressed(
            sample_dir / "attention.npz",
            **{f"attention__{name}": value for name, value in maps.items()},
        )
        (sample_dir / "attention.npz").chmod(0o666)
        for camera_name, image in images.items():
            image_path = sample_dir / f"{camera_name}.jpg"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            image_path.chmod(0o666)

        view_mass = {
            name: float(value[:, 0].sum(axis=(-2, -1)).mean())
            for name, value in maps.items()
        }
        metadata = {
            "sample": sample_name,
            "unix_time": time.time(),
            "chunking_mode": chunking_mode,
            "capture_ms": capture_ms,
            "camera_order": list(maps),
            "attention_shape": {name: list(value.shape) for name, value in maps.items()},
            "mean_attention_mass": view_mass,
            "probe": {
                "query": "generated clean action tokens",
                "head_reduction": "mean",
                "saved_layers": "all",
                "saved_action_queries": "all",
                "quicklook_layers": "last 6",
                "quicklook_action_queries": "first 10",
            },
        }
        metadata_path = sample_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata_path.chmod(0o666)
        overview_path = sample_dir / "overview.jpg"
        self._write_quicklook(overview_path, images, maps, view_mass)
        overview_path.chmod(0o666)
        manifest_path = self.run_dir / "manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        manifest_path.chmod(0o666)
        logger.info(
            "Saved attention capture %s (%.1f ms, camera mass=%s)",
            sample_dir,
            capture_ms,
            {name: round(value, 4) for name, value in view_mass.items()},
        )
        return sample_dir

    @staticmethod
    def _latest_rgb(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 4:
            image = image[-1]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    @staticmethod
    def _write_quicklook(
        output_path: Path,
        images: dict[str, np.ndarray],
        maps: dict[str, np.ndarray],
        view_mass: dict[str, float],
    ) -> None:
        panels = []
        camera_names = list(maps)
        total_mass = sum(view_mass.values()) or 1.0
        for camera_name in camera_names:
            image = images[camera_name]
            # [layers, batch, action query, patch row, patch col]
            heatmap = maps[camera_name][-6:, 0, :10].mean(axis=(0, 1))
            heatmap = cv2.resize(
                heatmap,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            low, high = np.percentile(heatmap, [5, 99])
            normalized = np.clip((heatmap - low) / max(high - low, 1e-12), 0, 1)
            colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(image, 0.58, colored, 0.42, 0)
            share = 100.0 * view_mass[camera_name] / total_mass
            cv2.putText(
                overlay,
                f"{camera_name}  visual-attn share {share:.1f}%",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(overlay)

        target_height = min(panel.shape[0] for panel in panels)
        resized = [
            cv2.resize(panel, (round(panel.shape[1] * target_height / panel.shape[0]), target_height))
            for panel in panels
        ]
        quicklook = np.concatenate(resized, axis=1)
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(quicklook, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
