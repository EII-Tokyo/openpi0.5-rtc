"""Validate the checkpoint input contract before creating any robot environment."""
import json
from pathlib import Path


def configure_video_input(server_metadata, model_dir):
    settings = server_metadata.get("runtime", {}).get("video_input")
    if not isinstance(settings, dict):
        raise ValueError("Server has no checkpoint video_input metadata; restart the policy server with updated code")
    required = {"config_name", "checkpoint_step", "camera_keys", "image_resolution", "num_frames", "stride_seconds", "mosaic"}
    if not required <= settings.keys():
        raise ValueError("Incomplete checkpoint video_input metadata")
    manifest_path = Path(model_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != settings["config_name"]:
            raise ValueError("Runtime checkpoint and policy server training configs differ")
    if Path(model_dir).name != str(settings["checkpoint_step"]):
        raise ValueError("Runtime checkpoint and policy server checkpoint steps differ")
    cameras = settings["camera_keys"]
    allowed = {"cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"}
    if not cameras or len(set(cameras)) != len(cameras) or not set(cameras) <= allowed:
        raise ValueError("Invalid checkpoint camera keys")
    resolution = settings["image_resolution"]
    if len(resolution) != 2 or any(not isinstance(v, int) or v <= 0 for v in resolution):
        raise ValueError("Invalid checkpoint image resolution")
    if not isinstance(settings["num_frames"], int) or settings["num_frames"] < 1:
        raise ValueError("Invalid checkpoint video memory frame count")
    if settings["stride_seconds"] <= 0:
        raise ValueError("Invalid checkpoint video memory stride")
    mosaic = settings["mosaic"]
    if mosaic is not None:
        from openpi.data.temporal_mosaic import HighCameraMosaicConfig
        mosaic_config = HighCameraMosaicConfig(**mosaic)
        if not set(mosaic_config.cameras) <= set(cameras):
            raise ValueError("Mosaic camera missing from checkpoint camera keys")
        if tuple(resolution) != (2 * mosaic_config.tile_size,) * 2:
            raise ValueError("Mosaic tile size does not match checkpoint resolution")
        if settings["num_frames"] != mosaic_config.history_seconds + 1 or settings["stride_seconds"] != 1.0:
            raise ValueError("Mosaic history does not match checkpoint video memory")
    return dict(settings)
