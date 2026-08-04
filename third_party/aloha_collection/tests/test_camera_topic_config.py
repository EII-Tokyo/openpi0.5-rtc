from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATIONARY_CONFIG = ROOT / "config" / "robot" / "aloha_stationary.yaml"


def test_stationary_cameras_subscribe_to_published_raw_images():
    config = yaml.safe_load(STATIONARY_CONFIG.read_text(encoding="utf-8"))

    assert (
        config["robot"]["cameras"]["common_parameters"]["color_image_topic_name"]
        == "{}/camera/color/image_raw"
    )
