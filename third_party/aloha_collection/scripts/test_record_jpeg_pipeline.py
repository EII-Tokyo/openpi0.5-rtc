#!/usr/bin/env python3
"""
自检：复现 record_episodes_copy.py 中「组装即 JPEG + compress_len + padding + HDF5 图像数据集」逻辑。
无需 ROS/机器人；用于回归检测 OOM 修复路径是否正确。

运行（在仓库根或任意目录）:
  python3 scripts/test_record_jpeg_pipeline.py
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

# 与 record_episodes_copy.py 保持一致
JPEG_QUALITY = 100
_JPEG_ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]


def _encode_rgb_to_jpeg_flat(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, _JPEG_ENCODE_PARAM)
    assert ok, "imencode failed"
    return np.asarray(enc, dtype=np.uint8).reshape(-1).copy()


def _padding_pass(
    camera_map: dict[str, str],
    data_dict: dict,
) -> tuple[np.ndarray, int]:
    compressed_len: list[list[int]] = []
    for yaml_name in camera_map:
        cam_name = camera_map[yaml_name]
        image_list = data_dict[f"/observations/images/{cam_name}"]
        row_lens: list[int] = []
        for idx, buf in enumerate(image_list):
            assert buf is not None and isinstance(buf, np.ndarray) and buf.ndim == 1 and buf.size > 0, (
                f"{cam_name} idx {idx} invalid"
            )
            row_lens.append(int(buf.size))
        compressed_len.append(row_lens)
    arr = np.array(compressed_len, dtype=np.int64)
    padded_size = int(arr.max())
    for yaml_name in camera_map:
        cam_name = camera_map[yaml_name]
        padded_images: list[np.ndarray] = []
        for row in data_dict[f"/observations/images/{cam_name}"]:
            row = np.asarray(row, dtype=np.uint8).reshape(-1)
            padded_img = np.zeros(padded_size, dtype=np.uint8)
            padded_img[: row.size] = row
            padded_images.append(padded_img)
        data_dict[f"/observations/images/{cam_name}"] = padded_images
    return arr, padded_size


def main() -> None:
    rng = np.random.default_rng(0)
    T = 8
    h, w = 120, 160
    camera_map = {
        "camera_high": "cam_high",
        "camera_low": "cam_low",
        "camera_wrist_left": "cam_left_wrist",
        "camera_wrist_right": "cam_right_wrist",
    }
    data_dict: dict = {}
    for _, save in camera_map.items():
        data_dict[f"/observations/images/{save}"] = []

    for _t in range(T):
        for yaml_name, save_name in camera_map.items():
            rgb = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
            flat = _encode_rgb_to_jpeg_flat(rgb)
            data_dict[f"/observations/images/{save_name}"].append(flat)

    data_dict_before_pad = copy.deepcopy(data_dict)
    compressed_len, padded_size = _padding_pass(camera_map, data_dict)
    assert compressed_len.shape == (len(camera_map), T)
    assert padded_size > 0

    tmp = Path(tempfile.mkdtemp()) / "probe.hdf5"
    with h5py.File(tmp, "w") as root:
        root.attrs["compress"] = True
        obs = root.create_group("observations")
        ig = obs.create_group("images")
        for _, cam_name in camera_map.items():
            d = data_dict[f"/observations/images/{cam_name}"]
            assert len(d) == T and all(x.shape == (padded_size,) for x in d)
            stacked = np.stack(d, axis=0)
            ig.create_dataset(cam_name, data=stacked, chunks=(1, padded_size))
        root.create_dataset("compress_len", data=compressed_len)

    cam_order = [camera_map[k] for k in camera_map]
    with h5py.File(tmp, "r") as root:
        cl = np.array(root["compress_len"])
        for ci, cam_name in enumerate(cam_order):
            ds = root["observations/images"][cam_name]
            for t in range(T):
                row = np.array(ds[t])
                n = int(cl[ci, t])
                stored = row[:n].tobytes()
                expected_flat = data_dict_before_pad[f"/observations/images/{cam_name}"][t]
                assert stored == expected_flat.tobytes(), f"{cam_name} t={t} JPEG bytes mismatch"
                dec = cv2.imdecode(np.frombuffer(stored, dtype=np.uint8), cv2.IMREAD_COLOR)
                assert dec is not None and dec.shape[:2] == (h, w)

    tmp.unlink(missing_ok=True)
    tmp.parent.rmdir()
    print("test_record_jpeg_pipeline: OK (encode + len + pad + hdf5 + decode roundtrip)")


if __name__ == "__main__":
    main()
