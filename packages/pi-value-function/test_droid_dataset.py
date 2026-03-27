"""Save failure episode videos from cadene/droid_1.0.1_v30."""

import pathlib
import numpy as np
import torch
import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "cadene/droid_1.0.1_v30"
EPISODES = [93000, 94000, 95000]
N_FAILURE_EPS = 4   # how many failure episodes to export
OUT_DIR = pathlib.Path("/home/eii/Desktop/openpi0.5-rtc/packages/pi-value-function/failure_videos")
OUT_DIR.mkdir(exist_ok=True)

# Cameras to show side-by-side (exterior_2 = right wrist, wrist = left wrist)
CAMERAS = [
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
]

print(f"Loading {REPO_ID} episodes {EPISODES} ...")
ds = LeRobotDataset(REPO_ID, episodes=EPISODES)
print(f"Total frames in loaded chunks: {len(ds)}")

hf = ds.hf_dataset
ep_indices = [int(e) for e in hf["episode_index"]]
successful = [bool(s) for s in hf["is_episode_successful"]]

ep_to_rows: dict[int, list[int]] = {}
ep_success: dict[int, bool] = {}
for row_i, (ep_id, succ) in enumerate(zip(ep_indices, successful)):
    ep_to_rows.setdefault(ep_id, []).append(row_i)
    ep_success[ep_id] = succ

failure_eps = [ep for ep, succ in ep_success.items() if not succ]
print(f"Failure episodes in loaded chunks: {len(failure_eps)}")

if not failure_eps:
    print("No failure episodes found. Try different EPISODES.")
    exit(0)

rng = np.random.default_rng(0)
chosen = sorted(rng.choice(failure_eps, size=min(N_FAILURE_EPS, len(failure_eps)), replace=False).tolist())
print(f"Exporting failure episodes: {chosen}")


def to_hwc_uint8(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


for ep_id in chosen:
    rows = ep_to_rows[ep_id]
    sample0 = ds[rows[0]]
    task = (sample0.get("task") or sample0.get("language_instruction") or
            sample0.get("task_category") or "no task")

    # Get frame dimensions from first frame
    frames_0 = [to_hwc_uint8(sample0[c]) for c in CAMERAS]
    h, w = frames_0[0].shape[:2]
    total_w = w * len(CAMERAS)

    out_path = OUT_DIR / f"ep{ep_id:06d}_failure.mp4"
    fps = 10
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, h))

    # Print all text fields for diagnosis
    text_fields = {k: v for k, v in sample0.items() if isinstance(v, str)}
    print(f"  ep={ep_id} ({len(rows)} frames) -> {out_path.name}")
    for k, v in text_fields.items():
        print(f"    {k}: '{v}'")

    for row_idx in rows:
        sample = ds[row_idx]
        imgs = [to_hwc_uint8(sample[c]) for c in CAMERAS]
        frame = np.concatenate(imgs, axis=1)          # side-by-side
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Overlay prompt text
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"FAIL | {task[:80]}"
        cv2.putText(frame_bgr, label, (8, h - 10), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, label, (8, h - 10), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame_bgr)

    writer.release()

print(f"\nDone. Videos saved to {OUT_DIR}/")
