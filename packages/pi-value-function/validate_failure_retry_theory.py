"""Validate failure->retry theory using raw GCS bucket.

For each sampled failure episode:
 - Check if recordings/MP4/ exists (else "SVO so skip")
 - Find the nearest REAL annotation entry after the failure's datetime (= suspected success)
 - Construct the success episode GCS path from the annotation datetime
 - Download videos, overlay labels, export to paired_videos/XXX/
"""

import calendar
import json
import pathlib
import random
import re
import subprocess
import tempfile
from datetime import datetime

import cv2
import numpy as np

ANNOTATIONS_PATH = "/tmp/droid_annotations.json"
FAILURE_LIST     = "/tmp/droid_failure_episodes.txt"
OUT_BASE         = pathlib.Path(
    "/home/eii/Desktop/openpi0.5-rtc/packages/pi-value-function/paired_videos"
)
OUT_BASE.mkdir(exist_ok=True)

N_PAIRS = 10
FPS     = 10
SEED    = 42

# ── datetime helpers ──────────────────────────────────────────────────────────

MONTH_TO_NUM = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def parse_gcs_folder_dt(name: str) -> datetime | None:
    """Parse GCS episode folder name to datetime."""
    # Format A: Thu_Apr__6_14_18_04_2023  (underscores for time, double _ before single-digit day)
    m = re.match(r"[A-Za-z]+_([A-Za-z]+)_+(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$", name)
    if m:
        mo, d, H, M, S, Y = m.groups()
        num = MONTH_TO_NUM.get(mo)
        if num:
            return datetime(int(Y), int(num), int(d), int(H), int(M), int(S))
    # Format B: Fri_May_12_13:43:15_2023  (colons for time)
    m = re.match(r"[A-Za-z]+_([A-Za-z]+)_(\d+)_(\d+):(\d+):(\d+)_(\d+)$", name)
    if m:
        mo, d, H, M, S, Y = m.groups()
        num = MONTH_TO_NUM.get(mo)
        if num:
            return datetime(int(Y), int(num), int(d), int(H), int(M), int(S))
    return None


def annotation_key_to_dt(dt_str: str) -> datetime | None:
    """Parse annotation key datetime: '2023-04-06-14h-18m-04s'."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h-(\d{2})m-(\d{2})s$", dt_str)
    if m:
        Y, mo, d, H, M, S = m.groups()
        return datetime(int(Y), int(mo), int(d), int(H), int(M), int(S))
    return None


def dt_to_gcs_folder(dt: datetime) -> str:
    """Convert datetime back to the colon-format GCS folder name."""
    dow   = calendar.day_abbr[dt.weekday()]   # Mon, Tue, …
    month = calendar.month_abbr[dt.month]     # Jan, Feb, …
    return f"{dow}_{month}_{dt.day}_{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}_{dt.year}"


def dt_to_date_dir(dt: datetime) -> str:
    return f"{dt.year}-{dt.month:02d}-{dt.day:02d}"


# ── GCS helpers ───────────────────────────────────────────────────────────────

def gcs_ls(path: str) -> list[str]:
    r = subprocess.run(["gsutil", "ls", path], capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def gcs_download(gcs_path: str, local_path: pathlib.Path) -> bool:
    r = subprocess.run(["gsutil", "-q", "cp", gcs_path, str(local_path)])
    return r.returncode == 0


# ── video helpers ─────────────────────────────────────────────────────────────

def overlay_text(frame_bgr: np.ndarray, text: str, h: int) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    # word-wrap at ~80 chars
    lines = []
    while len(text) > 80:
        cut = text[:80].rfind(" ")
        if cut == -1:
            cut = 80
        lines.append(text[:cut])
        text = text[cut:].lstrip()
    lines.append(text)
    for i, line in enumerate(lines):
        y = h - 10 - (len(lines) - 1 - i) * 18
        cv2.putText(frame_bgr, line, (8, y), font, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (8, y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


def write_video_with_label(src_mp4: pathlib.Path, out_mp4: pathlib.Path, label: str):
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        print(f"    Could not open {src_mp4.name}")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or FPS
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(overlay_text(frame, label, h))
    cap.release()
    writer.release()


# ── load data ─────────────────────────────────────────────────────────────────

with open(ANNOTATIONS_PATH) as f:
    annotations = json.load(f)

# Build sorted list of REAL annotation entries (datetime, collector_id, annotation)
real_entries: list[tuple[datetime, str, str, dict]] = []
for key, ann in annotations.items():
    parts = key.split("+")
    if len(parts) == 3 and parts[0] == "REAL":
        dt = annotation_key_to_dt(parts[2])
        if dt:
            real_entries.append((dt, key, parts[1], ann))
real_entries.sort(key=lambda x: x[0])
print(f"REAL annotation entries: {len(real_entries)}")

with open(FAILURE_LIST) as f:
    failure_folders = [l.strip() for l in f if l.strip()]

rng = random.Random(SEED)
rng.shuffle(failure_folders)

# ── main loop: sample failures, find success, export ─────────────────────────

pair_num = 0
idx = 0

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = pathlib.Path(tmpdir)

    while pair_num < N_PAIRS and idx < len(failure_folders):
        folder = failure_folders[idx].rstrip("/")
        idx += 1
        folder_name = folder.split("/")[-1]

        # Parse datetime
        dt_fail = parse_gcs_folder_dt(folder_name)
        if dt_fail is None:
            print(f"[{idx}] Cannot parse datetime: {folder_name}")
            continue

        # Check for MP4s
        mp4_files = [p for p in gcs_ls(folder + "/recordings/MP4/") if p.endswith(".mp4")]
        if not mp4_files:
            print(f"[{idx}] SVO so skip: {folder_name}")
            continue

        # Find nearest REAL annotation AFTER this failure
        next_ann = next(
            ((dt, key, cid, ann) for dt, key, cid, ann in real_entries if dt > dt_fail),
            None,
        )
        if next_ann is None:
            print(f"[{idx}] No success annotation after {dt_fail}, skipping")
            continue

        dt_succ, succ_key, succ_cid, succ_ann = next_ann
        prompt = (succ_ann.get("language_instruction1") or
                  succ_ann.get("language_instruction2") or "")

        # Construct success episode GCS path and check for MP4s
        succ_folder_gs = (
            f"gs://gresearch/robotics/droid_raw/1.0.1/REAL/success"
            f"/{dt_to_date_dir(dt_succ)}/{dt_to_gcs_folder(dt_succ)}/recordings/MP4/"
        )
        succ_mp4_files = [p for p in gcs_ls(succ_folder_gs) if p.endswith(".mp4")]
        if not succ_mp4_files:
            print(f"[{idx}] Success episode has no MP4 at {succ_folder_gs}, skipping")
            continue

        pair_num += 1
        pair_label = f"{pair_num:03d}"
        out_dir = OUT_BASE / pair_label
        out_dir.mkdir(exist_ok=True)

        gap_s = int((dt_succ - dt_fail).total_seconds())
        print(f"\nPair {pair_label}: gap={gap_s}s")
        print(f"  FAIL {folder_name}")
        print(f"  SUCC {dt_to_gcs_folder(dt_succ)}")
        print(f"  Prompt: '{prompt}'")

        # Download & label failure MP4s (use first camera only to keep it simple)
        fail_src = mp4_files[0]
        cam_id = fail_src.rstrip("/").split("/")[-1].replace(".mp4", "")
        fail_local = tmp / f"fail_{pair_label}_{cam_id}.mp4"
        print(f"  Downloading FAIL camera {cam_id} ...")
        if gcs_download(fail_src, fail_local):
            write_video_with_label(
                fail_local,
                out_dir / f"ep_FAIL_{folder_name[:40]}.mp4",
                f"FAIL | suspected prompt: {prompt}",
            )

        # Download & label success MP4s (same camera id if available, else first)
        succ_src = next((p for p in succ_mp4_files if cam_id in p), succ_mp4_files[0])
        succ_cam_id = succ_src.rstrip("/").split("/")[-1].replace(".mp4", "")
        succ_local = tmp / f"succ_{pair_label}_{succ_cam_id}.mp4"
        print(f"  Downloading SUCC camera {succ_cam_id} ...")
        if gcs_download(succ_src, succ_local):
            write_video_with_label(
                succ_local,
                out_dir / f"ep_SUCC_{dt_to_gcs_folder(dt_succ)[:40]}.mp4",
                f"SUCC | {prompt}",
            )

print(f"\nDone. {pair_num} pairs saved to {OUT_BASE}/")
