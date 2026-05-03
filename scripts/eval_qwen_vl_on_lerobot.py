#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_high_level_utils import (
    DEFAULT_CLASS_SPEC_PATH,
    describe_task_mode,
    infer_task_mode_from_repo_id,
    load_class_map,
    parse_class_id,
    row_class_id,
    system_prompt,
)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None

DEFAULT_CAMERA_COLUMNS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, dict):
        image_bytes = x.get("bytes")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        if x.get("path"):
            return Image.open(x["path"]).convert("RGB")
    t = x
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):
        t = t.transpose(1, 2, 0)
    if t.dtype != "uint8":
        if float(t.max()) <= 1.0:
            t = (t * 255.0).clip(0, 255).astype("uint8")
        else:
            t = t.astype("uint8")
    return Image.fromarray(t).convert("RGB")


def _pick_indices(ds, max_samples: int | None, seed: int) -> list[int]:
    if max_samples is None or max_samples >= len(ds):
        return list(range(len(ds)))
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    return indices[:max_samples]


class HfParquetDataset:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.files = sorted(
            filename
            for filename in HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
            if filename.startswith("data/") and filename.endswith(".parquet")
        )
        if not self.files:
            raise FileNotFoundError(f"No parquet files found in dataset repo: {repo_id}")
        self.paths = [hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset") for filename in self.files]
        self.row_counts = [pq.ParquetFile(path).metadata.num_rows for path in self.paths]
        self.cumulative = []
        total = 0
        for count in self.row_counts:
            total += count
            self.cumulative.append(total)
        self._tables = {}

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        prev = 0
        for file_idx, end in enumerate(self.cumulative):
            if idx < end:
                local_idx = idx - prev
                table = self._tables.get(file_idx)
                if table is None:
                    table = pq.read_table(self.paths[file_idx])
                    self._tables[file_idx] = table
                return dict(table.slice(local_idx, 1).to_pylist()[0])
            prev = end
        raise IndexError(idx)


def _load_eval_dataset(repo_id: str, root: Path | None):
    if root is not None:
        if LeRobotDataset is None:
            raise ImportError("lerobot is required when --root is provided")
        return LeRobotDataset(repo_id, root=root, force_cache_sync=False, download_videos=False, delta_timestamps=None)
    return HfParquetDataset(repo_id)


def _camera_name(column: str) -> str:
    return column.rsplit(".", 1)[-1]


def _user_prompt(camera_columns: tuple[str, ...]) -> str:
    camera_names = ", ".join(_camera_name(column) for column in camera_columns)
    return f"Use all {len(camera_columns)} images in this order: {camera_names}. Classify the current scene."


def _scalar_int(x) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        return int(x[0])
    return int(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default=None, help="Path to a PEFT LoRA checkpoint directory.")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--camera-columns", nargs="+", default=list(DEFAULT_CAMERA_COLUMNS))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--class-spec", type=Path, default=DEFAULT_CLASS_SPEC_PATH)
    args = ap.parse_args()

    ds = _load_eval_dataset(args.repo_id, args.root)
    class_map = load_class_map(args.class_spec)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    camera_columns = tuple(args.camera_columns)
    task_mode = infer_task_mode_from_repo_id(args.repo_id)
    prompt = system_prompt(class_map, task_mode)
    user_prompt = f"{_user_prompt(camera_columns)} Known task mode: {describe_task_mode(task_mode)}."
    indices = _pick_indices(ds, args.max_samples, args.seed)
    total = 0
    correct = 0
    latencies = []
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    pred_counter = Counter()
    parse_failures = 0
    raw_examples = []

    for i, idx in enumerate(indices):
        sample = ds[idx]
        true_id = row_class_id(sample, class_map)
        if true_id is None:
            raise KeyError("Cannot infer class_id from row.")
        images = [_to_pil(sample[column]) for column in camera_columns]
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": user_prompt}],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt")
        inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        pred_id = parse_class_id(raw_text, class_map)

        total += 1
        per_class[true_id]["total"] += 1
        if pred_id is None:
            parse_failures += 1
        else:
            pred_counter[pred_id] += 1
            if pred_id == true_id:
                correct += 1
                per_class[true_id]["correct"] += 1

        if len(raw_examples) < 20:
            raw_examples.append(
                {
                    "idx": int(idx),
                    "true_id": true_id,
                    "pred_id": pred_id,
                    "raw_text": raw_text,
                }
            )
        if (i + 1) % 20 == 0:
            print(f"processed {i+1}/{len(indices)} acc={correct/max(total,1):.4f} avg_latency={sum(latencies)/len(latencies):.3f}s", flush=True)

    result = {
        "repo_id": args.repo_id,
        "model_id": args.model_id,
        "adapter": args.adapter,
        "device": args.device,
        "camera_columns": list(camera_columns),
        "task_mode": task_mode,
        "num_samples": total,
        "accuracy": correct / max(total, 1),
        "avg_latency_sec": sum(latencies) / max(len(latencies), 1),
        "median_latency_sec": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "parse_failures": parse_failures,
        "pred_counts": dict(sorted(pred_counter.items())),
        "per_class": {
            str(cid): {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / max(stats["total"], 1),
                "label": {"bottle_state": class_map[cid][0], "subtask": class_map[cid][1]},
            }
            for cid, stats in sorted(per_class.items())
        },
        "examples": raw_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
