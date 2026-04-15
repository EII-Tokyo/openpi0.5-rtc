#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoModelForImageTextToText, AutoProcessor


CLASS_MAP = {
    0: ("Bottle on table, opening faces left", "Rotate so opening faces right"),
    1: ("Bottle on table, opening faces right", "Pick up with left hand"),
    2: ("Bottle in left hand and capped", "Unscrew cap"),
    3: ("Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"),
    4: ("Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"),
    5: ("Bottle in left hand and upside down", "Bottle to left trash bin"),
    6: ("Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"),
    7: ("Cap on table", "Pick up cap and place into right trash bin"),
    8: ("No bottle on table", "Return to initial pose"),
}


def _to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
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


def _pick_indices(ds: LeRobotDataset, max_samples: int | None, seed: int) -> list[int]:
    if max_samples is None or max_samples >= len(ds):
        return list(range(len(ds)))
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    return indices[:max_samples]


def _prompt() -> str:
    lines = [
        "Classify the robot scene into exactly one of 9 classes.",
        "Use all four images in this order: cam_high, cam_low, cam_left_wrist, cam_right_wrist.",
        "Output exactly one character: 0 or 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8.",
        "Do not output words.",
        "Do not output punctuation.",
        "Do not explain your answer.",
        "Classes:",
    ]
    for cid, (state, subtask) in CLASS_MAP.items():
        lines.append(f"{cid}: state={state}; action={subtask}")
    return "\n".join(lines)


def _parse_class_id(text: str) -> int | None:
    m = re.search(r"\b([0-8])\b", text)
    return int(m.group(1)) if m else None


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
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.root, force_cache_sync=False, download_videos=False, delta_timestamps=None)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompt = _prompt()
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
        true_id = _scalar_int(sample["class_id"])
        images = [
            _to_pil(sample["observation.images.cam_high"]),
            _to_pil(sample["observation.images.cam_low"]),
            _to_pil(sample["observation.images.cam_left_wrist"]),
            _to_pil(sample["observation.images.cam_right_wrist"]),
        ]
        messages = [{"role": "user", "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]}]
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
        pred_id = _parse_class_id(raw_text)

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
        "device": args.device,
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
                "label": {"bottle_state": CLASS_MAP[cid][0], "subtask": CLASS_MAP[cid][1]},
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
