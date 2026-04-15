import argparse
import io
import json
import statistics
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from openpi_client.runtime import gpt_high_level_policy as _gpt_policy


def _fetch_jpeg(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = response.read()
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _fetch_camera_images(api_base: str) -> dict[str, np.ndarray]:
    out = {}
    for camera_name in _gpt_policy.CAMERA_ORDER:
        url = f"{api_base.rstrip('/')}/api/cameras/{camera_name}/latest.jpg"
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = response.read()
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        out[camera_name] = np.asarray(image, dtype=np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8011")
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-mode", choices=("responses", "chat_completions"), default=None)
    parser.add_argument("--models", default="Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B")
    parser.add_argument("--image-modes", default="high_only,all_cameras")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", default="Process all bottles")
    parser.add_argument("--locked-bottle-description", default=None)
    parser.add_argument("--dump-results", default=None)
    args = parser.parse_args()

    images = _fetch_camera_images(args.api_base)
    policy = _gpt_policy.GptHighLevelPolicy(
        api_base=args.llm_api_base,
        api_mode=args.llm_api_mode,
    )
    policy.set_low_level_schema(
        state_subtask_pairs=[
            ("Bottle on table, opening faces left", "Rotate so opening faces right"),
            ("Bottle on table, opening faces right", "Pick up with left hand"),
            ("Bottle in left hand and capped", "Unscrew cap"),
            ("Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"),
            ("Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"),
            ("Bottle in left hand and upside down", "Bottle to left trash bin"),
            ("Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"),
            ("Cap on table", "Pick up cap and place into right trash bin"),
            ("No bottle on table", "Return to initial pose"),
        ],
        start_subtasks=("Rotate so opening faces right", "Pick up with left hand"),
    )
    obs = {
        "prompt": args.prompt,
        "images": images,
        "runtime_context": {
            "locked_bottle_description": args.locked_bottle_description,
        },
    }

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for image_mode in [m.strip() for m in args.image_modes.split(",") if m.strip()]:
            policy.set_config(model=model, image_mode=image_mode)
            durations = []
            last_result = None
            for _ in range(max(1, args.runs)):
                started_at = time.perf_counter()
                try:
                    result = policy.infer_subtask(obs)
                except (RuntimeError, urllib.error.URLError) as exc:
                    print(
                        json.dumps(
                            {
                                "model": model,
                                "image_mode": image_mode,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                    )
                    break
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                durations.append(elapsed_ms)
                last_result = result
                rows.append(
                    {
                        "model": model,
                        "image_mode": image_mode,
                        "elapsed_ms": round(elapsed_ms, 1),
                        "subtask_text": result.get("subtask_text"),
                        "server_timing": result.get("server_timing"),
                    }
                )
            if not durations:
                continue
            print(
                json.dumps(
                    {
                        "model": model,
                        "image_mode": image_mode,
                        "runs": len(durations),
                        "mean_ms": round(statistics.mean(durations), 1),
                        "median_ms": round(statistics.median(durations), 1),
                        "min_ms": round(min(durations), 1),
                        "max_ms": round(max(durations), 1),
                        "last_subtask_text": None if last_result is None else last_result.get("subtask_text"),
                    },
                    ensure_ascii=False,
                )
            )
    if args.dump_results:
        with open(args.dump_results, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
