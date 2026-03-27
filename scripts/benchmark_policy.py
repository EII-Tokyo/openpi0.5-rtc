"""Benchmark client for measuring policy serving latency.

Usage:
    uv run scripts/benchmark_policy.py --port 8000 --num-requests 50
    uv run scripts/benchmark_policy.py --port 8000 --num-requests 50 --use-rtc
"""

import argparse
import logging
import time

import numpy as np
import websockets.sync.client

# Use vendored msgpack_numpy from openpi_client
from openpi_client import msgpack_numpy


def make_droid_example() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def percentile_str(values: list[float], name: str) -> str:
    arr = np.array(values)
    return (
        f"{name}: "
        f"mean={np.mean(arr):.1f}ms  "
        f"min={np.min(arr):.1f}ms  "
        f"max={np.max(arr):.1f}ms  "
        f"p50={np.percentile(arr, 50):.1f}ms  "
        f"p95={np.percentile(arr, 95):.1f}ms  "
        f"p99={np.percentile(arr, 99):.1f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark policy server latency")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup requests to discard")
    parser.add_argument("--use-rtc", action="store_true", help="Enable RTC (real-time correction)")
    parser.add_argument("--prev-action-shape", type=str, default="16,32", help="Shape of prev_action array (comma-separated, should match model's action_horizon x action_dim)")
    args = parser.parse_args()

    uri = f"ws://{args.host}:{args.port}"
    packer = msgpack_numpy.Packer()

    logging.info("Connecting to %s ...", uri)
    ws = websockets.sync.client.connect(uri, compression=None, max_size=None)
    metadata = msgpack_numpy.unpackb(ws.recv())
    logging.info("Connected. Server metadata: %s", metadata)

    obs = make_droid_example()
    prev_action_shape = tuple(int(x) for x in args.prev_action_shape.split(","))
    prev_action = np.random.rand(*prev_action_shape) if args.use_rtc else None

    total_requests = args.warmup + args.num_requests
    round_trip_times = []
    server_timings = []

    for i in range(total_requests):
        data = {"obs": obs, "prev_action": prev_action, "use_rtc": args.use_rtc}
        packed = packer.pack(data)

        t_start = time.monotonic()
        ws.send(packed)
        response = ws.recv()
        t_end = time.monotonic()

        if isinstance(response, str):
            logging.error("Server error: %s", response)
            break

        result = msgpack_numpy.unpackb(response)
        rtt_ms = (t_end - t_start) * 1000

        is_warmup = i < args.warmup
        label = "warmup" if is_warmup else "bench"

        if not is_warmup:
            round_trip_times.append(rtt_ms)
            if "policy_timing" in result:
                server_timings.append(result["policy_timing"])

        policy_timing = result.get("policy_timing", {})
        server_timing = result.get("server_timing", {})
        logging.info(
            "[%s %3d] rtt=%.1fms  model=%.1fms  server_total=%.1fms",
            label, i,
            rtt_ms,
            policy_timing.get("model_infer_ms", 0),
            server_timing.get("total_ms", 0),
        )

    ws.close()

    if not round_trip_times:
        logging.warning("No benchmark data collected.")
        return

    print("\n" + "=" * 70)
    print(f"Benchmark results (n={len(round_trip_times)}, use_rtc={args.use_rtc})")
    print("=" * 70)

    print(percentile_str(round_trip_times, "Client round-trip"))

    if server_timings:
        print("\nServer-side breakdown:")
        keys = list(server_timings[0].keys())
        for k in keys:
            vals = [t[k] for t in server_timings]
            print(f"  {percentile_str(vals, k)}")

    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
