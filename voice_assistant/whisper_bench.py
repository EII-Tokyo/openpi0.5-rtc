import os
import time

import numpy as np
import torch
import whisper


def _run(model, mel, device, iters=3):
    # Warmup
    for _ in range(2):
        _ = model.decode(mel, whisper.DecodingOptions(language="en"))
        if device == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = model.decode(mel, whisper.DecodingOptions(language="en"))
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = os.getenv("WHISPER_MODEL", "medium")
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name, device=device)

    # Use model-specific mel channels (80 or 128) for large-v3/variants.
    mel = torch.zeros((model.dims.n_mels, 3000), device=device)

    times = _run(model, mel, device)
    print("Decode times (s):", ", ".join(f"{t:.4f}" for t in times))
    print(f"Avg: {sum(times) / len(times):.4f}s")


if __name__ == "__main__":
    main()
