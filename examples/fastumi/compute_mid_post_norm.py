"""Estimate normalization from a seeded midtraining mixture without decoding videos."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tyro

from openpi.policies.fastumi_policy import FastUMIRelativeInputs
from openpi.shared import normalize
from openpi.training.fastumi_two_phase import FastUMIEpisodes
from openpi.training.fastumi_two_phase import GroupSampler


def main(
    root: Path,
    samples: int = 1_048_576,
    seed: int = 42,
    output: Path = Path("assets/pi05_fastumi_midtrain/fastumi_mid_post"),
):
    if samples < 1024 or samples % 1024:
        raise ValueError("Normalization samples must be a positive multiple of 1024")
    dataset = FastUMIEpisodes(root, "midtrain", numeric_only=True)
    sampler = GroupSampler(dataset.fan_frames, dataset.public_frames, 1024, 0.1, seed=seed, batches=samples // 1024)
    transform = FastUMIRelativeInputs(prompt=None)
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    batches = {k: [] for k in stats}
    for count, index in enumerate(sampler, 1):
        sample = transform(dataset[index])
        for key in stats:
            batches[key].append(sample[key])
        if count % 1024 == 0:
            for key, accumulator in stats.items():
                accumulator.update(np.stack(batches[key]))
                batches[key].clear()
        if count % 65536 == 0:
            print(f"Normalization: {count}/{samples} midtraining samples", flush=True)
    result = {key: stat.get_statistics() for key, stat in stats.items()}
    for stat in result.values():
        if (
            not np.isfinite(stat.mean).all()
            or not np.isfinite(stat.std).all()
            or not np.all(stat.q99 - stat.q01 > 1e-5)
        ):
            raise ValueError("Degenerate normalization statistics")
    normalize.save(output, result)
    provenance = {
        "phase": "midtrain",
        "samples": samples,
        "seed": seed,
        "fan_fraction": 0.1,
        "index_sha256": hashlib.sha256((root / "episodes.json").read_bytes()).hexdigest(),
        "norm_sha256": hashlib.sha256((output / "norm_stats.json").read_bytes()).hexdigest(),
        "posttrain_public_samples": 0,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Midtraining-only normalization ready; reuse unchanged for posttraining", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
