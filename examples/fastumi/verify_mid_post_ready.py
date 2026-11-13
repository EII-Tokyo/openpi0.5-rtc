"""Final source, normalization, image, and two-phase configuration checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tyro

from openpi.policies.fastumi_policy import FastUMIRelativeInputs
from openpi.shared.normalize import load
from openpi.training.config import get_config
from openpi.training.data_loader import transform_dataset
from openpi.training.fastumi_two_phase import FastUMIEpisodes


def main(root: Path, split: Path = Path("examples/fastumi/mid_post_public_split.proposed.json")):
    report = json.loads((root / "ready.json").read_text())
    if report["split_sha256"] != hashlib.sha256(split.read_bytes()).hexdigest():
        raise ValueError("Split changed after preparation")
    norm_path = Path("assets/pi05_fastumi_midtrain/fastumi_mid_post")
    provenance = json.loads((norm_path / "provenance.json").read_text())
    if provenance["index_sha256"] != report["index_sha256"] or provenance["posttrain_public_samples"] != 0:
        raise ValueError("Normalization provenance disagrees with prepared sources")
    if provenance["norm_sha256"] != hashlib.sha256((norm_path / "norm_stats.json").read_bytes()).hexdigest():
        raise ValueError("Normalization changed after computation")
    stats = load(norm_path)
    if stats["state"].mean.shape != (1,) or stats["actions"].mean.shape != (10,):
        raise ValueError("Incorrect normalization dimensions")
    transform = FastUMIRelativeInputs(prompt=None)
    for phase, steps, fraction in (("midtrain", 200000, 0.1), ("posttrain", 100000, 0.5)):
        cfg = get_config(f"pi05_fastumi_{phase}")
        if cfg.num_train_steps != steps or cfg.lr_schedule.decay_steps != steps or cfg.data.fan_fraction != fraction:
            raise ValueError("Phase config changed from approved values")
        ds = FastUMIEpisodes(root, phase)
        # Real video/pose smoke checks at the start, middle, and end of one episode per source per phase.
        checked = set()
        for episode_index, episode in enumerate(ds.episodes):
            if episode["source"] in checked:
                continue
            checked.add(episode["source"])
            start = int(ds.ends[episode_index - 1]) if episode_index else 0
            for frame in (0, episode["length"] // 2, episode["length"] - 1):
                sample = transform(ds[start + frame])
                if sample["actions"].shape != (16, 10) or sample["state"].shape != (1,):
                    raise ValueError("Invalid real sample shapes")
                if not np.isfinite(sample["actions"]).all() or not str(sample["prompt"]).strip():
                    raise ValueError("Invalid target or instruction")
                if sample["image"]["left_wrist_0_rgb"].shape != (720, 1280, 3):
                    raise ValueError("Invalid decoded wrist image")
        model_sample = transform_dataset(ds, cfg.data.create(cfg.assets_dirs, cfg.model))[ds.fan_frames]
        if model_sample["state"].shape != (32,) or model_sample["actions"].shape != (16, 32):
            raise ValueError("Full training transform produced incorrect padded model shapes")
        print(f"{phase}: {len(ds)} frames, {len(checked)} sources; image/pose boundary checks passed", flush=True)
    print("PRELAUNCH CHECKS PASSED", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
