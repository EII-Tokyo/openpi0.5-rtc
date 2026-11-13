"""Audited v2.1 FastUMI episode reader and deterministic group-balanced sampling."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler


class GroupSampler(Sampler):
    """Uniform frames within each group; exact 50/50 batches and bounded rounding otherwise."""

    def __init__(self, fan_frames, public_frames, batch_size, fan_fraction, *, seed=42, batches=None):
        if min(fan_frames, public_frames, batch_size) <= 0 or not 0 < fan_fraction < 1:
            raise ValueError("Both groups and batch size must be positive, with fraction strictly between zero and one")
        self.fan_frames, self.public_frames = int(fan_frames), int(public_frames)
        self.batch_size, self.fan_fraction, self.seed = int(batch_size), fan_fraction, seed
        self.batches = batches if batches is not None else (fan_frames + public_frames) // batch_size
        if self.batches <= 0:
            raise ValueError("At least one batch is required")
        self.epoch = 0

    def __len__(self):
        return self.batches * self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        self.epoch += 1
        previous = 0
        for batch in range(self.batches):
            cumulative = round((batch + 1) * self.batch_size * self.fan_fraction)
            count = cumulative - previous
            previous = cumulative
            indices = np.concatenate(
                [
                    rng.integers(0, self.fan_frames, count),
                    self.fan_frames + rng.integers(0, self.public_frames, self.batch_size - count),
                ]
            )
            rng.shuffle(indices)
            yield from map(int, indices)


class FastUMIEpisodes:
    """Read numeric arrays from audited episodes and decode only the current RGB frame."""

    def __init__(self, root, phase, horizon=16, *, numeric_only=False):
        if phase not in ("midtrain", "posttrain"):
            raise ValueError(phase)
        self.root = Path(root or os.environ.get("FASTUMI_TWO_PHASE_ROOT", ".cache/fastumi-mid-post"))
        report = json.loads((self.root / "ready.json").read_text())
        index_path = self.root / "episodes.json"
        if not report["ready"] or hashlib.sha256(index_path.read_bytes()).hexdigest() != report["index_sha256"]:
            raise ValueError("Prepared dataset index differs from the audited version")
        index = json.loads(index_path.read_text())
        self.episodes = [e for e in index if e["group"] == "fan" or e["phase"] == phase]
        self.episodes.sort(key=lambda e: (e["group"] != "fan", e["source"], e["episode"]))
        self.ends = np.cumsum([e["length"] for e in self.episodes])
        self.fan_frames = sum(e["length"] for e in self.episodes if e["group"] == "fan")
        self.public_frames = int(self.ends[-1]) - self.fan_frames
        self.horizon, self.numeric_only = horizon, numeric_only
        self._cache = {}
        self._arrays = {}

    def __len__(self):
        return int(self.ends[-1])

    def _locate(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        offset = int(index - (self.ends[episode - 1] if episode else 0))
        return self.episodes[episode], offset

    def get_sample_context(self, index):
        episode, frame = self._locate(index)
        return {"repo_id": episode["source"], "episode_index": episode["episode"], "frame_index": frame}

    def __getitem__(self, index):
        episode, frame = self._locate(index)
        key = episode["numeric"]
        if "array_start" in episode:
            if key not in self._arrays:
                self._arrays[key] = {
                    name: np.load(self.root / key / f"{name}.npy", mmap_mode="r")
                    for name in ("state", "action", "timestamp", "task_index")
                }
            data = self._arrays[key]
            start = episode["array_start"]
        else:
            if key not in self._cache:
                if len(self._cache) >= 32:
                    self._cache.pop(next(iter(self._cache)))
                with np.load(self.root / key, allow_pickle=False) as source:
                    self._cache[key] = {k: source[k] for k in source.files}
            data = self._cache[key]
            start = 0
        targets = start + np.minimum(frame + episode["action_offset"] + np.arange(self.horizon), episode["length"] - 1)
        frame += start
        if self.numeric_only:
            image = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            from lerobot.datasets.video_utils import decode_video_frames  # noqa: PLC0415

            image = decode_video_frames(episode["video"], [float(data["timestamp"][frame])], 1e-3, backend="pyav")[0]
        return {
            "observation/image": image,
            "observation/state": data["state"][frame],
            "actions": data["action"][targets],
            "prompt": episode["tasks"][str(int(data["task_index"][frame]))],
        }
