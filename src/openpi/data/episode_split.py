"""Whole-episode training selection shared by sampling and normalization."""
import numpy as np


def training_episode_mask(repo_id, episode_indices, selections):
    episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    if selections is None:
        return np.ones(len(episodes), dtype=bool)
    if repo_id not in selections:
        raise ValueError(f"Missing training episode selection for {repo_id}")
    selected = selections[repo_id]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f"Empty or duplicate training episodes for {repo_id}")
    missing = set(selected) - set(episodes.tolist())
    if missing:
        raise ValueError(f"Unknown training episodes for {repo_id}: {sorted(missing)}")
    return np.isin(episodes, selected)
