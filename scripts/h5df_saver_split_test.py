import importlib
import sys
import types

import numpy as np
import pytest


RESET_POSITION = [
    [0.0, -0.96, 1.16, 1.57, 0.0, -1.57],
    [0.0, -0.96, 1.16, 0.0, 0.0, 0.0],
]


@pytest.fixture
def h5df_saver_module(monkeypatch):
    subscriber_mod = types.ModuleType("openpi_client.runtime.subscriber")

    class Subscriber:
        pass

    subscriber_mod.Subscriber = Subscriber
    runtime_mod = types.ModuleType("openpi_client.runtime")
    runtime_mod.subscriber = subscriber_mod
    openpi_client_mod = types.ModuleType("openpi_client")
    openpi_client_mod.runtime = runtime_mod
    hdf5_utils_mod = types.ModuleType("examples.aloha_real.hdf5_utils")
    typing_extensions_mod = types.ModuleType("typing_extensions")
    typing_extensions_mod.override = lambda fn: fn

    def placeholder_save_hdf5_episode(*args, **kwargs):
        raise AssertionError("save_hdf5_episode should be monkeypatched by the test")

    hdf5_utils_mod.save_hdf5_episode = placeholder_save_hdf5_episode

    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client_mod)
    monkeypatch.setitem(sys.modules, "openpi_client.runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "openpi_client.runtime.subscriber", subscriber_mod)
    monkeypatch.setitem(sys.modules, "examples.aloha_real.hdf5_utils", hdf5_utils_mod)
    monkeypatch.setitem(sys.modules, "typing_extensions", typing_extensions_mod)
    sys.modules.pop("examples.aloha_real.h5df_saver", None)
    return importlib.import_module("examples.aloha_real.h5df_saver")


@pytest.fixture
def saved_episodes(h5df_saver_module, monkeypatch):
    episodes = []

    def fake_save_hdf5_episode(observations, actions, dataset_dir, **kwargs):
        episodes.append(
            {
                "observations": list(observations),
                "actions": list(actions),
                "dataset_dir": dataset_dir,
                "kwargs": kwargs,
            }
        )
        return dataset_dir / f"episode_{len(episodes) - 1}.hdf5", None

    monkeypatch.setattr(h5df_saver_module._hdf5_utils, "save_hdf5_episode", fake_save_hdf5_episode)
    return episodes


def _qpos(offset: float = 0.0) -> np.ndarray:
    qpos = np.zeros(14, dtype=np.float32)
    qpos[:6] = np.asarray(RESET_POSITION[0], dtype=np.float32)
    qpos[7:13] = np.asarray(RESET_POSITION[1], dtype=np.float32)
    qpos[0] += offset
    return qpos


def _observation(offset: float = 0.0) -> dict:
    qpos = _qpos(offset)
    return {
        "qpos": qpos,
        "qvel": np.zeros_like(qpos),
        "effort": np.zeros_like(qpos),
        "images": {},
    }


def _action() -> dict:
    return {"actions": np.ones(14, dtype=np.float32)}


def _split_saver(h5df_saver_module, tmp_path, **kwargs):
    params = {
        "home_threshold": 0.05,
        "leave_threshold": 0.20,
        "stable_home_steps": 2,
        "min_episode_steps": 3,
    }
    params.update(kwargs)
    return h5df_saver_module.H5dfSaver(
        dataset_dir=tmp_path,
        compress_images=False,
        split_on_reset=True,
        reset_position=RESET_POSITION,
        **params,
    )


def test_split_saver_ignores_initial_home_and_saves_after_return(
    h5df_saver_module, saved_episodes, tmp_path
):
    saver = _split_saver(h5df_saver_module, tmp_path)
    saver.on_episode_start()

    for _ in range(3):
        saver.on_step(_observation(offset=0.0), _action())
    for _ in range(4):
        saver.on_step(_observation(offset=0.35), _action())
    for _ in range(2):
        saver.on_step(_observation(offset=0.0), _action())

    assert len(saved_episodes) == 1
    assert len(saved_episodes[0]["observations"]) == 6
    assert len(saved_episodes[0]["actions"]) == 6


def test_split_saver_writes_multiple_rollouts_in_one_runtime_episode(
    h5df_saver_module, saved_episodes, tmp_path
):
    saver = _split_saver(h5df_saver_module, tmp_path, stable_home_steps=1)
    saver.on_episode_start()

    for _ in range(2):
        saver.on_step(_observation(offset=0.35), _action())
    saver.on_step(_observation(offset=0.0), _action())
    for _ in range(3):
        saver.on_step(_observation(offset=0.35), _action())
    saver.on_step(_observation(offset=0.0), _action())

    assert [len(episode["observations"]) for episode in saved_episodes] == [3, 4]


def test_split_saver_does_not_save_if_robot_never_leaves_home(
    h5df_saver_module, saved_episodes, tmp_path
):
    saver = _split_saver(h5df_saver_module, tmp_path)
    saver.on_episode_start()

    for _ in range(5):
        saver.on_step(_observation(offset=0.0), _action())
    saver.on_episode_end()

    assert saved_episodes == []


def test_split_saver_flushes_active_rollout_on_runtime_episode_end(
    h5df_saver_module, saved_episodes, tmp_path
):
    saver = _split_saver(h5df_saver_module, tmp_path)
    saver.on_episode_start()

    for _ in range(4):
        saver.on_step(_observation(offset=0.35), _action())
    saver.on_episode_end()

    assert len(saved_episodes) == 1
    assert len(saved_episodes[0]["observations"]) == 4
