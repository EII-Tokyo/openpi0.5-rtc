import numpy as np
import pytest

from openpi.data.episode_split import training_episode_mask


def test_whole_episode_holdout():
    episodes = np.repeat([0, 1, 2], [3, 4, 2])
    mask = training_episode_mask('repo', episodes, {'repo': [0, 2]})
    np.testing.assert_array_equal(episodes[mask], [0, 0, 0, 2, 2])
    assert not np.any(episodes[mask] == 1)
    assert training_episode_mask('repo', episodes, None).all()


@pytest.mark.parametrize('selection', [{}, {'repo': []}, {'repo': [0, 0]}, {'repo': [3]}])
def test_invalid_selection_fails_closed(selection):
    with pytest.raises(ValueError):
        training_episode_mask('repo', [0, 1, 2], selection)


def test_config_matches_fixed_holdout():
    import json
    from pathlib import Path
    from openpi.training.config import get_config

    name = 'eii_hxz_exp_insert_3repo_cam4_fullft_h200x2_448_no_mem_5k_20261119'
    manifest = Path(__file__).parents[1] / 'training' / 'manifests' / name / 'episode_split.json'
    split = json.loads(manifest.read_text())
    config = get_config(name)
    assert sum(len(d['test_episode_indices']) for d in split['datasets']) == 5
    assert sum(len(d['train_episode_indices']) for d in split['datasets']) == 145
    for d in split['datasets']:
        assert config.data.train_episode_indices[d['repo_id']] == d['train_episode_indices']
        assert set(d['train_episode_indices']).isdisjoint(d['test_episode_indices'])
        assert sorted(d['train_episode_indices'] + d['test_episode_indices']) == list(range(50))
