"""Camera placement is dataset metadata; task files must remain camera-independent."""
from types import SimpleNamespace

import pytest

from openpi.data.episode_metadata import attach_episode_metadata


@pytest.mark.parametrize('position,text', [('front', 'near side of the workbench'), ('rear', 'far side of the workbench')])
def test_project_metadata_conditions_memory_only(position, text):
    """Both readers join dataset info after resolving the local EP without mutating the input."""
    metadata = SimpleNamespace(info={'camera_low_position': position},
                               episodes=[{'episode_index': 0, 'operator': 'Jia Long', 'rating': None}])
    original = {'episode_index': 0, 'task': 'rinse bottle'}
    result = attach_episode_metadata(original, metadata)
    assert original == {'episode_index': 0, 'task': 'rinse bottle'}
    assert result['task'] == 'rinse bottle'
    assert result['camera_low_position'] == position
    assert attach_episode_metadata(result, metadata)['task'] == result['task']
    assert metadata.info == {'camera_low_position': position}


def test_unset_does_not_guess():
    """Legacy datasets are unchanged when camera placement is unknown."""
    metadata = SimpleNamespace(info={}, episodes=[{'episode_index': 0}])
    result = attach_episode_metadata({'episode_index': 0, 'task': 'rinse bottle'}, metadata)
    assert result['task'] == 'rinse bottle'
    assert result['camera_low_position'] == ''


def test_invalid_metadata_rejected():
    """An unknown metadata value cannot condition a trajectory silently."""
    metadata = SimpleNamespace(info={'camera_low_position': 'left'}, episodes=[{'episode_index': 0}])
    with pytest.raises(ValueError):
        attach_episode_metadata({'episode_index': 0, 'task': 'rinse bottle'}, metadata)
