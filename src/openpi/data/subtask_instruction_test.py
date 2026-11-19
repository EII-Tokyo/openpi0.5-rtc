"""Ensure stage prompts replace global goals only for the opted-in training config."""
import dataclasses

import numpy as np
import pytest

from openpi.data import transforms
from openpi.training.config import get_config

CONFIG = 'eii_hxz_exp_insert_3repo_cam4_fullft_h200x2_448_no_mem_5k_20261119'


def test_stage_prompt_replaces_global_without_mutation():
    sample = {'task': 'complete the cycle', 'prompt': 'old prompt',
              'subtask_instruction': 'Align bottle 2 with the pipe.'}
    out = transforms.PromptFromSubtaskInstruction()(sample)
    assert out['prompt'] == sample['subtask_instruction']
    assert 'task' not in out and 'subtask_instruction' not in out
    assert sample['task'] == 'complete the cycle'


@pytest.mark.parametrize('value', [None, '', '  ', 2, ['pick', 'align']])
def test_invalid_stage_never_falls_back(value):
    with pytest.raises(ValueError, match='subtask_instruction'):
        transforms.PromptFromSubtaskInstruction()({'task': 'global goal', 'subtask_instruction': value})


def test_config_repack_and_legacy_default():
    pipeline = get_config(CONFIG).data.transform_pipeline
    assert pipeline.language_source == 'subtask_instruction'
    sample = {'observation': {'images': {key: np.zeros((3, 4, 4), np.uint8) for key in pipeline.raw_image_keys},
                              'state': np.zeros(14)},
              'action': np.zeros((50,14)), 'task': 'global goal', 'subtask_instruction': 'Release bottle 2.'}
    repacked = pipeline.training_repack_transform()(sample)
    assert transforms.PromptFromSubtaskInstruction()(repacked)['prompt'] == 'Release bottle 2.'
    del sample['subtask_instruction']
    with pytest.raises((KeyError, ValueError)):
        pipeline.training_repack_transform()(sample)
    legacy = dataclasses.replace(pipeline, language_source='task')
    assert transforms.PromptFromLeRobotTask()(legacy.training_repack_transform()(sample))['prompt'] == 'global goal'


def test_norm_repack_does_not_require_language():
    pipeline = get_config(CONFIG).data.transform_pipeline
    sample = {'observation': {'images': {key: np.zeros((3,4,4),np.uint8) for key in pipeline.raw_image_keys},
                              'state': np.zeros(14)}, 'action':np.zeros((50,14)), 'task':'stats'}
    out = transforms.compose(pipeline.raw_state_action_transforms())(sample)
    assert out['actions'].shape == (50,14)

