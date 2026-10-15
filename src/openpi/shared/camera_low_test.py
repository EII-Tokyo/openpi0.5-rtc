"""Ensure repo info.json placement and inference conditioning tokenize identically."""

import json
from types import SimpleNamespace
import numpy as np
import pytest

from openpi.data import tokenizer, transforms
from openpi.data.episode_metadata import attach_episode_metadata
from openpi.shared.prompt_metadata import camera_low_prompt, normalize_prompt_metadata


@pytest.mark.parametrize("position,text", [("front", "near side of the workbench"), ("rear", "far side of the workbench")])
def test_repo_info_matches_inference(position, text, tmp_path):
    """Read only info.json placement; the training task contains no camera suffix."""
    tok = tokenizer.PaligemmaTokenizer(200)
    state = np.zeros(14)
    exported = f"rinse bottle, Camera low: {text}"
    common = dict(operator="Jia Long", rating=5)
    expected, expected_mask = tok.tokenize(exported, state, **common)
    info_path = tmp_path / "meta" / "info.json"
    info_path.parent.mkdir()
    info_path.write_text(json.dumps({"camera_low_position": position}))
    metadata = SimpleNamespace(info=json.loads(info_path.read_text()),
                               episodes=[{"episode_index": 0, **common}])
    train_sample = attach_episode_metadata(dict(episode_index=0, prompt="rinse bottle", state=state), metadata)
    train = transforms.TokenizePrompt(tok, discrete_state_input=True)(train_sample)
    data = transforms.TokenizePrompt(tok, discrete_state_input=True)(dict(prompt="rinse bottle", state=state, camera_low_position=position, **common))
    np.testing.assert_array_equal(train["tokenized_prompt"], data["tokenized_prompt"])
    np.testing.assert_array_equal(expected, data["tokenized_prompt"])
    np.testing.assert_array_equal(expected_mask, data["tokenized_prompt_mask"])
    assert "camera_low_position" not in data
    decoded = tok._tokenizer.decode(expected[expected_mask].tolist())
    assert decoded.count("Camera low:") == 1
    assert f"Camera low: {text}, Operator: Jia Long, Rating: 5" in decoded


def test_legacy_preserved_and_explicit_override():
    """Missing placement never guesses; supplied placement replaces exported suffix."""
    assert camera_low_prompt("rinse bottle") == "rinse bottle"
    task = "rinse bottle, Camera low: near side of the workbench"
    assert camera_low_prompt(task) == task
    assert camera_low_prompt(task, "rear") == "rinse bottle, Camera low: far side of the workbench"
    assert camera_low_prompt("rinse bottle, Camera low: rear", "front") == task
    # The reader supplies an empty string for absent/null info.json placement.
    assert camera_low_prompt(task, "") == "rinse bottle"
    assert normalize_prompt_metadata(camera_low_position="front") == {"camera_low_position": "front"}


@pytest.mark.parametrize("value", [True, 1, "left", []])
def test_invalid_position(value):
    """Reject unknown positions rather than silently conditioning the wrong view."""
    with pytest.raises(ValueError):
        normalize_prompt_metadata(camera_low_position=value)
    with pytest.raises(ValueError):
        camera_low_prompt("rinse bottle", value)
