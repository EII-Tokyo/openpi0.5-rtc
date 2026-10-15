import json
import numpy as np
import pytest

from openpi.data import tokenizer, transforms
from openpi.shared.prompt_metadata import normalize_prompt_metadata, prompt_metadata_suffix
from openpi.data.episode_metadata import attach_episode_metadata
from types import SimpleNamespace


def test_nullable_episode_metadata_is_legacy_compatible():
    result = attach_episode_metadata(
        {"episode_index": 0}, SimpleNamespace(episodes=[{"episode_index": 0, "operator": None, "rating": None}])
    )
    assert prompt_metadata_suffix(result["operator"], result["rating"], result["rating_is_valid"]) == ""
    result = attach_episode_metadata(
        {"episode_index": 0, "operator": "Jia Long"},
        SimpleNamespace(episodes=[{"episode_index": 0, "operator": None, "rating": None}]),
    )
    assert result["operator"] == "Jia Long"


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({}, ""),
        ({"operator": "", "rating": 0, "rating_is_valid": False}, ""),
        ({"operator": "Jia Long"}, ", Operator: Jia Long"),
        ({"rating": np.int64(5)}, ", Rating: 5"),
        ({"operator": "Xinzheng He", "rating": 4, "rating_is_valid": True}, ", Operator: Xinzheng He, Rating: 4"),
        ({"operator": "Zhaonan Ma", "rating": 4, "rating_is_valid": False}, ", Operator: Zhaonan Ma"),
    ],
)
def test_optional_metadata(metadata, expected):
    assert prompt_metadata_suffix(**metadata) == expected


@pytest.mark.parametrize("rating", [True, 6, -1, 2.5, "5"])
def test_invalid_rating(rating):
    with pytest.raises(ValueError):
        normalize_prompt_metadata(rating=rating)


def test_legacy_tokens_are_identical():
    tok = tokenizer.PaligemmaTokenizer(200)
    state = np.zeros(14)
    subtask = json.dumps({"good_bad_action": "good action"})
    expected = tok._tokenizer.encode(
        "Task: rinse bottle, State: " + " ".join(["128"] * 14) + ";\nAction: ", add_bos=True
    )
    tokens, mask = tok.tokenize("rinse bottle", state, subtask)
    assert tokens[mask].tolist() == expected
    sentinel_tokens, sentinel_mask = tok.tokenize(
        "rinse bottle", state, subtask, operator="", rating=0, rating_is_valid=False
    )
    np.testing.assert_array_equal(tokens, sentinel_tokens)
    np.testing.assert_array_equal(mask, sentinel_mask)
    for old_label in ["normal", "good action", "bad action"]:
        legacy, legacy_mask = tok.tokenize("rinse bottle", state, json.dumps({"good_bad_action": old_label}))
        np.testing.assert_array_equal(tokens, legacy)
        np.testing.assert_array_equal(mask, legacy_mask)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"operator": "Jia Long"},
        {"rating": 5},
        {"operator": "Xinzheng He", "rating": 4, "rating_is_valid": True},
        {"operator": "", "rating": 0, "rating_is_valid": False},
        {"operator": "Jia Long", "rating": 5, "camera_low_position": "front"},
        {"camera_low_position": "rear"},
        {"camera_low_position": ""},
    ],
)
def test_training_and_inference_pipeline_match(metadata):
    root = "/app/checkpoints/eii_with_rinse_cam4_fullft_from_april_original_38repo/6beba2e5-e00a-431f-b8a8-b95bf48ac168/22000/assets"
    pipe = transforms.AlohaTransformPipeline(
        include_low=True,
        use_good_bad_action_prompt=True,
        image_resolution=(224, 224),
        max_token_len=200,
        discrete_state_input=True,
        assets=transforms.AssetsConfig(root, "aloha"),
    )
    images = {key: np.zeros((3, 224, 224), np.uint8) for key in pipe.raw_image_keys}
    common = {
        "task": "twist off the bottle cap and rinse the bottle",
        "subtask": json.dumps({"good_bad_action": "good action"}),
        **metadata,
    }
    train = {
        **common,
        "observation.state": np.zeros(14),
        "action": np.zeros((50, 14)),
        **{"observation.images." + key: value for key, value in images.items()},
    }
    infer = {**common, "state": np.zeros(14), "images": images}
    # Subtask is not required by new training data or inference clients.
    train.pop("subtask")
    infer.pop("subtask")
    a = transforms.CompositeTransform(pipe.training_input_transforms())(train)
    b = transforms.CompositeTransform(pipe.policy_input_transforms())(infer)
    np.testing.assert_array_equal(a["tokenized_prompt"], b["tokenized_prompt"])
    np.testing.assert_array_equal(a["tokenized_prompt_mask"], b["tokenized_prompt_mask"])
    assert not any(k in a or k in b for k in ["operator", "rating", "rating_is_valid", "camera_low_position"])
    tok = tokenizer.PaligemmaTokenizer(200)
    text = tok._tokenizer.decode(a["tokenized_prompt"][a["tokenized_prompt_mask"]].tolist())
    assert text.rstrip().endswith("Action:")
    fields = normalize_prompt_metadata(**metadata)
    if "operator" in fields:
        assert "Operator: " + fields["operator"] in text
    else:
        assert "Operator:" not in text
    if "rating" in fields:
        assert "Rating: " + str(fields["rating"]) in text
    else:
        assert "Rating:" not in text
