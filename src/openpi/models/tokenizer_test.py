import numpy as np
from unittest import mock

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks, *_ = tokenizer.tokenize("Hello, world!", train_action=False)

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_paligemma_train_action_tokenize_shapes():
    prompt = "Hello, world!"
    state = np.random.rand(14).astype(np.float32)
    actions = np.random.rand(3, 14).astype(np.float32)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=32, subtask_max_len=64)
    tokenizer._get_fast_tokenizer = lambda: (lambda batch: np.asarray([[1, 2, 3]], dtype=np.int32))

    tokens, token_masks, subtask_tokens, subtask_masks, loss_masks, fast_masks = tokenizer.tokenize(
        prompt,
        state=state,
        subtask={"subtask": "Unscrew cap"},
        actions=actions,
        train_action=True,
    )

    assert tokens.shape == (32,)
    assert token_masks.shape == (32,)
    assert subtask_tokens.shape == (64,)
    assert subtask_masks.shape == (64,)
    assert loss_masks.shape == (64,)
    assert fast_masks.shape == (64,)
    assert int(fast_masks.sum()) == 3


def test_train_fast_action_tokens_switch_changes_subtask_length():
    state = np.zeros((14,), dtype=np.float32)
    actions = np.zeros((2, 14), dtype=np.float32)
    subtask = {"subtask": "Pick up with left hand"}

    tokenizer_with_fast = _tokenizer.PaligemmaTokenizer(max_len=64, subtask_max_len=128)
    tokenizer_with_fast._get_fast_tokenizer = lambda: (lambda batch: np.asarray([[1, 2, 3]], dtype=np.int32))

    _, _, _, subtask_mask_with_fast, _, fast_mask_with_fast = tokenizer_with_fast.tokenize(
        "Twist the bottle.",
        state=state,
        subtask=subtask,
        actions=actions,
        train_action=True,
    )

    tokenizer_without_fast = _tokenizer.PaligemmaTokenizer(
        max_len=64,
        subtask_max_len=128,
        train_fast_action_tokens=False,
    )
    tokenizer_without_fast._get_fast_tokenizer = lambda: (_ for _ in ()).throw(
        AssertionError("FAST tokenizer should not be loaded when train_fast_action_tokens=False")
    )

    _, _, _, subtask_mask_without_fast, _, fast_mask_without_fast = tokenizer_without_fast.tokenize(
        "Twist the bottle.",
        state=state,
        subtask=subtask,
        actions=actions,
        train_action=True,
    )

    assert int(fast_mask_with_fast.sum()) == 3
    assert int(fast_mask_without_fast.sum()) == 0
    assert int(subtask_mask_with_fast.sum()) > int(subtask_mask_without_fast.sum())


def test_bottle_description_dropout_only_drops_target_text():
    state = np.zeros((14,), dtype=np.float32)
    subtask = {
        "bottle_description": "Blue sports bottle",
        "bottle_state": "Bottle in left hand and capped",
        "subtask": "Unscrew cap",
    }
    tokenizer = _tokenizer.PaligemmaTokenizer(
        max_len=64,
        subtask_max_len=64,
        train_fast_action_tokens=False,
        bottle_description_dropout_prob=1.0,
    )

    with mock.patch("openpi.models.tokenizer.random.random", return_value=0.0):
        _, _, subtask_tokens, subtask_mask, _, _ = tokenizer.tokenize(
            "Twist the bottle.",
            state=state,
            subtask=subtask,
            actions=None,
            train_action=True,
        )

    text = tokenizer.decode(subtask_tokens[subtask_mask].tolist())
    assert "Target:" not in text
    assert "Bottle State:" in text
    assert "Subtask:" in text


def test_bottle_description_dropout_zero_keeps_target_text():
    state = np.zeros((14,), dtype=np.float32)
    subtask = {
        "bottle_description": "Blue sports bottle",
        "bottle_state": "Bottle in left hand and capped",
        "subtask": "Unscrew cap",
    }
    tokenizer = _tokenizer.PaligemmaTokenizer(
        max_len=64,
        subtask_max_len=64,
        train_fast_action_tokens=False,
        bottle_description_dropout_prob=0.0,
    )

    with mock.patch("openpi.models.tokenizer.random.random", return_value=0.0):
        _, _, subtask_tokens, subtask_mask, _, _ = tokenizer.tokenize(
            "Twist the bottle.",
            state=state,
            subtask=subtask,
            actions=None,
            train_action=True,
        )

    text = tokenizer.decode(subtask_tokens[subtask_mask].tolist())
    assert "Target:" in text
