import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


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
