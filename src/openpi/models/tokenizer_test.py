import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_get_good_bad_action_label():
    assert _tokenizer.get_good_bad_action_label(None) == "normal"
    assert _tokenizer.get_good_bad_action_label("") == "normal"
    assert _tokenizer.get_good_bad_action_label('{"good_bad_action":"good action"}') == "good action"
    assert _tokenizer.get_good_bad_action_label({"good_bad_action": "bad action"}) == "bad action"
    assert _tokenizer.get_good_bad_action_label('{"good_bad_action":"unexpected"}') == "normal"


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
