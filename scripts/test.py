import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

from openpi.models.tokenizer import PaligemmaTokenizer
import numpy as np

# config = _config.get_config("pi05_droid")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# # Create a trained policy.
# policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
# example = droid_policy.make_droid_example()
# result = policy.infer(example)

# # Delete the policy to free up memory.
# del policy

# print("Actions shape:", result["actions"].shape)

# tokenizer = PaligemmaTokenizer(max_len=200)
# prompt = "[bad action] Do the followings: 1. If the bottle cap is facing left, rotate the bottle 180 degrees. 2. Pick up the bottle. 3. Twist off the bottle cap. 4. Put the bottle into the box on the left. 5. Put the cap into the box on the right. If the bottle cap falls onto the table, pick it up. 6. Return to home position."
# state = np.random.rand(14).astype(np.float32)
# print(state)
# tokens, masks = tokenizer.tokenize(prompt, state)
# print(tokens)
# print(masks)
# decoded_tokens = tokenizer._tokenizer.decode(tokens.tolist())
# decoded_token = tokenizer._tokenizer.decode([235274])
# print(decoded_token, decoded_tokens)

if __name__ == '__main__':
    config = _config.get_config("pi05_aloha_pen_uncap")
    # 设置 num_workers=0 以确保数据顺序严格按顺序输出
    # config = dataclasses.replace(config, num_workers=0)
    test_dataloader = _data_loader.create_data_loader(config, shuffle=False)

    for batch in test_dataloader:
        pass