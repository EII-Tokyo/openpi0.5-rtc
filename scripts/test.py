from openpi.training import config as _config

from openpi.policies import policy_config, aloha_policy, droid_policy
from openpi.shared import download

config = _config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = droid_policy.make_droid_example()
action_chunk = policy.infer(example)["actions"]
print(action_chunk.shape)

