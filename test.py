from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

repo_id = "lyl472324464/aloha_static_tape"

dataset = LeRobotDataset(repo_id)

dataset.push_to_hub()