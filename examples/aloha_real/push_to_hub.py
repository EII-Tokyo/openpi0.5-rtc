from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

repo_id = "lyl472324464/twist-many-20251015"

dataset = LeRobotDataset(repo_id)

dataset.push_to_hub()