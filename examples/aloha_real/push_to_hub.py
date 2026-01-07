from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "lyl472324464/twist_two_20251118"

dataset = LeRobotDataset(repo_id)

dataset.push_to_hub()