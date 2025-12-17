from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "lyl472324464/remove-label-20251021"

dataset = LeRobotDataset(repo_id)

dataset.push_to_hub()