import lerobot.datasets.lerobot_dataset as lerobot_dataset

repo_ids = ["lyl472324464/2025.12.23_twist_one"]
for repo_id in repo_ids:
    dataset = lerobot_dataset.LeRobotDataset(repo_id)

    dataset.push_to_hub()