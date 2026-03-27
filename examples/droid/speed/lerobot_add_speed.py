from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from huggingface_hub import HfApi
import json
from datetime import datetime, timezone, timedelta

repos = ["michios/droid_xxjd",
         "michios/droid_xxjd_2",
         "michios/droid_xxjd_3",
         "michios/droid_xxjd_4",
         "michios/droid_xxjd_5",
         "michios/droid_xxjd_6",
         "michios/droid_xxjd_7",
         "michios/droid_xxjd_8_2" 
        ]
annotations_path = "examples/droid/speed/annotations_tmp.jsonl"
backup_branch = "pre_speed_processing_backup"

initial_upload_dates = []
current_repo_idx = 0
current_parquet_idx = 0

api = HfApi()
for repo in repos:
    # get init upload dates
    info = api.dataset_info(repo)
    initial_upload_dates.append({"repo": repo, "created": info.created_at})
    refs = api.list_repo_refs(repo, repo_type="dataset")
    branch_names = [b.name for b in refs.branches]
    if backup_branch not in branch_names:
        api.create_branch(repo, branch=backup_branch, repo_type="dataset")
        print(f"Created branch {backup_branch} in dataset {repo}")
    

jst = timezone(timedelta(hours=9))
ds = LeRobotDataset(repos[current_repo_idx], download_videos=False)
files = api.list_repo_tree(repos[current_repo_idx], repo_type="dataset", path_in_repo="data")
print(files)

# with open(annotations_path, "r") as f:
#     # check if date lines up
#     for line in f:
#         data = json.loads(line)
#         dt = data["uuid"].split("+")[-1]
#         dt = datetime.strptime(dt, "%Y-%m-%d-%Hh-%Mm-%Ss").replace(tzinfo=jst)
#         if (dt > initial_upload_dates[current_repo_idx]["created"]):
#             current_repo_idx += 1
#             print(current_repo_idx, dt)
#             if (current_repo_idx > len(initial_upload_dates)):
#                 break
#             # load current dataset
#             ds = LeRobotDataset(repos[current_repo_idx], download_videos=False)
#         # check if task lines up
#         assert(data["prompts"]["default"] ==