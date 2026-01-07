"""
Script to fix dataset issues:
1. Create tasks.jsonl from tasks.parquet (or create a default one)
2. Create version tag for the dataset
"""

import json
import jsonlines
from pathlib import Path
from huggingface_hub import HfApi

# Dataset paths
repo_id = "lyl472324464/twist_two_202511"
dataset_root = Path.home() / ".cache" / "huggingface" / "lerobot" / "lyl472324464" / "twist_two_202511"
meta_dir = dataset_root / "meta"
tasks_parquet = meta_dir / "tasks.parquet"
tasks_jsonl = meta_dir / "tasks.jsonl"
info_json = meta_dir / "info.json"

def create_tasks_jsonl():
    """Create tasks.jsonl file from tasks.parquet or create a default one."""
    if tasks_jsonl.exists():
        print(f"✓ tasks.jsonl already exists at {tasks_jsonl}")
        return
    
    # Try to read from parquet if it exists
    if tasks_parquet.exists():
        try:
            # Try using polars (might be available)
            import polars as pl
            df = pl.read_parquet(tasks_parquet)
            print(f"✓ Read tasks from parquet: {df}")
            
            # Convert to jsonl format
            tasks_data = df.to_dicts()
            with jsonlines.open(tasks_jsonl, mode='w') as writer:
                for task_data in tasks_data:
                    task_index = task_data.get("task_index", 0)
                    # Extract task description - could be in different fields
                    task_desc = (
                        task_data.get("task") 
                        or task_data.get("__index_level_0__")
                        or task_data.get("task_description")
                        or f"task_{task_index}"
                    )
                    # If task_desc is a dict, try to extract the string from it
                    if isinstance(task_desc, dict):
                        task_desc = task_desc.get("__index_level_0__") or task_desc.get("task") or str(task_desc)
                    
                    writer.write({
                        "task_index": task_index,
                        "task": task_desc
                    })
            print(f"✓ Created tasks.jsonl from parquet")
            return
        except ImportError:
            print("⚠ polars not available, trying alternative method...")
        except Exception as e:
            print(f"⚠ Error reading parquet with polars: {e}, trying alternative method...")
    
    # Fallback: Create a default tasks.jsonl from info.json
    if info_json.exists():
        with open(info_json, 'r') as f:
            info = json.load(f)
        
        total_tasks = info.get("total_tasks", 1)
        
        # Create default task entries
        with jsonlines.open(tasks_jsonl, mode='w') as writer:
            for task_index in range(total_tasks):
                # Default task description - you may want to customize this
                task_dict = {
                    "task_index": task_index,
                    "task": f"twist_two_task_{task_index}" if total_tasks > 1 else "twist_two"
                }
                writer.write(task_dict)
        print(f"✓ Created default tasks.jsonl with {total_tasks} task(s)")
    else:
        # Last resort: create a single default task
        with jsonlines.open(tasks_jsonl, mode='w') as writer:
            writer.write({
                "task_index": 0,
                "task": "twist_two"
            })
        print(f"✓ Created default tasks.jsonl with single task")

def create_version_tag():
    """Create version tag for the dataset."""
    if not info_json.exists():
        print("⚠ info.json not found, cannot determine version")
        return
    
    with open(info_json, 'r') as f:
        info = json.load(f)
    
    codebase_version = info.get("codebase_version", "v3.0")
    tag_name = codebase_version
    
    print(f"Creating tag '{tag_name}' for dataset {repo_id}...")
    
    try:
        hub_api = HfApi()
        hub_api.create_tag(repo_id, tag=tag_name, repo_type="dataset")
        print(f"✓ Successfully created tag '{tag_name}'")
    except Exception as e:
        print(f"⚠ Error creating tag: {e}")
        print(f"  You may need to run this manually:")
        print(f"  from huggingface_hub import HfApi")
        print(f"  hub_api = HfApi()")
        print(f"  hub_api.create_tag('{repo_id}', tag='{tag_name}', repo_type='dataset')")

if __name__ == "__main__":
    print("Fixing dataset issues...")
    print(f"Dataset root: {dataset_root}")
    print()
    
    # Create tasks.jsonl
    print("1. Creating tasks.jsonl...")
    create_tasks_jsonl()
    print()
    
    # Create version tag
    print("2. Creating version tag...")
    create_version_tag()
    print()
    
    print("✓ Done! You can now try running push_to_hub.py again.")

