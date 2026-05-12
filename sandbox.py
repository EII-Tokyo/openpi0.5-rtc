#!/usr/bin/env python3
import pandas as pd
import json
import sys
from pathlib import Path

def extract_episodes(csv_path: str):
    """
    Extract episode indices grouped by dataset from a CSV file.
    Handles the format: row,A,B where A=dataset_id, B=episode_index
    """
    df = pd.read_csv(csv_path)
    
    # Clean: drop rows where episode index is empty/NaN
    df = df.dropna(subset=["B"])
    df["B"] = df["B"].astype(int)  # Ensure integer type
    
    # Group episodes by dataset (column A)
    episodes_by_dataset = {}
    for dataset_id, group in df.groupby("A"):
        eps = sorted(group["B"].unique().tolist())  # Remove duplicates, sort
        episodes_by_dataset[dataset_id] = eps
        print(f"✅ {dataset_id}: {len(eps)} episodes")
        print(f"   Range: {min(eps)} to {max(eps)}")
        print(f"   First 10: {eps[:10]}")
        print(f"   Last 10: {eps[-10:]}")
        print()
    
    return episodes_by_dataset

def format_for_cli(episodes: list[int]) -> str:
    """Format episode list for --operation.splits parameter"""
    # Create the splits JSON: {"selected": [0,1,2,...]}
    splits_dict = {"selected": episodes}
    return json.dumps(splits_dict)

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_episodes.py <csv_file> [output_file]")
        print("Example: python extract_episodes.py episodes.csv > splits.sh")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Extract
    episodes_by_dataset = extract_episodes(csv_path)
    print(episodes_by_dataset)
    
    # # Generate shell commands
    # print("\n" + "="*60)
    # print("📋 COPY THESE COMMANDS FOR lerobot-edit-dataset")
    # print("="*60 + "\n")
    
    # for dataset_id, eps in episodes_by_dataset.items():
    #     # Sanitize dataset name for variable names
    #     var_name = dataset_id.replace("/", "_").replace("-", "_").replace(".", "_")
    #     splits_json = format_for_cli(eps)
        
    #     print(f"# === {dataset_id} ===")
    #     print(f"{var_name}_SPLITS='{splits_json}'")
    #     print()
    #     print(f"lerobot-edit-dataset \\")
    #     print(f"    --repo_id {dataset_id} \\")
    #     print(f"    --new_repo_id {dataset_id}_subset \\")
    #     print(f"    --operation.type split \\")
    #     print(f"    --operation.splits \"${{{var_name}_SPLITS}}\" \\")
    #     print(f"    --push_to_hub false")
    #     print()
    
    # # Optional: Save to file
    # if output_path:
    #     with open(output_path, "w") as f:
    #         for dataset_id, eps in episodes_by_dataset.items():
    #             var_name = dataset_id.replace("/", "_").replace("-", "_").replace(".", "_")
    #             splits_json = format_for_cli(eps)
    #             f.write(f"{var_name}_SPLITS='{splits_json}'\n")
    #     print(f"💾 Saved variable definitions to {output_path}")

if __name__ == "__main__":
    main()