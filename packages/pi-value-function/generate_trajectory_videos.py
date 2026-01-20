#!/usr/bin/env python3
"""Generate trajectory visualization videos from value function training data.

This script:
1. Loads trajectories directly from LeRobot datasets (no model/tokenizer needed)
2. For each trajectory, generates a video showing:
   - Left side: The observation image at each timestep
   - Right side: Line chart with value curve and current timestep marker
3. Saves videos as MP4 files

Usage:
    python generate_trajectory_videos.py
"""

import sys
from pathlib import Path
import json

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from lerobot.datasets import lerobot_dataset

# np random seed for reproducibility
np.random.seed(32)


def compute_value_target(
    timestep: int,
    episode_length: int,
    is_success: bool,
    c_fail: float,
    raw_value_min: float,
    raw_value_max: float,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> float:
    """Compute value target and normalize to [value_min, value_max].

    Raw value formula:
    - Success: V(s_t) = -(T - t)  where T is episode length, t is timestep
    - Failure: V(s_t) = -(T - t) - C_fail

    Then normalize raw value from [raw_value_min, raw_value_max] to [value_min, value_max]
    """
    # Compute raw value
    if is_success:
        raw_value = -(episode_length - timestep)
    else:
        raw_value = -(episode_length - timestep) - c_fail

    # Normalize to [value_min, value_max]
    normalized = (raw_value - raw_value_min) / (raw_value_max - raw_value_min)
    normalized = normalized * (value_max - value_min) + value_min

    # Clip to ensure values stay within bounds (handles outliers beyond percentile-based range)
    normalized = np.clip(normalized, value_min, value_max)

    return float(normalized)


def load_failure_costs(json_path: Path) -> dict[str, float]:
    """Load failure costs from JSON config.

    Returns:
        Dictionary mapping prompt -> c_fail cost
    """
    if not json_path.exists():
        print(f"Warning: Failure cost JSON not found at {json_path}")
        return {}

    with open(json_path) as f:
        data = json.load(f)

    costs = {}
    for entry in data:
        prompt = entry["prompt"].lower().strip()
        costs[prompt] = entry["c_fail"]

    return costs


def get_episode_info(dataset):
    """Extract episode boundaries and metadata from LeRobot dataset.

    Returns:
        List of dicts with: episode_id, start_idx, end_idx, length, task
    """
    episodes = []

    for ep_id in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_id]
        start_idx = ep["dataset_from_index"]
        end_idx = ep["dataset_to_index"]
        length = end_idx - start_idx

        # Get task from metadata
        # First get task_index from the data
        sample = dataset.hf_dataset[start_idx]
        task_idx = sample.get("task_index", None)

        if task_idx is not None and dataset.meta.tasks is not None:
            # Convert to int (HF datasets may return numpy/tensor types)
            task_idx = int(task_idx)
            # Look up task string from tasks metadata
            task = dataset.meta.tasks.iloc[task_idx].name
        else:
            task = ""

        episodes.append({
            "episode_id": ep_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "length": length,
            "task": task,
        })

    return episodes


def collect_full_episodes(
    repo_id: str,
    is_success: bool,
    num_episodes: int,
    failure_costs: dict[str, float],
    default_c_fail: float,
    task_normalization: dict[str, tuple[float, float]],
    global_raw_value_min: float,
    global_raw_value_max: float,
    filter_prompt: str | None = None,
):
    """Collect full episode trajectories from a LeRobot dataset.

    Args:
        filter_prompt: If provided, only collect episodes with this exact prompt (case-insensitive)
        num_episodes: Max number of episodes to collect (if filter_prompt, collects all matching up to this limit)

    Returns:
        List of episode dictionaries containing:
        - images: List of RGB images [H, W, 3]
        - values: List of normalized values
        - timesteps: List of timestep indices
        - prompt: Task prompt string
        - success: Boolean success flag
        - episode_id: Episode identifier
    """
    print(f"\nLoading dataset: {repo_id}")
    dataset = lerobot_dataset.LeRobotDataset(repo_id)

    episode_list = get_episode_info(dataset)
    print(f"Found {len(episode_list)} total episodes")

    # Filter by prompt if specified
    if filter_prompt:
        filter_prompt_norm = filter_prompt.lower().strip()
        filtered_list = [
            ep for ep in episode_list
            if ep['task'].lower().strip() == filter_prompt_norm
        ]
        print(f"  Filtered to {len(filtered_list)} episodes matching prompt: '{filter_prompt}'")
        episode_list = filtered_list

        if not episode_list:
            print(f"  Warning: No episodes found with prompt '{filter_prompt}'")
            return []

        # If filtering by prompt, collect all episodes (up to num_episodes)
        episodes_to_collect = min(num_episodes, len(episode_list))
        selected_indices = range(episodes_to_collect)
    else:
        # Random sampling
        episodes_to_collect = min(num_episodes, len(episode_list))
        selected_indices = [np.random.randint(0, len(episode_list)) for _ in range(episodes_to_collect)]

    episodes = []
    for idx in selected_indices:
        ep_info = episode_list[idx]
        print(f"  Processing episode {ep_info['episode_id']} "
              f"(length={ep_info['length']}, task='{ep_info['task']}')")

        images = []
        values = []
        timesteps = []

        for t in range(ep_info['length']):
            global_idx = ep_info['start_idx'] + t
            sample = dataset.hf_dataset[global_idx]

            # Get the base camera image (exterior_image_1_left)
            image = np.array(sample["exterior_image_1_left"])
            if image.dtype in (np.float32, np.float64):
                image = (image * 255).astype(np.uint8)
            if image.shape[0] == 3:  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
            images.append(image)

            # Compute value for this timestep
            prompt = ep_info['task'].lower().strip()
            c_fail = 0.0 if is_success else failure_costs.get(prompt, default_c_fail)

            # Get task-specific normalization range
            task_key = prompt if prompt else None
            if task_key in task_normalization:
                raw_value_min, raw_value_max = task_normalization[task_key]
            else:
                # Fallback to global normalization for unseen tasks
                raw_value_min, raw_value_max = global_raw_value_min, global_raw_value_max

            value = compute_value_target(
                timestep=t,
                episode_length=ep_info['length'],
                is_success=is_success,
                c_fail=c_fail,
                raw_value_min=raw_value_min,
                raw_value_max=raw_value_max,
                value_min=-1.0,
                value_max=0.0,
            )
            values.append(value)
            timesteps.append(t)

        episodes.append({
            'images': images,
            'values': values,
            'timesteps': timesteps,
            'prompt': ep_info['task'] or "No prompt",
            'success': is_success,
            'episode_id': ep_info['episode_id'],
            'length': ep_info['length'],
        })

    return episodes


def create_trajectory_video(episode, output_path, fps=10):
    """Create a video visualization for a single trajectory.

    Args:
        episode: Episode dictionary from collect_full_episodes
        output_path: Output video file path
        fps: Frames per second for the video
    """
    images = episode['images']
    values = episode['values']
    timesteps = episode['timesteps']
    prompt = episode['prompt']
    success = episode['success']

    # Setup figure with two subplots side by side
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)

    # Left subplot: Image
    ax_img = fig.add_subplot(gs[0])
    ax_img.axis('off')

    # Right subplot: Value chart
    ax_chart = fig.add_subplot(gs[1])

    # Initialize image display
    img_display = ax_img.imshow(images[0])

    # Setup value chart
    ax_chart.set_xlim(-0.5, len(timesteps) - 0.5)
    ax_chart.set_ylim(-1.05, 0.05)
    ax_chart.set_xlabel('Timestep', fontsize=12)
    ax_chart.set_ylabel('Normalized Value', fontsize=12)
    ax_chart.grid(True, alpha=0.3)

    # Plot the full value curve
    line, = ax_chart.plot(timesteps, values, 'b-', linewidth=2, alpha=0.7, label='Value trajectory')

    # Current position marker (vertical line + point)
    vline = ax_chart.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    point, = ax_chart.plot([0], [values[0]], 'ro', markersize=10, label='Current timestep')

    ax_chart.legend(loc='lower right')

    # Title with episode info
    status = "SUCCESS" if success else "FAILURE"
    title_color = 'green' if success else 'red'
    fig.suptitle(
        f"Episode {episode['episode_id']} - {status}\n"
        f"Prompt: {prompt}\n"
        f"Length: {len(timesteps)} timesteps | Value range: [{min(values):.3f}, {max(values):.3f}]",
        fontsize=14,
        fontweight='bold',
        color=title_color
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    def update_frame(frame_idx):
        """Update function for animation."""
        # Update image
        img_display.set_array(images[frame_idx])

        # Update current position marker
        vline.set_xdata([timesteps[frame_idx]])
        point.set_data([timesteps[frame_idx]], [values[frame_idx]])

        # Update title with current timestep
        ax_img.set_title(f"Timestep {timesteps[frame_idx]}/{len(timesteps)-1}", fontsize=12)

        return img_display, vline, point

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(images),
        interval=1000/fps,
        blit=True,
        repeat=True
    )

    # Save as MP4
    print(f"Saving video to {output_path}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Pi Value Function'), bitrate=3000)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Video saved successfully!")


def main():
    """Main function to generate trajectory videos."""
    print("=" * 80)
    print("Trajectory Video Generator")
    print("=" * 80)

    # Load failure costs
    script_dir = Path(__file__).parent
    failure_cost_json = script_dir / "configs" / "failure_costs.json"
    default_c_fail = 100.0

    print(f"\nLoading failure costs from: {failure_cost_json}")
    failure_costs = load_failure_costs(failure_cost_json)
    print(f"Loaded {len(failure_costs)} prompt costs")

    # Compute normalization parameters (matching data_loader.py logic)
    # We'll use the same repos to compute these stats
    success_repo_ids = ["michios/droid_xxjd_7"]
    failure_repo_ids = ["michios/droid_xxjd_fail_1"]

    print("\nComputing per-task normalization statistics...")

    # Group episodes by task
    task_episodes = defaultdict(list)
    all_lengths = []

    # Get episodes from all datasets
    for repo_id in success_repo_ids + failure_repo_ids:
        print(f"  Scanning {repo_id}...")
        ds = lerobot_dataset.LeRobotDataset(repo_id)
        episodes = get_episode_info(ds)
        is_success = repo_id in success_repo_ids

        for ep in episodes:
            task_key = ep['task'].lower().strip() if ep['task'] else None
            task_episodes[task_key].append({
                'length': ep['length'],
                'success': is_success,
            })
            all_lengths.append(ep['length'])

    # Compute per-task normalization ranges
    task_normalization = {}

    print("\n=== Per-Task Normalization Statistics ===")
    for task_key in sorted(task_episodes.keys(), key=lambda x: (x is None, x)):
        episodes = task_episodes[task_key]
        lengths = [ep['length'] for ep in episodes]

        # Get failure cost for this task
        c_fail = failure_costs.get(task_key, default_c_fail) if task_key else default_c_fail

        # Compute 75th percentile episode length for this task
        if len(lengths) >= 4:
            typical_length = int(np.percentile(lengths, 75))
        else:
            typical_length = int(np.mean(lengths)) if lengths else 0

        # Compute task-specific raw value range
        raw_value_max = 0.0
        raw_value_min = -(typical_length + c_fail)

        task_normalization[task_key] = (raw_value_min, raw_value_max)

        # Log task statistics
        task_name = task_key if task_key else "<no task>"
        num_success = sum(1 for e in episodes if e['success'])
        num_failure = len(episodes) - num_success
        print(f"  Task '{task_name}':")
        print(f"    Episodes: {len(episodes)} ({num_success} success, {num_failure} failure)")
        print(f"    Length (75th percentile): {typical_length}")
        print(f"    C_fail: {c_fail:.1f}")
        print(f"    Normalization range: [{raw_value_min:.1f}, {raw_value_max:.1f}]")

    # Compute global fallback
    global_typical_length = int(np.percentile(all_lengths, 75))
    global_c_fail = np.mean(list(failure_costs.values())) if failure_costs else default_c_fail
    global_raw_value_min = -(global_typical_length + global_c_fail)
    global_raw_value_max = 0.0

    print(f"\n  Global fallback range: [{global_raw_value_min:.1f}, {global_raw_value_max:.1f}]")
    print("=" * 50)

    # Configure which prompt to visualize
    # Set to None to randomly sample episodes, or specify a prompt to see all episodes for that task
    target_prompt = "Put the battery bank in the orange box"
    max_episodes_per_type = 5  # Max episodes to collect per success/failure

    # Collect episodes
    print("\n" + "=" * 80)
    print("Collecting episodes...")
    if target_prompt:
        print(f"Filtering for prompt: '{target_prompt}'")
    print("=" * 80)

    success_episodes = collect_full_episodes(
        repo_id=success_repo_ids[0],
        is_success=True,
        num_episodes=max_episodes_per_type,
        failure_costs=failure_costs,
        default_c_fail=default_c_fail,
        task_normalization=task_normalization,
        global_raw_value_min=global_raw_value_min,
        global_raw_value_max=global_raw_value_max,
        filter_prompt=target_prompt,
    )

    failure_episodes = collect_full_episodes(
        repo_id=failure_repo_ids[0],
        is_success=False,
        num_episodes=max_episodes_per_type,
        failure_costs=failure_costs,
        default_c_fail=default_c_fail,
        task_normalization=task_normalization,
        global_raw_value_min=global_raw_value_min,
        global_raw_value_max=global_raw_value_max,
        filter_prompt=target_prompt,
    )

    all_episodes = success_episodes + failure_episodes

    if not all_episodes:
        print("\n" + "=" * 80)
        print("No episodes found matching criteria!")
        print("=" * 80)
        return

    # Generate videos
    print("\n" + "=" * 80)
    print(f"Generating {len(all_episodes)} videos...")
    print("=" * 80)

    output_dir = Path(__file__).parent / "trajectory_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    for i, episode in enumerate(all_episodes):
        # Create descriptive filename
        status = "success" if episode['success'] else "failure"
        ep_id = episode['episode_id']
        output_path = output_dir / f"trajectory_ep{ep_id}_{status}.mp4"

        print(f"\n--- Video {i+1}/{len(all_episodes)} ---")
        print(f"Episode ID: {episode['episode_id']}")
        print(f"Status: {'SUCCESS' if episode['success'] else 'FAILURE'}")
        print(f"Prompt: {episode['prompt']}")
        print(f"Length: {episode['length']} timesteps")
        print(f"Value range: [{min(episode['values']):.3f}, {max(episode['values']):.3f}]")

        create_trajectory_video(episode, output_path, fps=10)
        print(f"Saved to: {output_path}")
        output_files.append(output_path)

    print("\n" + "=" * 80)
    print("All videos generated successfully!")
    print("=" * 80)
    print(f"\nGenerated {len(output_files)} videos:")
    for filepath in output_files:
        print(f"  - {filepath.name}")


if __name__ == "__main__":
    main()
