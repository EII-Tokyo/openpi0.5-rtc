import copy

import pytest

from examples.fastumi.prepare_two_phase import POSE_NAMES
from examples.fastumi.prepare_two_phase import partition


def inventory():
    return {
        "repo_id": "test/public",
        "revision": "a" * 40,
        "tasks": [
            {
                "path": f"single_arm/task_{t}",
                "info": {
                    "codebase_version": "v2.1",
                    "fps": 20,
                    "total_episodes": 8,
                    "total_frames": 36,
                    "features": {
                        "observation.state": {"shape": [7], "names": {"motors": POSE_NAMES}},
                        "action": {"shape": [7], "names": {"motors": POSE_NAMES}},
                        "observation.images.camera_rgb_image": {"dtype": "video", "shape": [720, 1280, 3]},
                    },
                },
                "episodes": [{"episode_index": e, "length": e + 1, "tasks": [f"task {t}"]} for e in range(8)],
                "tasks": [{"task_index": 0, "task": f"task {t}"}],
            }
            for t in range(3)
        ],
    }


def test_partition_is_disjoint_exhaustive_and_stable():
    source = inventory()
    original = copy.deepcopy(source)
    split = partition(source, 7, seed=42)
    assert split["summary"]["posttrain"]["episodes"] == 7
    assert split["summary"]["midtrain"]["episodes"] == 17
    assert sum(s["frames"] for s in split["summary"].values()) == 108
    ids = [
        {(s["path"], e) for s in split["public_phases"][phase] for e in s["episode_indices"]}
        for phase in ("midtrain", "posttrain")
    ]
    assert not ids[0] & ids[1]
    assert len(ids[0] | ids[1]) == 24
    assert source == original
    source["tasks"].reverse()
    for task in source["tasks"]:
        task["episodes"].reverse()
    assert partition(source, 7, seed=42) == split
    assert partition(source, 7, seed=43)["public_phases"] != split["public_phases"]


@pytest.mark.parametrize("count", [0, -1, 24, 25])
def test_partition_rejects_empty_phase(count):
    with pytest.raises(ValueError, match="positive count"):
        partition(inventory(), count)


def test_partition_rejects_bad_metadata():
    source = inventory()
    source["tasks"][0]["episodes"][0]["episode_index"] = 1
    with pytest.raises(ValueError, match="Missing or duplicate"):
        partition(source, 7)
    source = inventory()
    source["tasks"][0]["episodes"][0]["length"] += 1
    with pytest.raises(ValueError, match="frame counts"):
        partition(source, 7)
    source = inventory()
    source["tasks"][0]["episodes"][0]["tasks"] = ["wrong instruction"]
    with pytest.raises(ValueError, match="instructions"):
        partition(source, 7)
