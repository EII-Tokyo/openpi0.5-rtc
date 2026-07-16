from __future__ import annotations

import sys

from aloha_isaac_replay.scripts.replay_aloha_qpos_arm_only import main as arm_replay_main


def main() -> int:
    if "--include-gripper" not in sys.argv:
        sys.argv.append("--include-gripper")
    return arm_replay_main()


if __name__ == "__main__":
    raise SystemExit(main())
