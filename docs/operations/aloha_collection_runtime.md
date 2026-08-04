# Project-Owned ALOHA Collection Runtime

The ROS2 collection runtime for this project is stored at:

```text
/home/eii/openpi0.5-rtc-reward-learning/third_party/aloha_collection
```

It is an audited source snapshot of `/home/eii/aloha-2.0`. The legacy source
directory is not modified by this project and must not be used for new
collection launches after this runtime is deployed.

## Preview

From machine 103:

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
third_party/aloha_collection/scripts/collect.sh --dry-run
```

The preview must show the project-owned source as the host side of the mount:

```text
/home/eii/openpi0.5-rtc-reward-learning/third_party/aloha_collection:/root/interbotix_ws/src/aloha
```

## Collection

Only start collection when the operator has cleared the real-robot workspace
and explicitly authorized robot operation:

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
third_party/aloha_collection/scripts/collect.sh
```

Additional recorder arguments follow `--`, for example:

```bash
third_party/aloha_collection/scripts/collect.sh -- \
  --start-trigger b
```

The launcher owns the `aloha2-collect` container and resolves its source root
from the copied runtime directory. Do not launch
`/home/eii/aloha-2.0/scripts/collect.sh` for this project.

## Runtime Separation

- ROS2 data collection uses the project-owned `aloha_collection` copy and the
  standalone `aloha2-collect` container.
- ROS1 inference continues to use the root `docker-compose.yml` and its
  `aloha_ros_nodes` service.
- The root Compose mount is intentionally unchanged by the recorder health
  fix.

## Post-Rearm Health Contract

After the operator completes the open-close rearm gesture, the recorder:

1. restores teleoperation modes and torque state;
2. establishes a new joint-state sequence baseline;
3. waits for three new valid samples from every initialized robot;
4. enters `teleop_wait` with the unchanged 100 ms leader freshness limit.

Failure to recover within the existing two-second health-gate timeout remains
fail-closed and enters the existing safe cleanup path.
