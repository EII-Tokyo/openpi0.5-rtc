#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-192.168.1.103}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/eii/openpi0.5-rtc-reward-learning}"
SWITCH_TO_USER_ROS=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/probe_103_aloha_readonly_joint_facts.sh [--switch-to-user-ros]

Read-only ALOHA1 joint/order/limit evidence from the user's 103 project.

Default mode does not stop or start containers. With --switch-to-user-ros, the
script stops the known non-user openpi05-rlt robot web/ROS containers and starts
the user's minimal ROS read stack: ros_master, redis, aloha_ros_nodes, rosbridge.

The script never sends arm motion commands, torque commands, write-register
commands, home/sleep tasks, or runtime actor tasks.
USAGE
}

while (($#)); do
  case "$1" in
    --switch-to-user-ros)
      SWITCH_TO_USER_ROS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ssh "$REMOTE_HOST" "REMOTE_PROJECT='$REMOTE_PROJECT' SWITCH_TO_USER_ROS='$SWITCH_TO_USER_ROS' bash -s" <<'REMOTE'
set -euo pipefail

if [ ! -d "$REMOTE_PROJECT" ]; then
  echo "missing project: $REMOTE_PROJECT" >&2
  exit 1
fi

cd "$REMOTE_PROJECT"

echo "# host"
hostname
whoami
pwd

echo
echo "# docker ps before"
docker ps --format '{{.Names}}	{{.Label "com.docker.compose.project"}}	{{.Label "com.docker.compose.service"}}	{{.Status}}' \
  | grep -E 'openpi|aloha|ros|eii|reward|rtc' || true

if [ "$SWITCH_TO_USER_ROS" = "1" ]; then
  echo
  echo "# stopping non-user openpi05-rlt robot-related containers"
  containers=$(
    docker ps --format '{{.Names}}	{{.Label "com.docker.compose.project"}}	{{.Label "com.docker.compose.service"}}' \
      | awk -F '\t' '$2=="openpi05-rlt" && $3 ~ /^(aloha_ros_nodes|ros_master|redis|voice_web_backend|voice_web_frontend)$/ {print $1}'
  )
  if [ -n "$containers" ]; then
    printf '%s\n' "$containers"
    docker stop $containers
  else
    echo "none"
  fi

  echo
  echo "# starting user minimal ROS read stack"
  docker compose --profile rlt up -d --no-build ros_master redis aloha_ros_nodes rosbridge
fi

container="openpi_reward_learning_eii-aloha_ros_nodes-1"
echo
echo "# user container status"
docker ps --format '{{.Names}}	{{.Status}}' | grep "$container"

echo
echo "# wait for ROS joint topics"
for i in $(seq 1 30); do
  if docker exec "$container" bash -lc 'source /opt/ros/noetic/setup.bash; rostopic list 2>/dev/null | grep -q /puppet_left/joint_states'; then
    echo "ROS_READY_AFTER=$i"
    break
  fi
  sleep 1
  if [ "$i" = 30 ]; then
    echo "ROS_READY_TIMEOUT" >&2
    exit 1
  fi
done

echo
echo "# topics"
docker exec "$container" bash -lc \
  'source /opt/ros/noetic/setup.bash; rostopic list | grep -E "/(puppet|master)_(left|right)/(joint_states|commands)|/cam_(high|low|left_wrist|right_wrist)" | sort'

echo
echo "# one joint_states sample per arm"
for topic in /puppet_left/joint_states /puppet_right/joint_states /master_left/joint_states /master_right/joint_states; do
  echo
  echo "## $topic"
  docker exec "$container" bash -lc "source /opt/ros/noetic/setup.bash; timeout 5 rostopic echo -n 1 $topic"
done

ros_setup='source /opt/ros/noetic/setup.bash; source /root/interbotix_ws/devel/setup.bash'

echo
echo "# robot_info"
for ns in puppet_left puppet_right master_left master_right; do
  echo
  echo "## $ns group arm"
  docker exec "$container" bash -lc "$ros_setup; timeout 5 rosservice call /$ns/get_robot_info group arm" | sed -n '1,80p'
  echo
  echo "## $ns single gripper"
  docker exec "$container" bash -lc "$ros_setup; timeout 5 rosservice call /$ns/get_robot_info single gripper" | sed -n '1,60p'
done

echo
echo "# read-only DYNAMIXEL register values"
for ns in puppet_left puppet_right master_left master_right; do
  for spec in "group arm" "single gripper"; do
    set -- $spec
    cmd_type=$1
    name=$2
    for reg in Operating_Mode Min_Position_Limit Max_Position_Limit Profile_Velocity Profile_Acceleration; do
      printf '#REG %s %s/%s %s ' "$ns" "$cmd_type" "$name" "$reg"
      docker exec "$container" bash -lc "$ros_setup; timeout 5 rosservice call /$ns/get_motor_registers $cmd_type $name $reg 0" \
        | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g'
      printf '\n'
    done
  done
done

echo
echo "# docker ps after"
docker ps --format '{{.Names}}	{{.Label "com.docker.compose.project"}}	{{.Label "com.docker.compose.service"}}	{{.Status}}' \
  | grep -E 'openpi|aloha|ros|eii|reward|rtc' || true
REMOTE
