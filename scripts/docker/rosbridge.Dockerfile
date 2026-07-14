# ROS1 Noetic rosbridge service for ROS MCP access.
#
# Keep this as a project-local image layer instead of installing packages on the
# 103 host. The base image matches aloha_ros_nodes so custom ALOHA message
# packages and the ROS workspace layout stay aligned with the active robot stack.

FROM lyl472324464/robot:aloha-ros1.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-twisted \
        ros-noetic-rosbridge-server \
    && rm -rf /var/lib/apt/lists/*
