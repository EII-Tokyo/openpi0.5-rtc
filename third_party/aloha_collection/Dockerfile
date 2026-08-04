FROM osrf/ros:humble-desktop-full

# Update package lists and install curl
RUN apt update && apt install -y curl

# Download and install Interbotix X-Series Arm
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
RUN chmod +x xsarm_amd64_install.sh
RUN ./xsarm_amd64_install.sh -d humble -n

WORKDIR /root/interbotix_ws/src/aloha

COPY . .

# Change to interbotix_ws directory and run rosdep with error handling
WORKDIR /root/interbotix_ws

# Update package lists and install missing dependencies
RUN apt update && apt install -y \
    python3-pip \
    python3-rosdep \
    python3-serial \
    python3-colcon-common-extensions

# Initialize rosdep if not already done
RUN rosdep init || true
RUN rosdep update

# Install dependencies with error handling - skip problematic packages
RUN rosdep install --from-paths src --ignore-src -r -y --skip-keys="ament_python" || true

# Manually install problematic packages that might be missing
RUN apt install -y \
    ros-humble-apriltag-ros \
    ros-humble-usb-cam \
    || echo "Some packages may not be available, continuing..."

# Fix the FK update setting
RUN sed -i 's/iterative_update_fk = True/iterative_update_fk = False/' /root/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot/arm.py || true

# Build the workspace
RUN colcon build

# Setup environment
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /root/interbotix_ws/install/setup.bash" >> ~/.bashrc

# Install Python dependencies
RUN pip install dm_env tqdm h5py

CMD ["bash"]
