
# ---------------------------------------------------------------------------
# Kinova PI‑0 – Docker image
#   * ROS Noetic + Gazebo + ros_kortex
#   * Python deps for PI‑0 client + policy
# ---------------------------------------------------------------------------
FROM ros:noetic-desktop

# Avoid interactive apt dialogs
ARG DEBIAN_FRONTEND=noninteractive

# ----------------------------  System dependencies  -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential \
    # Kortex + Gazebo integration
    ros-noetic-kortex-msgs ros-noetic-kortex-gazebo \
    ros-noetic-ros-control ros-noetic-ros-controllers \
    ros-noetic-cv-bridge python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------  Python deps  -------------------------------
COPY examples/kinova_real/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ----------------------------  Workspace  ---------------------------------
WORKDIR /pi0
COPY . /pi0

ENV PYTHONPATH=/pi0:$PYTHONPATH
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Default entrypoint launches Gazebo sim; override in docker‑compose if needed
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && roslaunch kortex_gazebo spawn_7dof_gazebo.launch"]
