
# ---------------------------------------------------------------------------
# Kinova PI‑0 SIM – Docker image (Gazebo only, no hardware rules)
#   • ROS Noetic desktop‑full (includes Gazebo 11)
#   • ros_kortex with Gazebo plugins
#   • PI‑0 Python deps + simulator env
# ---------------------------------------------------------------------------
FROM ros:noetic-desktop-full

ARG DEBIAN_FRONTEND=noninteractive

# ---------- system dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip build-essential \
    ros-noetic-kortex-msgs ros-noetic-kortex-gazebo \
    ros-noetic-ros-control ros-noetic-ros-controllers \
    ros-noetic-cv-bridge python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# ---------- python deps ----------
COPY examples/kinova_sim/requirements.txt /tmp/req.txt
RUN pip3 install --no-cache-dir -r /tmp/req.txt

# ---------- workspace ----------
WORKDIR /pi0
COPY . /pi0
ENV PYTHONPATH=/pi0:$PYTHONPATH

# Source ROS at shell spawn
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Default to launching the Gazebo world only.
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && roslaunch kortex_gazebo spawn_7dof_gazebo.launch"]
