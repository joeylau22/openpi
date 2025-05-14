
"""dm_env wrapper for the **Gazebo** Kinova Gen3 simulation (ros_kortex).

Unlike *kinova_real_env.py* this version speaks directly to the
`/my_gen3/joint_trajectory_controller/command` topic (ros_control).
It requires **ros_kortex/kortex_gazebo** to be running.

Observation:
    qpos : (8,) float32   joint position rad + gripper pos (m)
    qvel : (8,) float32   joint velocity rad/s + gripper vel (m/s)
Action:
    ndarray shape (8,)    7 joint positions (rad) + gripper open∈[0,1]

Usage
-----
    roslaunch kortex_gazebo spawn_7dof_gazebo.launch
    python examples/kinova_sim/kinova_sim_main.py
"""

from __future__ import annotations
import time
from typing import List, Dict, Optional

import dm_env
import numpy as np
#import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

from examples.kinova_real import kinova_constants as constants

TRAJ_TOPIC = "/my_gen3/joint_trajectory_controller/command"
NUM_JOINTS = 7
ACTION_DIM = 8

class SimEnv(dm_env.Environment):
    def __init__(self, *, control_dt: float = constants.DT) -> None:
        self._dt = control_dt
        self._latest_js: Optional[JointState] = None
        self._step_counter = 0
        self._done = False

#        rospy.init_node("kinova_sim_env", anonymous=True)
#        self._traj_pub = rospy.Publisher(TRAJ_TOPIC, JointTrajectory, queue_size=1)
#        rospy.Subscriber("/my_gen3/joint_states", JointState, self._js_cb, queue_size=1)

        # Wait until first JointState
#        rospy.loginfo("Waiting for /my_gen3/joint_states...")
#        while self._latest_js is None and not rospy.is_shutdown():
#            time.sleep(0.05)

    # ------------------------------------------------------------------ #
    #  Callbacks                                                         #
    # ------------------------------------------------------------------ #
    def _js_cb(self, msg: JointState):
        self._latest_js = msg

    # ------------------------------------------------------------------ #
    #  dm_env                                                            #
    # ------------------------------------------------------------------ #
    def reset(self) -> dm_env.TimeStep:
        self._send_arm(constants.START_ARM_POSE[:NUM_JOINTS])
        self._send_gripper(constants.GRIPPER_POSITION_OPEN)
        time.sleep(2.0)

        self._step_counter = 0
        self._done = False
        return dm_env.restart(self._get_obs())

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        if self._done:
            raise RuntimeError("reset environment before stepping again")

        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Action must have shape {ACTION_DIM}, got {action.shape}")

        self._send_arm(action[:NUM_JOINTS].tolist())
        self._send_gripper(constants.GRIPPER_POSITION_UNNORMALIZE_FN(float(action[NUM_JOINTS])))
        time.sleep(self._dt)

        self._step_counter += 1
        self._done = self._step_counter >= 1000
        obs = self._get_obs()
        reward = 0.0

        if self._done:
            return dm_env.termination(reward, obs)
        else:
            return dm_env.transition(reward, obs)

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _send_arm(self, positions: List[float]):
        traj = JointTrajectory()
        traj.joint_names = constants.JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = positions
#        pt.time_from_start = rospy.Duration.from_sec(self._dt)
        traj.points.append(pt)
        self._traj_pub.publish(traj)

    def _send_gripper(self, position_m: float):
        # Simulation exposes a fake joint 'finger_joint'; mirror pos.
        grasp_traj = JointTrajectory()
        grasp_traj.joint_names = ["finger_joint"]
        pt = JointTrajectoryPoint()
        pt.positions = [constants.POS2JOINT(position_m)]
#        pt.time_from_start = rospy.Duration.from_sec(self._dt)
        grasp_traj.points.append(pt)
        self._traj_pub.publish(grasp_traj)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        js = self._latest_js
        assert js is not None
        qpos = list(js.position[:NUM_JOINTS]) + [constants.JOINT2POS(js.position[-1])]
        qvel = list(js.velocity[:NUM_JOINTS]) + [0.0]
        return {"qpos": np.asarray(qpos, dtype=np.float32),
                "qvel": np.asarray(qvel, dtype=np.float32)}

    def close(self):
        pass
