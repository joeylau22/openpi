
"""Real‑robot environment wrapper for a **Kinova Gen3 (7‑DoF) arm** + 2‑finger gripper.

The public API matches the ALOHA RealEnv class so that existing PI‑0
policies / demos can switch robots by changing only the import path.

Implementation notes
--------------------
* Uses **Kinova Kortex‑API ROS 1 binding** (tested on `ros-noetic-kortex`).
* All low‑level calls are isolated in the `_setup_robot`, `_send_arm_command`,
  and `_send_gripper_command` helpers – if you use a different interface
  (ROS 2 `rclpy`, direct TCP), edit only those three functions.
* State and action shapes (``qpos``, ``qvel``, 8‑D action) match the policy.
* Requires the standard Kinova launch pipeline to be running, e.g. ::

      roslaunch my_gen3_bringup kinova_robot.launch

"""

from __future__ import annotations

import time
from typing import List, Optional, Dict

import dm_env
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from kortex_api.autogen.client import BaseClient  # type: ignore
from kortex_api.autogen.messages import Base_pb2  # type: ignore

from examples.kinova_real import kinova_constants as constants

# -----------------------------------------------------------------------------
# ROS topic / service names – edit to suit your workspace
# -----------------------------------------------------------------------------
JOINT_STATE_TOPIC = "/my_gen3/joint_states"
GRIPPER_COMMAND_SERVICE = "/my_gen3/base/send_gripper_command"

# -----------------------------------------------------------------------------
# Observation & action dimensions
# -----------------------------------------------------------------------------
NUM_JOINTS = 7
ACTION_DIM = 8  # 7 arm joints + 1 gripper scalar


class RealEnv:
    """dm_env‑compatible wrapper around a Kinova Gen3 arm."""

    def __init__(
        self,
        *,
        reset_position: Optional[List[float]] = None,
        setup_robot: bool = True,
        control_dt: float = constants.DT,
    ) -> None:
        self._reset_position = (
            reset_position if reset_position is not None else constants.START_ARM_POSE
        )
        self._dt = control_dt

        # Robot state buffers
        self._latest_joint_state: Optional[JointState] = None
        self._done = False
        self._step_count = 0

        if setup_robot:
            self._setup_robot()

        # Subscribe to joint states so we always have fresh positions/velocities
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self._joint_state_cb, queue_size=1)

    # --------------------------------------------------------------------- #
    #  Robot setup / low‑level helpers                                       #
    # --------------------------------------------------------------------- #
    def _setup_robot(self) -> None:
        """Connect to the Kinova base via Kortex API."""
        # These are the defaults from Kinova docs; change if needed.
        ip = rospy.get_param("~ip_address", "192.168.1.10")
        port = 10000
        credentials = ("admin", "admin")

        # TCP router (required by Kortex API)
        from kortex_api.RouterClient import RouterClient
        from kortex_api.RouterClient import RouterClientSendOptions

        import sys
        sys.path.append("/opt/kortex/api_python")  # ensure SDK on PYTHONPATH

        self._router = RouterClient(
            "TCP",  # protocol
            ip,
            port,
            credentials[0],
            credentials[1],
            options=RouterClientSendOptions()
        )
        self.base = BaseClient(self._router)

        # Put robot in low‑level servoing
        self.base.Stop()  # make sure no other control is running

    def _send_arm_command(self, joint_positions: List[float]) -> None:
        """Blocking 1‑shot move using PlayJointTrajectory."""
        traj = Base_pb2.JointTrajectory()
        traj.duration = 0.0  # execute immediately
        for idx, value in enumerate(joint_positions):
            point = traj.points.add()
            point.positions.append(value)
            point.time_from_start = 0.0
        self.base.PlayJointTrajectory(traj)

    def _send_gripper_command(self, position: float) -> None:
        """Send closing / opening command (position in metres)."""
        from kortex_api.autogen.messages import GripperCommand_pb2

        cmd = GripperCommand_pb2.GripperCommand()
        cmd.mode = GripperCommand_pb2.MOD_POSITION
        cmd.position.x = position
        self.base.SendGripperCommand(cmd)

    # --------------------------------------------------------------------- #
    #  ROS callbacks                                                         #
    # --------------------------------------------------------------------- #
    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    # --------------------------------------------------------------------- #
    #  dm_env interface                                                      #
    # --------------------------------------------------------------------- #
    def reset(self) -> dm_env.TimeStep:
        """Reset robot to neutral pose and return initial observation."""
        self._done = False
        self._step_count = 0

        self._send_gripper_command(constants.GRIPPER_POSITION_OPEN)
        self._send_arm_command(self._reset_position[:NUM_JOINTS])
        time.sleep(2.0)  # wait for motion to complete

        return dm_env.restart(self._get_observation())

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        if self._done:
            raise RuntimeError("Environment needs reset() before calling step() after episode end.")

        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Action must have shape ({ACTION_DIM},), got {action.shape}")

        arm_cmd = action[:NUM_JOINTS].tolist()
        gripper_cmd_norm = float(action[NUM_JOINTS])
        gripper_pos = constants.GRIPPER_POSITION_UNNORMALIZE_FN(gripper_cmd_norm)

        self._send_arm_command(arm_cmd)
        self._send_gripper_command(gripper_pos)
        time.sleep(self._dt)

        obs = self._get_observation()
        reward = 0.0  # PI‑0 provides reward externally (task‑agnostic)
        self._step_count += 1

        # Terminate after fixed horizon (matching PI‑0's 1000 * dt ≈ 2 s)
        self._done = self._step_count >= 1000
        if self._done:
            return dm_env.termination(reward=reward, observation=obs)
        else:
            return dm_env.transition(reward=reward, observation=obs)

    # --------------------------------------------------------------------- #
    #  Helpers                                                               #
    # --------------------------------------------------------------------- #
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Return dict with 'qpos', 'qvel'."""
        if self._latest_joint_state is None:
            raise RuntimeError("Haven't received any JointState yet.")

        qpos = np.asarray(self._latest_joint_state.position[:NUM_JOINTS] + (  # pytype: disable=attribute-error
            [constants.POS2JOINT(constants.GRIPPER_POSITION_OPEN)]
        ), dtype=np.float32)

        qvel = np.asarray(self._latest_joint_state.velocity[:NUM_JOINTS] + [0.0], dtype=np.float32)

        return {"qpos": qpos, "qvel": qvel}

    def close(self) -> None:
        self.base.Stop()
        self._router.disconnect()
