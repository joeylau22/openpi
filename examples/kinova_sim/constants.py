
"""Kinova Gen3 7‑DoF arm – constants for PI‑0 client
These values provide a sane default for running the PI‑0 real‑robot client
on a Kinova Gen3 arm with the 2‑finger gripper.  Adjust   *START_ARM_POSE*
and *GRIPPER_* limits if you have a different Kinova model (e.g. Gen3 Lite,
3‑finger AGI gripper, etc.).
"""

##############################################################################
# Timing
##############################################################################

# Control timestep (s).  Kinova examples run happily at 500 Hz.
DT = 0.002


##############################################################################
# Joints
##############################################################################

# Kinova Kortex default joint names
JOINT_NAMES = [
    "joint_1",  # base rotation
    "joint_2",  # shoulder elevation
    "joint_3",  # shoulder twist
    "joint_4",  # elbow bend
    "joint_5",  # forearm twist
    "joint_6",  # wrist bend
    "joint_7",  # wrist twist
]

# Neutral “ready” pose: 7 arm joints followed by a single gripper position
# (here: gripper fully open). All values are in radians except the last entry,
# which is metres of finger separation.
START_ARM_POSE = [
    0.0,          # joint_1
    -1.0,         # joint_2
    0.0,          # joint_3
    1.57,         # joint_4
    0.0,          # joint_5
    1.57,         # joint_6
    0.0,          # joint_7
    0.09,         # gripper open (m)
]

##############################################################################
# Gripper limits
##############################################################################

# Finger tip separation in metres
GRIPPER_POSITION_OPEN  = 0.09   # fingers fully apart  ≈ 90 mm
GRIPPER_POSITION_CLOSE = 0.00   # fingers touching

# Corresponding single “finger_joint” angle reported in /joint_states (rad)
GRIPPER_JOINT_OPEN  = 0.80
GRIPPER_JOINT_CLOSE = 0.00

##############################################################################
# Helper conversion functions
##############################################################################

# Position <‑‑> normalised [0‑1]
GRIPPER_POSITION_NORMALIZE_FN = lambda x: (
    (x - GRIPPER_POSITION_CLOSE) / (GRIPPER_POSITION_OPEN - GRIPPER_POSITION_CLOSE)
)
GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: (
    x * (GRIPPER_POSITION_OPEN - GRIPPER_POSITION_CLOSE) + GRIPPER_POSITION_CLOSE
)

# Joint <‑‑> normalised [0‑1]
GRIPPER_JOINT_NORMALIZE_FN = lambda x: (
    (x - GRIPPER_JOINT_CLOSE) / (GRIPPER_JOINT_OPEN - GRIPPER_JOINT_CLOSE)
)
GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: (
    x * (GRIPPER_JOINT_OPEN - GRIPPER_JOINT_CLOSE) + GRIPPER_JOINT_CLOSE
)

# Direct conversions
POS2JOINT = lambda x: (
    GRIPPER_POSITION_NORMALIZE_FN(x) * (GRIPPER_JOINT_OPEN - GRIPPER_JOINT_CLOSE) + GRIPPER_JOINT_CLOSE
)
JOINT2POS = lambda x: GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - GRIPPER_JOINT_CLOSE) / (GRIPPER_JOINT_OPEN - GRIPPER_JOINT_CLOSE)
)

# Mid‑range joint value (rad)
GRIPPER_JOINT_MID = 0.5 * (GRIPPER_JOINT_OPEN + GRIPPER_JOINT_CLOSE)
