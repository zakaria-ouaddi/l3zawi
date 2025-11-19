from copy import deepcopy
from geometry_msgs.msg import PoseStamped, Quaternion
from giskardpy_ros.python_interface.python_interface import GiskardWrapperNode
from giskardpy_ros.ros2 import rospy

# --------------------------- Robot joint poses ---------------------------
ready_pose = {
    # Right arm
    'right_shoulder_pan_joint': 0.07298205015556784,
    'right_shoulder_lift_joint': -1.5451006079298082,
    'right_elbow_joint': 1.6973897170940788,
    'right_wrist_1_joint': 1.2240374339159892,
    'right_wrist_2_joint': 0.9671098455282143,
    'right_wrist_3_joint': -2.867628177785656,
    # Right gripper
    'right_robotiq_85_left_knuckle_joint':  4.3396027437163125e-18,
    'right_robotiq_85_left_finger_tip_joint': 0.0,
    'right_robotiq_85_right_knuckle_joint': 0.0,
    'right_robotiq_85_right_finger_tip_joint': 0.0,
    'right_robotiq_85_left_inner_knuckle_joint': 0.0,
    'right_robotiq_85_right_inner_knuckle_joint': 0.0,
}

tucked_pose = {
    # Right arm
    'right_shoulder_pan_joint': 0.04,
    'right_shoulder_lift_joint': -1.75,
    'right_elbow_joint': 1.32,
    'right_wrist_1_joint': 1.3,
    'right_wrist_2_joint': 1.9,
    'right_wrist_3_joint': -0.6,
    # Right gripper
    'right_robotiq_85_left_knuckle_joint': 2.6037616462297832e-17,
    'right_robotiq_85_right_knuckle_joint': 0.01,
    'right_robotiq_85_left_inner_knuckle_joint': 0.02,
    'right_robotiq_85_right_inner_knuckle_joint': 0.02,
    'right_robotiq_85_left_finger_tip_joint': 0.04,
    'right_robotiq_85_right_finger_tip_joint': 0.04,
}

right_closed = {
    "right_robotiq_85_right_finger_tip_joint": 0.0,
    "right_robotiq_85_left_finger_tip_joint": 0.0,
}
right_open = {
    "right_robotiq_85_right_finger_tip_joint": 0.2,
    "right_robotiq_85_left_finger_tip_joint": 0.2,
}

# --------------------------- Initialization ---------------------------
rospy.init_node('stack_cups_giskard')
giskard = GiskardWrapperNode('stack_cups')
giskard.spin_in_background()

giskard.motion_goals.add_joint_position(tucked_pose, name='tuck', end_condition='tuck')
giskard.motion_goals.add_joint_position(ready_pose, start_condition='tuck')
giskard.motion_goals.avoid_all_collisions()
giskard.add_default_end_motion_conditions()




giskard.execute()


