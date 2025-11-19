import time
import yaml
import rclpy
import copy

from giskardpy_ros.python_interface.python_interface import GiskardWrapperNode
from giskardpy_ros.ros2 import rospy

from geometry_msgs.msg import Pose, PoseStamped


def load_grasps(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    poses = []
    for g in data['grasps']:
        pose = Pose()
        pose.position.x = g['position']['x']
        pose.position.y = g['position']['y']
        pose.position.z = g['position']['z']
        pose.orientation.x = g['orientation']['x']
        pose.orientation.y = g['orientation']['y']
        pose.orientation.z = g['orientation']['z']
        pose.orientation.w = g['orientation']['w']
        poses.append(pose)
    print(f'Loaded {len(poses)} grasps from {file_path}')
    return poses


def offset_pose(original_pose: Pose, dx, dy, dz):
    new_pose = PoseStamped()
    new_pose.header.frame_id = "table"
    new_pose.pose.position.x = original_pose.position.x + dx
    new_pose.pose.position.y = original_pose.position.y + dy
    new_pose.pose.position.z = original_pose.position.z + dz
    new_pose.pose.orientation = original_pose.orientation  # Keep the same
    return new_pose


giskard_node_name = 'asdf'

ready_pose = {
    # Left arm
    'left_shoulder_pan_joint': 0.0,
    'left_shoulder_lift_joint': -1.57,
    'left_elbow_joint': 1.57,
    'left_wrist_1_joint': 0.0,
    'left_wrist_2_joint': 0.0,
    'left_wrist_3_joint': 0.0,
    # Left gripper
    'left_robotiq_85_left_knuckle_joint': 0.8,
    'left_robotiq_85_right_knuckle_joint': 0.8,
    'left_robotiq_85_left_inner_knuckle_joint': 0.8,
    'left_robotiq_85_right_inner_knuckle_joint': 0.8,
    'left_robotiq_85_left_finger_tip_joint': 0.8,
    'left_robotiq_85_right_finger_tip_joint': 0.8,
    # Right arm
    'right_shoulder_pan_joint': 0.0,
    'right_shoulder_lift_joint': -1.57,
    'right_elbow_joint': 1.57,
    'right_wrist_1_joint': 0.0,
    'right_wrist_2_joint': 0.0,
    'right_wrist_3_joint': 0.0,
    # Right gripper
    'right_robotiq_85_left_knuckle_joint': 0.8,
    'right_robotiq_85_right_knuckle_joint': 0.8,
    'right_robotiq_85_left_inner_knuckle_joint': 0.8,
    'right_robotiq_85_right_inner_knuckle_joint': 0.8,
    'right_robotiq_85_left_finger_tip_joint': 0.8,
    'right_robotiq_85_right_finger_tip_joint': 0.8,
}

poses = load_grasps('Cube_Pad_grasps.yaml')

# rclpy.shutdown()

# rospy.signal_shutdown()
rospy.init_node(giskard_node_name)
print('rospy initialized')
giskard = GiskardWrapperNode("my_tracy_node")
giskard.spin_in_background()
print('spinning the nodes')

# ----------- Add the cup -----------
cup_name = "single_cup"
cup_pose_xyz = (0.8, 0.4, 0.01)
cup_size = (0.09, 0.09, 0.09)
cup_color = (0.3, 0.6, 1.0, 1.0)
cup_pose_msg = PoseStamped()
cup_pose_msg.header.frame_id = 'table'
cup_pose_msg.pose.position.x, cup_pose_msg.pose.position.y, cup_pose_msg.pose.position.z = cup_pose_xyz
cup_pose_msg.pose.orientation.w = 1.0
giskard.world.add_box(name=cup_name, size=cup_size, pose=cup_pose_msg)
giskard.world.dye_group(group_name=cup_name, rgba=cup_color)

for pose in poses:
    # transfer absolut pose to the pose of the object on the table set in the YCB_object_publisher.launch.py
    adjusted_pose = offset_pose(pose, 0.8, 0.4, 0.01)

    # goal = PoseStamped()
    goal = adjusted_pose

    # set robot to the starting position
    print(f"Sending goal to Giskard")
    giskard.motion_goals.add_joint_position(ready_pose, name=giskard_node_name, end_condition=giskard_node_name)
    giskard.motion_goals.allow_all_collisions()
    giskard.add_default_end_motion_conditions()

    t = time.time()
    result = giskard.execute()
    print(time.time() - t)
    print("Giskard ready")

    # set the grasp pose
    giskard.motion_goals.add_cartesian_pose(
        goal_pose=goal,
        # root_link='world',
        root_link='right_base_link',
        # tip_link='ee_link',
        tip_link='r_gripper_tool_frame',
    )
    # giskard.motion_goals.allow_all_collisions()
    # giskard.motion_goals.allow_self_collision()
    giskard.motion_goals.allow_all_collisions()
    # giskard.motion_goals.avoid_collision(group1='table', group2=giskard.robot_name)
    giskard.add_default_end_motion_conditions()

    t = time.time()
    result = giskard.execute()
    print(time.time() - t)
    print("Giskard executed")

    time.sleep(3)



#==========================================================================================================#


'''import os
import sys
import yaml
from copy import deepcopy
from geometry_msgs.msg import PoseStamped, Quaternion
from giskardpy_ros.python_interface.python_interface import GiskardWrapperNode
from giskardpy_ros.ros2 import rospy

def load_grasps_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['grasps']

def make_quaternion(qdict):
    q = Quaternion()
    q.x = qdict['x']
    q.y = qdict['y']
    q.z = qdict['z']
    q.w = qdict['w']
    return q

def transform_grasp_pose(grasp, cup_pose):
    # grasp: dict with 'position' and 'orientation'
    # cup_pose: (x, y, z) tuple
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.pose.position.x = cup_pose[0] + grasp['position']['x']
    pose.pose.position.y = cup_pose[1] + grasp['position']['y']
    pose.pose.position.z = cup_pose[2] + grasp['position']['z']
    pose.pose.orientation = make_quaternion(grasp['orientation'])
    return pose

if __name__ == "__main__":
    # ----------- Parameters -----------
    yaml_path = "Cube_Pad_grasps.yaml"  # Path to your yaml file
    cup_name = "single_cup"
    cup_pose_xyz = (0.7, 0.0, 0.95)
    cup_size = (0.09, 0.09, 0.09)
    cup_color = (0.3, 0.6, 1.0, 1.0)

    # Arm poses (customize as needed)
    ready_pose = {
        # Right arm
        'right_shoulder_pan_joint': 0.07298205015556784,
        'right_shoulder_lift_joint': -1.5451006079298082,
        'right_elbow_joint': 1.6973897170940788,
        'right_wrist_1_joint': 1.2240374339159892,
        'right_wrist_2_joint': 0.9671098455282143,
        'right_wrist_3_joint': -2.867628177785656,
        # Right gripper
        'right_robotiq_85_left_knuckle_joint': 4.3396027437163125e-18,
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
        'right_robotiq_85_right_knuckle_joint': 0.0,
        'right_robotiq_85_left_inner_knuckle_joint': 0.0,
        'right_robotiq_85_right_inner_knuckle_joint': 0.0,
        'right_robotiq_85_left_finger_tip_joint': 0.0,
        'right_robotiq_85_right_finger_tip_joint': 0.0,
    }

    # ----------- Init ROS & Giskard -----------
    rospy.init_node('test_grasps_on_single_cup')
    giskard = GiskardWrapperNode('test_grasps_single')
    giskard.spin_in_background()

    # ----------- Add the cup -----------
    cup_pose_msg = PoseStamped()
    cup_pose_msg.header.frame_id = 'world'
    cup_pose_msg.pose.position.x, cup_pose_msg.pose.position.y, cup_pose_msg.pose.position.z = cup_pose_xyz
    cup_pose_msg.pose.orientation.w = 1.0
    giskard.world.add_box(name=cup_name, size=cup_size, pose=cup_pose_msg)
    giskard.world.dye_group(group_name=cup_name, rgba=cup_color)


    # ----------- Move to tucked pose -----------
    giskard.motion_goals.add_joint_position(tucked_pose, name='tuck', end_condition='tuck')
    giskard.motion_goals.add_joint_position(ready_pose, start_condition='tuck')
    giskard.motion_goals.avoid_all_collisions()
    giskard.add_default_end_motion_conditions()
    giskard.execute()

    # ----------- Load grasps -----------
    if not os.path.isfile(yaml_path):
        print(f"ERROR: Could not find yaml file: {yaml_path}")
        sys.exit(1)
    grasps = load_grasps_from_yaml(yaml_path)

    # ----------- Test each grasp -----------
    valid_ids = []
    for grasp in grasps:
        pose_id = grasp['id']
        grasp_pose = transform_grasp_pose(grasp, cup_pose_xyz)

        giskard.motion_goals.avoid_all_collisions()
        giskard.motion_goals.allow_collision(group1='robot', group2=cup_name)
        giskard.motion_goals.allow_self_collision()

        pose_goal_name = f'grasp_{pose_id}'
        giskard.motion_goals.add_cartesian_pose(
            name=pose_goal_name,
            goal_pose=grasp_pose,
            tip_link='r_gripper_tool_frame',
            root_link='right_base_link',
            end_condition=pose_goal_name
        )
        giskard.add_default_end_motion_conditions()

        try:
            giskard.execute()
            print(f"[OK] Grasp pose id {pose_id}")
            valid_ids.append(pose_id)
        except Exception as e:
            print(f"[FAIL] Grasp pose id {pose_id} failed: {e}")

        # Always return to tuck so next test is consistent
        giskard.motion_goals.add_joint_position(tucked_pose, name='tuck', end_condition='tuck')
        giskard.motion_goals.add_joint_position(ready_pose, start_condition='tuck')
        giskard.motion_goals.avoid_all_collisions()
        giskard.add_default_end_motion_conditions()
        giskard.execute()

    print(f"\nValid poses ids: {valid_ids}")'''
