import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, PushRosNamespace
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import Parameter

def generate_launch_description():
    # Define the arguments for the kinematics configuration files
    kinematics_config_left = DeclareLaunchArgument(
        'kinematics_config_left',
        default_value='$(find iai_tracy_ur)/include/iai_tracy_ur/left_ur10e_calibration.yaml',
        description='Path to the left arm kinematics configuration file'
    )

    kinematics_config_right = DeclareLaunchArgument(
        'kinematics_config_right',
        default_value='$(find iai_tracy_ur)/include/iai_tracy_ur/right_ur10e_calibration.yaml',
        description='Path to the right arm kinematics configuration file'
    )

    return LaunchDescription([
        # Declare kinematics configuration arguments
        kinematics_config_left,
        kinematics_config_right,

        # Include the iai_tracy_description launch file
        Node(
            package='iai_tracy_description',
            executable='upload.launch',
            name='upload_description',
            parameters=[
                {'kinematics_config_left': kinematics_config_left, 'kinematics_config_right': kinematics_config_right}
            ]
        ),
        
        # Include the ur_robot_driver for the left arm
        Node(
            package='ur_robot_driver',
            executable='ur10e_bringup.launch',
            namespace='left_arm',
            arguments=[
                'robot_ip:=192.168.102.154',
                'tf_prefix:=left_',
                'controller_config_file:=$(find iai_tracy_ur)/config/ur10e_controllers_left.yaml',
                'controllers:=joint_state_controller_left scaled_pos_joint_traj_controller_left',
                'stopped_controllers:=pos_joint_traj_controller_left',
                'kinematics_config:=$(arg kinematics_config_left)',
                'robot_description_file:=$(find iai_tracy_description)/launch/upload.launch',
                'reverse_port:=50011',
                'script_sender_port:=50012',
                'trajectory_port:=50013',
                'script_command_port:=50014'
            ]
        ),
        
        # Include the ur_robot_driver for the right arm
        Node(
            package='ur_robot_driver',
            executable='ur10e_bringup.launch',
            namespace='right_arm',
            arguments=[
                'robot_ip:=192.168.102.153',
                'tf_prefix:=right_',
                'controller_config_file:=$(find iai_tracy_ur)/config/ur10e_controllers_right.yaml',
                'controllers:=joint_state_controller_right scaled_pos_joint_traj_controller_right',
                'stopped_controllers:=pos_joint_traj_controller_right',
                'kinematics_config:=$(arg kinematics_config_right)',
                'robot_description_file:=$(find iai_tracy_description)/launch/upload.launch',
                'reverse_port:=50001',
                'script_sender_port:=50002',
                'trajectory_port:=5003',
                'script_command_port:=50005'
            ]
        ),

        # Gripper nodes for the right and left grippers
        Node(
            package='robotiq_2f_gripper_control',
            executable='Robotiq2FGripperRtuNode.py',
            name='right_gripper_driver',
            arguments=['/dev/ttyUSB0'],
            namespace='right_gripper'
        ),
        
        Node(
            package='robotiq_2f_gripper_control',
            executable='Robotiq2FGripperRtuNode.py',
            name='left_gripper_driver',
            arguments=['/dev/ttyUSB1'],
            namespace='left_gripper'
        ),

        # Gripper action server nodes
        Node(
            package='robotiq_2f_gripper_action_server',
            executable='robotiq_2f_gripper_action_server_node',
            name='gripper_action_server_right',
            parameters=[{'gripper_name': 'right_gripper'}],
            remappings=[
                ('input', '/right_gripper/Robotiq2FGripperRobotInput'),
                ('output', '/right_gripper/Robotiq2FGripperRobotOutput')
            ]
        ),

        Node(
            package='robotiq_2f_gripper_action_server',
            executable='robotiq_2f_gripper_action_server_node',
            name='gripper_action_server_left',
            parameters=[{'gripper_name': 'left'}],
            remappings=[
                ('input', '/left_gripper/Robotiq2FGripperRobotInput'),
                ('output', '/left_gripper/Robotiq2FGripperRobotOutput')
            ]
        ),

        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{
                'source_list': ['/left_arm/joint_states', '/right_arm/joint_states'],
                'rate': 120,
                'use_gui': False
            }],
            output='screen'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen'
        ),
    ])
