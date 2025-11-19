import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from giskardpy.user_interface import GiskardWrapper
from giskardpy.motion_statechart.monitors.joint_monitors import JointGoalReached
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.model.collision_avoidance_config import DisableCollisionAvoidanceConfig
from giskardpy.model.world_config import WorldConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig


def load_urdf_with_resolved_meshes(urdf_path):
    with open(urdf_path, 'r') as f:
        urdf_str = f.read()

    # Replace package URI with absolute path (make sure to use triple slashes after file:)
    urdf_str = urdf_str.replace(
        'package://iai_tracy_description/meshes/',
        'file://home/zakaria/workspace/ros/src/iai_tracy-ros2-jazzy/iai_tracy_description/meshes/'
    )
    return urdf_str



class MyRobotWorldConfig(WorldConfig):
    def __init__(self):
        super().__init__()

    def setup(self):
        urdf_path = '../../resources/robots/tracy.urdf'
        urdf_string = load_urdf_with_resolved_meshes(urdf_path)
        self.add_robot_urdf(urdf_string, group_name="my_robot")

        # You can add additional world setup here, e.g. fixed joints, colors, etc.
        # Example:
        # self.set_default_color(0.8, 0.8, 0.8, 1.0)

class SimpleArmMover(Node):
    def __init__(self):
        super().__init__('simple_arm_mover')

        # Create world config and setup
        world_config = MyRobotWorldConfig()
        world_config.setup()

        collision_avoidance = DisableCollisionAvoidanceConfig()
        controller_config = QPControllerConfig()

        # Initialize Giskard with your world config
        self.giskard = GiskardWrapper(
            world_config=world_config,
            collision_avoidance_config=collision_avoidance,
            qp_controller_config=controller_config
        )

        # Define joint goals
        right_arm_goal = {
            'r_shoulder_pan_joint': -1.7125,
            'r_shoulder_lift_joint': -0.25672,
            'r_upper_arm_roll_joint': -1.46335,
            'r_elbow_flex_joint': -2.12,
            'r_forearm_roll_joint': 1.76632,
            'r_wrist_flex_joint': -0.10001,
            'r_wrist_roll_joint': 0.05106
        }

        left_arm_goal = {
            'l_shoulder_pan_joint': 1.9652,
            'l_shoulder_lift_joint': -0.26499,
            'l_upper_arm_roll_joint': 1.3837,
            'l_elbow_flex_joint': -2.12,
            'l_forearm_roll_joint': 16.99,
            'l_wrist_flex_joint': -0.10001,
            'l_wrist_roll_joint': 0
        }

        # Add joint position goals
        self.giskard.motion_goals.add_joint_position(goal_state=right_arm_goal, name='right_arm')
        self.giskard.motion_goals.add_joint_position(goal_state=left_arm_goal, name='left_arm')

        # Compile and execute
        self.giskard.compile()
        self.get_logger().info("Moving both arms...")
        self.giskard.execute(sim_time=5.0, plot=False, plot_legend=False)
        self.get_logger().info("Motion complete.")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleArmMover()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

