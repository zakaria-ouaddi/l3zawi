import time

from pycram.datastructures.dataclasses import Color
from pycram.datastructures.enums import Arms, WorldMode
from pycram.designators.object_designator import *
from pycram.process_module import real_robot, simulated_robot
from pycram.robot_plans.actions import *
from pycram.ros_utils.object_state_updater import RobotStateUpdater
from pycram.ros_utils.robot_state_updater import WorldStateUpdater
from pycram.worlds.bullet_world import BulletWorld
from pycrap.ontologies import Robot
from pycram.external_interfaces import giskard

box1Start = PoseStamped.from_list([0.75,-0.25,0.9], [0, 0, 0, 1])
box2Start = PoseStamped.from_list([0.75,0,0.9], [0, 0, 0, 1])
box3Start = PoseStamped.from_list([0.75,0.25,0.9], [0, 0, 0, 1])

box1Target = PoseStamped.from_list([0.75,0, 0.95])
box3Target = PoseStamped.from_list([0.75,0, 1])

world = BulletWorld(WorldMode.GUI)

tracy = Object("tracy", Robot, "tracy.urdf", pose=PoseStamped.from_list([0, 0, 0], [0, 0, 0, 1]), ignore_cached_files=True)
tracy_description = ObjectDesignatorDescription(names=["tracy"]).resolve()

box1 = Object("box1", PhysicalObject, "test_box.urdf", pose=box1Start, color=Color(1, 0, 0, 1))
box2 = Object("box2", PhysicalObject, "test_box.urdf", pose=box2Start, color=Color(0, 1, 0, 1))
box3 = Object("box3", PhysicalObject, "test_box.urdf", pose=box3Start, color=Color(1, 0, 1, 1))

r = WorldStateUpdater("/tf", "/joint_states")

with real_robot:
    #MoveTCPMotion(target=PoseStamped.from_list([1, -0.5, 1.5], [0, 0, 0, 1]), arm=Arms.RIGHT).perform()
    #MoveTCPMotion(target=PoseStamped.from_list([1, 0.5, 1.5], [0, 0, 0, 1]), arm=Arms.LEFT).perform()

    ParkArmsActionDescription(Arms.BOTH).perform()
    PickAndPlaceActionDescription(box1, box1Target, Arms.RIGHT, GraspDescription(ApproachDirection.RIGHT, VerticalAlignment.TOP)).perform()
    PickAndPlaceActionDescription(box3, box3Target, Arms.LEFT, GraspDescription(ApproachDirection.LEFT, VerticalAlignment.TOP)).perform()

print("Plan finished")
r._stop_subscription()

#world.exit()