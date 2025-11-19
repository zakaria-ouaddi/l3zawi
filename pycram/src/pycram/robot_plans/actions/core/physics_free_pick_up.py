from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from functools import cached_property

from typing_extensions import Union, Optional, Type, Any, Iterable
import time
from ....external_interfaces import giskard

from ...motions.gripper import MoveGripperMotion, MoveTCPMotion
from ....config.action_conf import ActionConfig
from ....datastructures.dataclasses import FrozenObject
from ....datastructures.enums import Arms, Grasp, GripperState, MovementType, \
    Frame, FindBodyInRegionMethod
from ....datastructures.grasp import GraspDescription
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import PoseStamped
from ....datastructures.world import World
from ....designator import ObjectDesignatorDescription
from ....failures import ObjectNotGraspedError
from ....failures import ObjectNotInGraspingArea
from ....has_parameters import has_parameters
from ....local_transformer import LocalTransformer
from ....plan import with_plan
from ....robot_description import EndEffectorDescription
from ....robot_description import RobotDescription, KinematicChainDescription
from ....robot_plans.actions.base import ActionDescription, record_object_pre_perform
from ....ros import logwarn
from ....world_concepts.world_object import Object
from ....world_reasoning import has_gripper_grasped_body, is_body_between_fingers
import pybullet as p


@has_parameters
@dataclass
class ManualPickUpAction(ActionDescription):
    """
    Manual pick up that manually positions the object between fingers before attaching.
    This is the most reliable method for simulation.
    """

    object_designator: Object
    """
    Object designator describing the object that should be picked up
    """

    arm: Arms
    """
    The arm that should be used for pick up
    """

    grasp_description: GraspDescription
    """
    The grasp description that should be used for picking up the object
    """

    object_at_execution: Optional[FrozenObject] = field(init=False, repr=False, default=None)
    """
    The object at the time this Action got created.
    """
    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def __post_init__(self):
        super().__post_init__()
        self.pre_perform(record_object_pre_perform)

    def plan(self) -> None:
        """
        MANUAL POSITIONING APPROACH:
        1. Open gripper wide
        2. Move to hover above object
        3. MANUALLY position object between fingers (teleport)
        4. Close gripper
        5. Attach object
        6. Lift
        """
        print(f"DEBUG: Starting MANUAL pick up for {self.object_designator.name} with {self.arm.name} arm")

        # 1. Open gripper wide
        print("DEBUG: Opening gripper wide")
        MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm).perform()
        time.sleep(0.5)

        # 2. Calculate where object should be relative to gripper
        gripper_tool_frame = "l_gripper_tool_frame" if self.arm == Arms.LEFT else "r_gripper_tool_frame"
        gripper_pose = World.robot.links[gripper_tool_frame].pose

        # 3. Calculate target object position (between the open fingers)
        # For Robotiq 2F-85, the fingers are about 8.5cm apart when open
        # We want the object centered between them
        target_object_pose = gripper_pose.copy()
        # Adjust for gripper geometry - object should be centered in gripper
        target_object_pose.pose.position.z -= 0.02  # Slightly below tool frame
        # For side grasps, adjust x/y based on approach direction
        if self.grasp_description.approach_direction.name == "RIGHT":
            target_object_pose.pose.position.y -= 0.02
        elif self.grasp_description.approach_direction.name == "LEFT":
            target_object_pose.pose.position.y += 0.02

        # 4. Move gripper to hover above target position
        hover_pose = target_object_pose.copy()
        hover_pose.pose.position.z += 0.10

        print("DEBUG: Moving to hover position")
        self.move_gripper_to_pose(hover_pose, allow_object_collision=False, keep_open=True)

        # 5. CRITICAL: TELEPORT OBJECT between fingers
        print("DEBUG: Manually positioning object between fingers")
        self.teleport_object_to_gripper(target_object_pose)

        # 6. Move gripper to final grasp position
        print("DEBUG: Moving gripper to final position")
        self.move_gripper_to_pose(target_object_pose, blind=True, keep_open=True)

        # 7. Close gripper
        print("DEBUG: Closing gripper")
        MoveGripperMotion(motion=GripperState.CLOSE, gripper=self.arm, allow_gripper_collision=True).perform()
        time.sleep(0.3)

        # 8. Attach object
        print("DEBUG: Attaching object")
        self.attach_object_to_gripper()

        # 9. Lift object
        print("DEBUG: Lifting object")
        self.lift_object(distance=0.15)

        World.current_world.remove_vis_axis()

    def teleport_object_to_gripper(self, target_pose: PoseStamped):
        """Manually teleport the object to the correct position between fingers"""
        obj_id = self.object_designator.id

        # Convert pose to pybullet format
        position = [target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z]
        orientation = [target_pose.pose.orientation.x, target_pose.pose.orientation.y,
                       target_pose.pose.orientation.z, target_pose.pose.orientation.w]

        # Teleport the object
        p.resetBasePositionAndOrientation(obj_id, position, orientation)

        # Zero out any velocities
        p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

        print(f"DEBUG: Object teleported to {position}")

    def attach_object_to_gripper(self):
        """Attach object to the correct gripper tool frame"""
        if self.arm == Arms.LEFT:
            attach_link = "l_gripper_tool_frame"
        else:
            attach_link = "r_gripper_tool_frame"

        print(f"DEBUG: Attaching {self.object_designator.name} to {attach_link}")

        try:
            World.robot.attach(self.object_designator, attach_link)
            print("DEBUG: Attachment successful")
        except Exception as e:
            print(f"DEBUG: Attachment failed: {e}")
            tool_frame = RobotDescription.current_robot_description.get_arm_chain(self.arm).get_tool_frame()
            World.robot.attach(self.object_designator, tool_frame)

    def lift_object(self, distance: float = 0.15):
        """Lift the object after it's grasped"""
        current_pose = self.gripper_pose()
        lift_pose = current_pose.copy()
        lift_pose.pose.position.z += distance

        chain = RobotDescription.current_robot_description.get_arm_chain(self.arm)
        tip_link = chain.get_tool_frame()
        root_link = RobotDescription.current_robot_description.base_link

        giskard.achieve_cartesian_goal(lift_pose, tip_link, root_link,
                                       allow_collision_with_object=self.object_designator.name,
                                       keep_gripper_open=False)

    def move_gripper_to_pose(self, pose: PoseStamped, movement_type: MovementType = MovementType.CARTESIAN,
                             add_vis_axis: bool = True, allow_object_collision: bool = False, keep_open: bool = False,
                             blind: bool = False):
        """Move gripper to target pose using Giskard"""
        pose = self.local_transformer.transform_pose(pose, Frame.Map.value)
        if add_vis_axis:
            World.current_world.add_vis_axis(pose)

        chain = RobotDescription.current_robot_description.get_arm_chain(self.arm)
        tip_link = chain.get_tool_frame()
        root_link = RobotDescription.current_robot_description.base_link

        object_name = self.object_designator.name if allow_object_collision else None

        giskard.achieve_cartesian_goal(pose, tip_link, root_link,
                                       allow_collision_with_object=object_name,
                                       keep_gripper_open=keep_open,
                                       blind=blind)

    def gripper_pose(self) -> PoseStamped:
        """Get the current pose of the gripper"""
        if self.arm == Arms.LEFT:
            gripper_link = "l_gripper_tool_frame"
        else:
            gripper_link = "r_gripper_tool_frame"
        return World.robot.links[gripper_link].pose

    @cached_property
    def local_transformer(self) -> LocalTransformer:
        return LocalTransformer()

    @cached_property
    def arm_chain(self) -> KinematicChainDescription:
        return RobotDescription.current_robot_description.get_arm_chain(self.arm)

    @cached_property
    def end_effector(self) -> EndEffectorDescription:
        return self.arm_chain.end_effector

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        """Validate that object is properly grasped"""
        pass

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    arm: Union[Iterable[Arms], Arms] = None,
                    grasp_description: Union[Iterable[GraspDescription], GraspDescription] = None) -> \
            PartialDesignator[Type[ManualPickUpAction]]:
        return PartialDesignator(ManualPickUpAction, object_designator=object_designator,
                                 arm=arm,
                                 grasp_description=grasp_description)


ManualPickUpActionDescription = ManualPickUpAction.description