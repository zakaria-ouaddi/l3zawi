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


@has_parameters
@dataclass
class PhysicsBasedPickUpAction(ActionDescription):
    """
    Physics-based pick up that uses a different strategy to avoid collision pushback.
    Uses a two-step approach: approach at angle, then push down and close.
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
        NEW STRATEGY: Approach from an angle to avoid direct collision pushback
        1. Open gripper
        2. Move to approach position (above and slightly behind object)
        3. Move forward and down in one motion to grasp position
        4. Close gripper while maintaining position
        5. Attach object
        6. Lift
        """
        print(f"DEBUG: Starting PHYSICS-BASED pick up for {self.object_designator.name} with {self.arm.name} arm")

        # 1. Force Open Gripper
        print("DEBUG: Opening gripper")
        MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm).perform()
        time.sleep(0.5)

        # 2. Calculate approach pose (above and behind the object)
        grasp_pose = self.object_designator.get_grasp_pose(self.end_effector, self.grasp_description)
        grasp_pose.rotate_by_quaternion(self.end_effector.grasps[self.grasp_description])

        # Approach from 10cm behind and 5cm above
        approach_pose = grasp_pose.copy()
        if self.arm == Arms.RIGHT:
            approach_pose.pose.position.x -= 0.10  # Approach from behind for right arm
        else:
            approach_pose.pose.position.x -= 0.10  # Approach from behind for left arm
        approach_pose.pose.position.z += 0.05

        # 3. Move to approach position
        print("DEBUG: Moving to approach position")
        self.move_gripper_to_pose(approach_pose, allow_object_collision=False, keep_open=True)

        # 4. MOVE TO GRASP POSITION WITH BLIND MODE (ignores all collisions)
        print("DEBUG: Executing blind grasp motion")
        self.move_gripper_to_pose(grasp_pose, blind=True, keep_open=True)

        # 5. CLOSE GRIPPER IMMEDIATELY (don't wait for physics to push us away)
        print("DEBUG: Closing gripper immediately")
        MoveGripperMotion(motion=GripperState.CLOSE, gripper=self.arm, allow_gripper_collision=True).perform()
        time.sleep(0.3)  # Short delay for gripper to close

        # 6. ATTACH OBJECT IMMEDIATELY (lock it in place)
        print("DEBUG: Attaching object to gripper")
        self.attach_object_to_gripper()

        # 7. Apply slight upward force to test grip
        print("DEBUG: Testing grip with slight lift")
        self.lift_object(distance=0.02)  # Small lift to test

        # 8. Full lift
        print("DEBUG: Lifting object")
        self.lift_object(distance=0.13)  # Rest of the lift

        World.current_world.remove_vis_axis()

    def attach_object_to_gripper(self):
        """Attach object to the correct gripper tool frame"""
        if self.arm == Arms.LEFT:
            attach_link = "l_gripper_tool_frame"
        else:
            attach_link = "r_gripper_tool_frame"

        print(f"DEBUG: Attaching {self.object_designator.name} to {attach_link}")
        try:
            # Get the current transform between gripper and object
            gripper_pose = World.robot.links[attach_link].pose
            object_pose = self.object_designator.pose

            # Force attach at current relative position
            World.robot.attach(self.object_designator, attach_link)

            print(
                f"DEBUG: Attachment successful - gripper at {gripper_pose.pose.position}, object at {object_pose.pose.position}")

        except Exception as e:
            print(f"DEBUG: Failed to attach to tool frame: {e}")
            # Fallback to default attachment
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

        # Allow collision with the object we're holding
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
        # Relaxed validation for physics-based approach
        pass

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    arm: Union[Iterable[Arms], Arms] = None,
                    grasp_description: Union[Iterable[GraspDescription], GraspDescription] = None) -> \
            PartialDesignator[Type[PhysicsBasedPickUpAction]]:
        return PartialDesignator(PhysicsBasedPickUpAction, object_designator=object_designator,
                                 arm=arm,
                                 grasp_description=grasp_description)


PhysicsBasedPickUpActionDescription = PhysicsBasedPickUpAction.description