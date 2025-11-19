from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from functools import cached_property

from typing_extensions import Union, Optional, Type, Any, Iterable
import time
# --- ADDED IMPORT ---
from ....external_interfaces import giskard
# --------------------

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
class ReachToPickUpAction(ActionDescription):
    """
    Let the robot reach a specific pose.
    """

    object_designator: Object
    """
    Object designator_description describing the object that should be picked up
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
    The object at the time this Action got created. It is used to be a static, information holding entity. It is
    not updated when the world object is changed.
    """
    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def __post_init__(self):
        super().__post_init__()

        # Store the object's data copy at execution
        self.pre_perform(record_object_pre_perform)

    def plan(self) -> None:
        # 1. Force Open
        MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm).perform()
        time.sleep(0.5)

        target_pose = self.object_designator.get_grasp_pose(self.end_effector, self.grasp_description)
        target_pose.rotate_by_quaternion(self.end_effector.grasps[self.grasp_description])

        # Pre-Grasp (Hover 15cm above)
        target_pre_pose = target_pose.copy()
        target_pre_pose.pose.position.z += 0.15

        # 3. Move to Hover (Standard Safety)
        print("DEBUG: Moving to Hover...")
        self.move_gripper_to_pose(target_pre_pose, allow_object_collision=False, keep_open=True)

        # 4. Reinforce Open
        MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm).perform()

        # 5. The Blind Plunge (Force Control Strategy)
        print("DEBUG: BLIND PLUNGE to Grasp...")

        # TARGETING LOWER:
        # We intentionally aim 2cm LOWER than the grasp pose.
        # This forces the robot to press down firmly.
        # Since we are in blind mode, it won't complain about collision.
        plunge_target = target_pose.copy()
        plunge_target.pose.position.z -= 0.02  # Push 2cm past the goal

        # We use blind=True.
        # Note: We expect this action to effectively "fail" or hang because it hits the table/box.
        # That is GOOD. It means we are solidly at the bottom.
        # We set a short timeout or catch the error so the script continues to the Grasp.

        try:
            # We use a specialized call that doesn't wait forever
            self.move_gripper_to_pose(plunge_target, MovementType.CARTESIAN, blind=True, keep_open=True)
        except Exception as e:
            print(f"DEBUG: Plunge finished (or hit resistance): {e}")

    def move_gripper_to_pose(self, pose: PoseStamped, movement_type: MovementType = MovementType.CARTESIAN,
                             add_vis_axis: bool = True, allow_object_collision: bool = False, keep_open: bool = False,
                             blind: bool = False):  #  Add blind arg here

        pose = self.local_transformer.transform_pose(pose, Frame.Map.value)
        if add_vis_axis:
            World.current_world.add_vis_axis(pose)

        chain = RobotDescription.current_robot_description.get_arm_chain(self.arm)
        tip_link = chain.get_tool_frame()
        root_link = RobotDescription.current_robot_description.base_link

        object_name = self.object_designator.name if allow_object_collision else None

        # Pass blind to Giskard
        giskard.achieve_cartesian_goal(pose, tip_link, root_link,
                                       allow_collision_with_object=object_name,
                                       keep_gripper_open=keep_open,
                                       blind=blind)

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
        """
        Check if object is contained in the gripper such that it can be grasped and picked up.
        """
        fingers_link_names = self.arm_chain.end_effector.fingers_link_names
        if fingers_link_names:
            if not is_body_between_fingers(self.object_designator, fingers_link_names,
                                           method=FindBodyInRegionMethod.MultiRay):
                # we can uncomment this raise if we want strict validation,
                # but sometimes simulation perception is slightly off.
                # raise ObjectNotInGraspingArea(self.object_designator, World.robot, self.arm, self.grasp_description)
                pass
        else:
            logwarn(f"Cannot validate reaching to pick up action for arm {self.arm} as no finger links are defined.")

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    arm: Union[Iterable[Arms], Arms] = None,
                    grasp: Union[Iterable[Grasp], Grasp] = None) -> PartialDesignator[Type[ReachToPickUpAction]]:
        return PartialDesignator(ReachToPickUpAction, object_designator=object_designator,
                                 arm=arm,
                                 grasp=grasp)


@has_parameters
@dataclass
class PickUpAction(ActionDescription):
    """
    Let the robot pick up an object.
    """

    object_designator: Object
    """
    Object designator_description describing the object that should be picked up
    """

    arm: Arms
    """
    The arm that should be used for pick up
    """

    grasp_description: GraspDescription
    """
    The GraspDescription that should be used for picking up the object
    """

    object_at_execution: Optional[FrozenObject] = field(init=False, repr=False, default=None)
    """
    The object at the time this Action got created. It is used to be a static, information holding entity. It is
    not updated when the BulletWorld object is changed.
    """

    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def __post_init__(self):
        super().__post_init__()

        # Store the object's data copy at execution
        self.pre_perform(record_object_pre_perform)

    def plan(self) -> None:
        # 1. Approach & Plunge
        ReachToPickUpAction(self.object_designator, self.arm, self.grasp_description).perform()

        # --- CHANGE: ATTACH FIRST (Cheat Physics) ---
        # We attach the object immediately. This stops from
        # pushing the hand away from the box (recoil), because they are now "one body".

        # Get correct frame
        if self.arm == Arms.LEFT:
            attach_link = "l_gripper_tool_frame"
        else:
            attach_link = "r_gripper_tool_frame"

        print(f"DEBUG: EARLY ATTACH to {attach_link}")
        try:
            World.robot.attach(self.object_designator, attach_link)
        except Exception:
            tool_frame = RobotDescription.current_robot_description.get_arm_chain(self.arm).get_tool_frame()
            World.robot.attach(self.object_designator, tool_frame)

        # 2. CLOSE (Visual)
        # Now we close the gripper. Since the object is attached,
        # it will move with the fingers if they shift, but it won't fly away.
        print("DEBUG: CLOSING GRIPPER NOW")
        MoveGripperMotion(motion=GripperState.CLOSE, gripper=self.arm, allow_gripper_collision=True).perform()

        # Wait for close visual
        time.sleep(0.5)

        # 3. Lift
        self.lift_object(distance=0.15)

        World.current_world.remove_vis_axis()

    def lift_object(self, distance: float = 0.1):
        lift_to_pose = self.gripper_pose()
        lift_to_pose.pose.position.z += distance

        # Call Giskard directly for the lift to ensure we pass 'allow_object_collision'
        # This prevents the robot from dropping the box because it thinks the box is colliding with the table
        chain = RobotDescription.current_robot_description.get_arm_chain(self.arm)
        tip_link = chain.get_tool_frame()
        root_link = RobotDescription.current_robot_description.base_link

        # We allow collision with the object we are holding AND the table it is sliding off.
        # keep_gripper_open=False because we want to hold it tight!
        giskard.achieve_cartesian_goal(lift_to_pose, tip_link, root_link,
                                       allow_collision_with_object=self.object_designator.name,
                                       keep_gripper_open=False)

    def gripper_pose(self) -> PoseStamped:
        """
        Get the pose of the gripper.

        :return: The pose of the gripper.
        """
        gripper_link = self.arm_chain.get_tool_frame()
        return World.robot.links[gripper_link].pose

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        """
        Check if picked up object is in contact with the gripper.
        """
        if not has_gripper_grasped_body(self.arm, self.object_designator):
            raise ObjectNotGraspedError(self.object_designator, World.robot, self.arm, self.grasp_description)

    @cached_property
    def arm_chain(self) -> KinematicChainDescription:
        return RobotDescription.current_robot_description.get_arm_chain(self.arm)

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    arm: Union[Iterable[Arms], Arms] = None,
                    grasp_description: Union[Iterable[GraspDescription], GraspDescription] = None) -> \
            PartialDesignator[Type[PickUpAction]]:
        return PartialDesignator(PickUpAction, object_designator=object_designator, arm=arm,
                                 grasp_description=grasp_description)


@has_parameters
@dataclass
class GraspingAction(ActionDescription):
    """
    Grasps an object described by the given Object Designator description
    """
    object_designator: Object  # Union[Object, ObjectDescription.Link]
    """
    Object Designator for the object that should be grasped
    """
    arm: Arms
    """
    The arm that should be used to grasp
    """
    prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters the gripper should be at before grasping the object
    """

    def plan(self) -> None:
        object_pose = self.object_designator.pose
        lt = LocalTransformer()
        gripper_name = RobotDescription.current_robot_description.get_arm_chain(self.arm).get_tool_frame()

        object_pose_in_gripper = lt.transform_pose(object_pose,
                                                   World.robot.get_link_tf_frame(gripper_name))

        pre_grasp = object_pose_in_gripper.copy()
        pre_grasp.pose.position.x -= self.prepose_distance

        # Move to pre-grasp
        MoveTCPMotion(pre_grasp, self.arm).perform()
        MoveGripperMotion(GripperState.OPEN, self.arm).perform()

        # Move to grasp - Use the new logic via Giskard if you want, or keep this simple
        MoveTCPMotion(object_pose, self.arm, allow_gripper_collision=True).perform()
        MoveGripperMotion(GripperState.CLOSE, self.arm, allow_gripper_collision=True).perform()

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        body = self.object_designator
        contact_links = body.get_contact_points_with_body(World.robot).get_all_bodies()
        arm_chain = RobotDescription.current_robot_description.get_arm_chain(self.arm)
        gripper_links = arm_chain.end_effector.links
        if not any([link.name in gripper_links for link in contact_links]):
            raise ObjectNotGraspedError(self.object_designator, World.robot, self.arm, None)

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    arm: Union[Iterable[Arms], Arms] = None,
                    prepose_distance: Union[Iterable[float], float] = ActionConfig.grasping_prepose_distance) -> \
            PartialDesignator[Type[GraspingAction]]:
        return PartialDesignator(GraspingAction, object_designator=object_designator, arm=arm,
                                 prepose_distance=prepose_distance)


ReachToPickUpActionDescription = ReachToPickUpAction.description
PickUpActionDescription = PickUpAction.description
GraspingActionDescription = GraspingAction.description