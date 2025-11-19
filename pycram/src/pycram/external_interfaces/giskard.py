from typing import TYPE_CHECKING

from geometry_msgs.msg import PointStamped, QuaternionStamped, Vector3Stamped

from giskard_msgs.msg import CollisionEntry, WorldBody

if TYPE_CHECKING:
    # Only imported for the type checker / PyCharm, not at runtime
    from giskardpy_ros.python_interface.python_interface import GiskardWrapper

import json
import threading

import sys

from ..ros import Time, node
from ..ros import logwarn, loginfo_once, loginfo
from ..ros import get_node_names

from ..datastructures.enums import JointType, ObjectType, Arms
from ..datastructures.pose import PoseStamped
from ..datastructures.world import World
from ..datastructures.dataclasses import MeshVisualShape, BoxVisualShape
from ..ros import get_service_proxy
from ..world_concepts.world_object import Object
from ..robot_description import RobotDescription

from typing_extensions import List, Dict, Callable, Optional
from threading import Lock, RLock
from ..ros import logging as log, node

# giskard_wrapper = None
giskard_wrapper: Optional['GiskardWrapper'] = None
giskard_update_service = None
is_init = False

number_of_par_goals = 0
giskard_lock = Lock()
giskard_rlock = RLock()
with giskard_rlock:
    par_threads = {}
    par_motion_goal = {}


def thread_safe(func: Callable) -> Callable:
    """
    Adds thread safety to a function via a decorator. This uses the giskard_lock

    :param func: Function that should be thread safe
    :return: A function with thread safety
    """

    def wrapper(*args, **kwargs):
        with giskard_rlock:
            return func(*args, **kwargs)

    return wrapper


def init_giskard_interface(func: Callable) -> Callable:
    """
    Checks if the ROS messages are available and if giskard is running, if that is the case the interface will be
    initialized.

    :param func: Function this decorator should be wrapping
    :return: A callable function which initializes the interface and then calls the wrapped function
    """

    def wrapper(*args, **kwargs):

        # from giskardpy_ros.python_interface.old_python_interface import OldGiskardWrapper as GiskardWrapper
        from giskardpy_ros.python_interface.python_interface import GiskardWrapper
        from giskard_msgs.msg import WorldBody, ExecutionState, CollisionEntry
        from geometry_msgs.msg import PoseStamped as ROSPoseStamped, PointStamped, QuaternionStamped, Vector3Stamped

        global giskard_wrapper
        global giskard_update_service
        global is_init
        if is_init and "giskard" in get_node_names():
            return func(*args, **kwargs)
        elif is_init and "giskard" not in get_node_names():
            logwarn("Giskard node is not available anymore, could not initialize giskard interface")
            is_init = False
            giskard_wrapper = None
            return

        if "giskard_msgs" not in sys.modules:
            logwarn("Could not initialize the Giskard interface since the giskard_msgs are not imported")
            return

        if "giskard" in get_node_names():
            giskard_wrapper = GiskardWrapper(node)
            loginfo_once("Successfully initialized Giskard interface")
            is_init = True
        else:
            logwarn("Giskard is not running, could not initialize Giskard interface")
            return
        return func(*args, **kwargs)

    return wrapper


# Believe state management between pycram and giskard


@init_giskard_interface
def initial_adding_objects() -> None:
    """
    Adds object that are loaded in the World to the Giskard belief state, if they are not present at the moment.
    """
    groups = giskard_wrapper.world.get_group_names()
    for obj in World.current_world.objects:
        if obj is World.robot or obj is World.current_world.get_prospection_object_for_object(
                World.robot) or obj.name == "floor":
            continue
        name = obj.name
        if name not in groups:
            print(obj)
            spawn_object(obj)


@init_giskard_interface
def removing_of_objects() -> None:
    """
    Removes objects that are present in the Giskard belief state but not in the World from the Giskard belief state.
    """
    groups = giskard_wrapper.world.get_group_names()
    object_names = list(
        map(lambda obj: object_names.name, World.current_world.objects))
    diff = list(set(groups) - set(object_names))
    for grp in diff:
        giskard_wrapper.world.remove_group(grp)


@init_giskard_interface
def sync_worlds(projection: bool = False) -> None:
    """
    Synchronizes the World and the Giskard belief state, this includes adding and removing objects to the Giskard
    belief state such that it matches the objects present in the World and moving the robot to the position it is
    currently at in the World.

    :param projection: Whether the sync in projection mode or reality.
    """
    add_gripper_groups()
    world_object_names = set()
    for obj in World.current_world.objects:
        if obj.name != RobotDescription.current_robot_description.name and obj.obj_type != ObjectType.ROBOT and len(
                obj.link_name_to_id) != 1:
            world_object_names.add(obj.name)
        if obj.name == RobotDescription.current_robot_description.name or obj.obj_type == ObjectType.ROBOT:
            joint_config = obj.get_positions_of_all_joints()
            non_fixed_joints = list(filter(lambda joint: joint.type != JointType.FIXED and not joint.is_virtual,
                                           obj.joints.values()))
            joint_config_filtered = {joint.name: joint_config[joint.name] for joint in non_fixed_joints}

            if projection:
                giskard_wrapper.monitors.add_set_seed_configuration(joint_config_filtered,
                                                                    RobotDescription.current_robot_description.name)
                giskard_wrapper.monitors.add_set_seed_odometry(obj.get_pose().ros_message(),
                                                               RobotDescription.current_robot_description.name)
    giskard_object_names = set(giskard_wrapper.world.get_group_names())
    # robot_name = {RobotDescription.current_robot_description.name}
    if not world_object_names.issubset(giskard_object_names):
        giskard_wrapper.world.clear()
    initial_adding_objects()


@init_giskard_interface
def update_pose(object: Object):
    """
    Sends an update message to giskard to update the object position. Might not work when working on the real robot just
    in standalone mode.

    :param object: Object that should be updated
    :return: An UpdateWorldResponse
    """
    return giskard_wrapper.world.update_group_pose(object.name, object.get_pose())


@init_giskard_interface
def spawn_object(object: Object) -> None:
    """
    Spawns a World Object in the giskard belief state.

    :param object: World object that should be spawned
    """
    if len(object.link_name_to_id) == 1:
        geometry = object.get_link_geometry(object.root_link.name)
        if type(geometry) == list:
            geometry = geometry[0]
        if isinstance(geometry, MeshVisualShape):
            filename = geometry.file_name
            spawn_mesh(object.name, filename, object.get_pose())
        elif isinstance(geometry, BoxVisualShape):
            spawn_box(object.name, geometry.size, object.get_pose())
    else:
        ww = spawn_urdf(object.name, object.path, object.get_pose())
        log.loginfo("GiskardSpawnURDF Return value: {} ObjectName:{}".format(ww, object.name))


@init_giskard_interface
def remove_object(object: Object):
    """
    Removes an object from the giskard belief state.

    :param object: The World Object that should be removed
    """
    return giskard_wrapper.world.remove_group(object.name)


@init_giskard_interface
def spawn_urdf(name: str, urdf_path: str, pose: PoseStamped):
    """
    Spawns an URDF in giskard's belief state.

    :param name: Name of the URDF
    :param urdf_path: Path to the URDF file
    :param pose: Pose in which the URDF should be spawned
    :return: An UpdateWorldResponse message
    """
    urdf_string = ""
    with open(urdf_path) as f:
        urdf_string = f.read()

    return giskard_wrapper.world.add_urdf(name, urdf_string, pose.ros_message())


@init_giskard_interface
def spawn_box(name: str, size: tuple[float, float, float], pose: PoseStamped):
    import geometry_msgs.msg

    pose_xyz = pose.pose.position.to_list()
    pose_xyzw = pose.pose.orientation.to_list()
    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pose_xyz
    pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = pose_xyzw

    return giskard_wrapper.world.add_box(name, size, pose)


@init_giskard_interface
def spawn_mesh(name: str, path: str, pose: 'PoseStamped'):
    """
    Spawns a mesh into giskard's belief state

    :param name: Name of the mesh
    :param path: Path to the mesh file
    :param pose: Pose in which the mesh should be spawned
    :return: An UpdateWorldResponse message
    """
    return giskard_wrapper.world.add_mesh(name, path, pose)


# Sending Goals to Giskard

@thread_safe
def _manage_par_motion_goals(goal_func, *args):
    """
    Manages multiple goals that should be executed in parallel.
    """
    # key is the instance of the parallel language element, value is a list of threads that should be executed in
    # parallel
    for key, value in par_threads.items():
        # if the current thread is in the list of threads that should be executed in parallel backup the current list
        # of motion goals and monitors
        if threading.get_ident() in value:
            tmp_goals = giskard_wrapper.motion_goals.get_goals()
            tmp_monitors = giskard_wrapper.monitors.get_monitors()

            if key in par_motion_goal.keys():
                # giskard_wrapper.cmd_seq = par_motion_goal[key]
                giskard_wrapper.motion_goals._goals = par_motion_goal[key][0]
                giskard_wrapper.monitors._monitors = par_motion_goal[key][1]
            else:
                giskard_wrapper.clear_motion_goals_and_monitors()

            goal_func(*args)

            # Check if there are multiple constraints that use the same joint, if this is the case the
            used_joints = set()
            for cmd in giskard_wrapper.motion_goals.get_goals():
                par_value_pair = json.loads(cmd.kwargs)
                if "tip_link" in par_value_pair.keys() and "root_link" in par_value_pair.keys():
                    if par_value_pair["tip_link"] == RobotDescription.current_robot_description.base_link:
                        continue
                    chain = World.robot.description.get_chain(par_value_pair["root_link"],
                                                              par_value_pair["tip_link"])
                    if set(chain).intersection(used_joints) != set():
                        giskard_wrapper.motion_goals._goals = tmp_goals
                        giskard_wrapper.monitors._monitors = tmp_monitors
                        raise AttributeError(
                            f"The joint(s) {set(chain).intersection(used_joints)} is used by multiple Designators")
                    else:
                        [used_joints.add(joint) for joint in chain]

                elif "goal_state" in par_value_pair.keys():
                    if set(par_value_pair["goal_state"].keys()).intersection(used_joints) != set():
                        giskard_wrapper.motion_goals._goals = tmp_goals
                        giskard_wrapper.monitors._monitors = tmp_monitors
                        raise AttributeError(
                            f"The joint(s) {set(par_value_pair['goal_state'].keys()).intersection(used_joints)} is used by multiple Designators")
                    else:
                        [used_joints.add(joint) for joint in par_value_pair["goal_state"].keys()]

            par_threads[key].remove(threading.get_ident())
            # If this is the last thread that should be executed in parallel, execute the complete sequence of motion
            # goals
            if len(par_threads[key]) == 0:
                if key in par_motion_goal.keys():
                    del par_motion_goal[key]
                del par_threads[key]
                # giskard_wrapper.add_default_end_motion_conditions()
                res = execute()
                giskard_wrapper.motion_goals._goals = tmp_goals
                giskard_wrapper.monitors._monitors = tmp_monitors
                return res
            # If there are still threads that should be executed in parallel, save the current state of motion goals and
            # monitors
            else:
                par_motion_goal[key] = [giskard_wrapper.motion_goals.get_goals(),
                                        giskard_wrapper.monitors.get_monitors()]
                giskard_wrapper.motion_goals._goals = tmp_goals
                giskard_wrapper.monitors._monitors = tmp_monitors
                return True


# ------------------------------------------------------------------------------
# NEW: Blind Joint Goal (For Parking/Unstucking)
# ------------------------------------------------------------------------------
@init_giskard_interface
@thread_safe
def achieve_joint_goal_blind(goal_poses: Dict[str, float]):
    """
    Takes a dictionary of joint position that should be achieved.
    Explicitly ALLOWS ALL COLLISIONS during this motion to prevent getting stuck.
    """
    sync_worlds()

    # 1. Disable safety (Allow collisions)
    giskard_wrapper.motion_goals.allow_all_collisions()

    # 2. Add goal
    # Note: We bypass par_motion_goals for blind moves to keep it simple/safe
    giskard_wrapper.motion_goals.add_joint_position(goal_poses)

    # 3. Execute and force default conditions (but we handle collisions manually above)
    giskard_wrapper.add_default_end_motion_conditions()
    result = giskard_wrapper.execute()

    # 4. RE-ENABLE SAFETY immediately
    giskard_wrapper.motion_goals.avoid_all_collisions()

    return print(result.error)


@init_giskard_interface
@thread_safe
def achieve_joint_goal(goal_poses: Dict[str, float]):
    """
    Takes a dictionary of joint position that should be achieved.
    Standard safe move (avoids collisions).
    """
    set_joint_goal(goal_poses)
    # We do NOT call allow_collision() here anymore. Safety is default.
    return execute()


@init_giskard_interface
@thread_safe
def set_joint_goal(goal_poses: Dict[str, float]) -> None:
    """
    Takes a dictionary of joint position that should be achieved, the keys in the dictionary are the joint names and
    values are the goal joint positions.

    :param goal_poses: Dictionary with joint names and position goals
    """
    sync_worlds()
    par_return = _manage_par_motion_goals(giskard_wrapper.motion_goals.add_joint_position, goal_poses)
    if par_return:
        return par_return

    giskard_wrapper.motion_goals.add_joint_position(goal_poses)


@init_giskard_interface
@thread_safe
def achieve_cartesian_goal(goal_pose: 'PoseStamped', tip_link: str, root_link: str,
                           position_threshold: float = 0.005,
                           orientation_threshold: float = 0.02,
                           use_monitor: bool = True,
                           grippers_that_can_collide: Optional[Arms] = None,
                           allow_collision_with_object: Optional[str] = None,
                           keep_gripper_open: bool = False,
                           blind: bool = False):  # <--- NEW PARAMETER

    print(
        f"[Giskard] Cart Goal: {goal_pose.pose.position.x:.3f}, {goal_pose.pose.position.y:.3f}, {goal_pose.pose.position.z:.3f} | Blind: {blind} | KeepOpen: {keep_gripper_open}")

    sync_worlds()
    par_return = _manage_par_motion_goals(set_cart_goal, goal_pose.ros_message(),
                                          tip_link, root_link)
    if par_return: return par_return

    # --- Monitor Logic ---
    cart_monitor1 = None
    if use_monitor:
        cart_monitor1 = giskard_wrapper.monitors.add_cartesian_pose(root_link=root_link, tip_link=tip_link,
                                                                    goal_pose=goal_pose.ros_message(),
                                                                    position_threshold=position_threshold,
                                                                    orientation_threshold=orientation_threshold,
                                                                    name='cart goal 1')
        end_monitor = giskard_wrapper.monitors.add_local_minimum_reached(start_condition=cart_monitor1)

    giskard_wrapper.motion_goals.add_cartesian_pose(name='g1', root_link=root_link, tip_link=tip_link,
                                                    goal_pose=goal_pose.ros_message(),
                                                    end_condition=cart_monitor1)

    if use_monitor:
        giskard_wrapper.monitors.add_end_motion(start_condition=end_monitor)

    # --- Collision Logic ---
    if blind:
        # NUCLEAR OPTION: Allow absolutely everything.
        # The robot becomes a ghost. It cannot be scared by collisions.
        giskard_wrapper.motion_goals.allow_all_collisions()
    else:
        # Standard Safety
        giskard_wrapper.motion_goals.avoid_all_collisions()

        if grippers_that_can_collide is not None:
            allow_gripper_collision(grippers_that_can_collide)

        if allow_collision_with_object:
            add_gripper_groups()
            groups = giskard_wrapper.world.get_group_names()
            target_side = "left" if "left" in tip_link.lower() else "right" if "right" in tip_link.lower() else None

            if target_side:
                gripper_groups = [g for g in groups if target_side in g and "gripper" in g]
                for grp in gripper_groups:
                    giskard_wrapper.motion_goals.allow_collision(grp, allow_collision_with_object)
                    giskard_wrapper.motion_goals.allow_collision(grp, "table")
                    giskard_wrapper.motion_goals.allow_collision(grp, "floor")

    # --- Keep Open Logic (Apply even if blind) ---
    if keep_gripper_open:
        target_side = "left" if "left" in tip_link.lower() else "right" if "right" in tip_link.lower() else None
        if target_side:
            joint_name = f"{target_side}_robotiq_85_left_knuckle_joint"
            # Weight 1 Million
            giskard_wrapper.motion_goals.add_joint_position({joint_name: 0.0}, weight=1000000.0)

    return execute(add_default=False)

@init_giskard_interface
@thread_safe
def achieve_straight_cartesian_goal(goal_pose: 'PoseStamped', tip_link: str,
                                    root_link: str,
                                    grippers_that_can_collide: Optional[Arms] = None):
    """
    Moves tip_link to goal_pose in a straight line.
    """
    sync_worlds()
    par_return = _manage_par_motion_goals(giskard_wrapper.motion_goals.add_cartesian_pose_straight,
                                          goal_pose.ros_message(),
                                          tip_link, root_link)
    if par_return:
        return par_return

    # Default Safety
    giskard_wrapper.motion_goals.avoid_all_collisions()

    if grippers_that_can_collide is not None:
        allow_gripper_collision(grippers_that_can_collide)

    set_straight_cart_goal(goal_pose.ros_message(), tip_link, root_link)
    # Pass False to respect our collision settings
    return execute(add_default=False)


@init_giskard_interface
@thread_safe
def achieve_translation_goal(goal_point: List[float], tip_link: str, root_link: str):
    """
    Moves tip_link to position defined by goal_point.
    """
    sync_worlds()
    par_return = _manage_par_motion_goals(giskard_wrapper.set_translation_goal, make_point_stamped(goal_point),
                                          tip_link, root_link)
    if par_return:
        return par_return

    giskard_wrapper.set_translation_goal(make_point_stamped(goal_point), tip_link, root_link)
    return execute()


@init_giskard_interface
@thread_safe
def achieve_straight_translation_goal(goal_point: List[float], tip_link: str, root_link: str):
    """
    Moves tip_link to position defined by goal_point in a straight line.
    """
    sync_worlds()
    par_return = _manage_par_motion_goals(giskard_wrapper.set_straight_translation_goal,
                                          make_point_stamped(goal_point),
                                          tip_link, root_link)
    if par_return:
        return par_return

    giskard_wrapper.set_straight_translation_goal(make_point_stamped(goal_point), tip_link, root_link)
    return execute()


@init_giskard_interface
@thread_safe
def achieve_rotation_goal(quat: List[float], tip_link: str, root_link: str):
    """
    Rotates tip_link to orientation defined by quat.
    """
    sync_worlds()
    par_return = _manage_par_motion_goals(giskard_wrapper.set_rotation_goal, make_quaternion_stamped(quat),
                                          tip_link, root_link)
    if par_threads:
        return par_return

    giskard_wrapper.set_rotation_goal(make_quaternion_stamped(quat), tip_link, root_link)
    return execute()


@init_giskard_interface
@thread_safe
def achieve_align_planes_goal(goal_normal: List[float], tip_link: str, tip_normal: List[float],
                              root_link: str):
    """
    Aligns planes.
    """
    sync_worlds()
    par_return = _manage_par_motion_goals(giskard_wrapper.set_align_planes_goal, make_vector_stamped(goal_normal),
                                          tip_link, make_vector_stamped(tip_normal), root_link)
    if par_return:
        return par_return

    giskard_wrapper.set_align_planes_goal(make_vector_stamped(goal_normal), tip_link,
                                          make_vector_stamped(tip_normal),
                                          root_link)
    return execute()


@init_giskard_interface
@thread_safe
def achieve_open_container_goal(tip_link: str, environment_link: str):
    sync_worlds()
    par_return = _manage_par_motion_goals(giskard_wrapper.set_open_container_goal, tip_link, environment_link)
    if par_return:
        return par_return
    giskard_wrapper.set_open_container_goal(tip_link, environment_link)
    return execute()


@init_giskard_interface
@thread_safe
def achieve_close_container_goal(tip_link: str, environment_link: str):
    sync_worlds()
    par_return = _manage_par_motion_goals(giskard_wrapper.set_close_container_goal, tip_link, environment_link)
    if par_return:
        return par_return

    giskard_wrapper.set_close_container_goal(tip_link, environment_link)
    return execute()


@init_giskard_interface
def achieve_cartesian_waypoints_goal(waypoints: List['PoseStamped'], tip_link: str,
                                     root_link: str, enforce_final_orientation: bool = True):
    """
    Achieve sequence of waypoints.
    """
    old_position_monitor = None
    old_orientation_monitor = None

    for i, waypoint in enumerate(waypoints):
        point = make_point_stamped(waypoint.position_as_list())
        orientation = make_quaternion_stamped(waypoint.orientation_as_list())
        start_condition = '' if not old_position_monitor else old_position_monitor

        # -------- Monitor Logic ------------
        if not enforce_final_orientation or (enforce_final_orientation and i == len(waypoints) - 1):
            if not enforce_final_orientation:
                start_condition = '' if not old_orientation_monitor else f'{old_orientation_monitor} and {old_position_monitor}'
            orientation_monitor = giskard_wrapper.monitors.add_cartesian_orientation(goal_orientation=orientation,
                                                                                     tip_link=tip_link,
                                                                                     root_link=root_link,
                                                                                     start_condition=start_condition,
                                                                                     name=str(
                                                                                         id(waypoint)) + 'orientation')
            old_orientation_monitor = orientation_monitor

        # in all cases a position monitor is needed for each waypoint
        position_monitor = giskard_wrapper.monitors.add_cartesian_position(goal_point=point, tip_link=tip_link,
                                                                           root_link=root_link,
                                                                           start_condition=start_condition,
                                                                           name=str(id(waypoint)),
                                                                           threshold=0.01 + (
                                                                                       0.01 * (len(waypoints) - 1 - i)))
        # -------- Task Logic ---------------
        task_end_condition = position_monitor
        if not enforce_final_orientation or (enforce_final_orientation and i == len(waypoints) - 1):
            task_end_condition = f'{orientation_monitor} and {position_monitor}'
            giskard_wrapper.motion_goals.add_cartesian_orientation(goal_orientation=orientation,
                                                                   tip_link=tip_link, root_link=root_link,
                                                                   end_condition=task_end_condition,
                                                                   start_condition=start_condition,
                                                                   name=str(id(waypoint)) + 'orientation')

        giskard_wrapper.motion_goals.add_cartesian_position(goal_point=point, tip_link=tip_link,
                                                            root_link=root_link,
                                                            end_condition=task_end_condition,
                                                            start_condition=start_condition,
                                                            name=str(id(waypoint)))

        old_position_monitor = position_monitor

    giskard_wrapper.monitors.add_end_motion(start_condition=f'{old_position_monitor} and {old_orientation_monitor}')
    giskard_wrapper.monitors.add_max_trajectory_length(30)
    # Default execute, add_default=False here means we trust monitor logic, but let's stick to default pattern if needed
    giskard_wrapper.execute(add_default=False)


# Projection Goals


@init_giskard_interface
def projection_cartesian_goal(goal_pose: 'PoseStamped', tip_link: str, root_link: str):
    sync_worlds(projection=True)
    set_cart_goal(goal_pose.ros_message(), tip_link, root_link)
    return giskard_wrapper.projection()


@init_giskard_interface
def projection_cartesian_goal_with_approach(approach_pose: 'PoseStamped', goal_pose: 'PoseStamped', tip_link: str,
                                            root_link: str,
                                            robot_base_link: str):
    sync_worlds(projection=True)
    giskard_wrapper.motion_goals.allow_all_collisions()
    set_cart_goal(approach_pose.ros_message(), robot_base_link, "map")
    giskard_wrapper.projection()
    giskard_wrapper.motion_goals.avoid_all_collisions()
    set_cart_goal(goal_pose.ros_message(), tip_link, root_link)
    return giskard_wrapper.projection()


@init_giskard_interface
def projection_joint_goal(goal_poses: Dict[str, float], allow_collisions: bool = False):
    sync_worlds(projection=True)
    if allow_collisions:
        giskard_wrapper.motion_goals.allow_all_collisions()
    giskard_wrapper.motion_goals.set_joint_goal(goal_poses)
    return giskard_wrapper.projection()


# Managing collisions

@init_giskard_interface
def allow_gripper_collision(gripper: Arms, at_goal: bool = False) -> None:
    """
    Allows the specified gripper to collide with anything.
    """
    from giskard_msgs.msg import CollisionEntry
    add_gripper_groups()
    for gripper_group in get_gripper_group_names():
        if gripper.name.lower() in gripper_group or gripper == Arms.BOTH:
            if at_goal:
                giskard_wrapper.motion_goals.allow_collision(gripper_group, CollisionEntry.ALL)
            else:
                giskard_wrapper.motion_goals.allow_collision(gripper_group, CollisionEntry.ALL)


@init_giskard_interface
def allow_all_collision():
    giskard_wrapper.motion_goals.allow_all_collisions()


@init_giskard_interface
def get_gripper_group_names() -> List[str]:
    groups = giskard_wrapper.world.get_group_names()
    return list(filter(lambda elem: "gripper" in elem, groups))


@init_giskard_interface
def add_gripper_groups() -> None:
    with giskard_lock:
        for name in giskard_wrapper.world.get_group_names():
            if "gripper" in name:
                return
        for description in RobotDescription.current_robot_description.get_manipulator_chains():
            giskard_wrapper.world.register_group(description.name + "_gripper", description.end_effector.start_link)


@init_giskard_interface
def avoid_all_collisions() -> None:
    giskard_wrapper.motion_goals.avoid_all_collisions()


@init_giskard_interface
def allow_self_collision() -> None:
    giskard_wrapper.motion_goals.allow_self_collision()


@init_giskard_interface
def avoid_collisions(object1: Object, object2: Object) -> None:
    giskard_wrapper.motion_goals.avoid_collision(-1, object1.name, object2.name)


# Creating ROS messages

@init_giskard_interface
def make_world_body(object: Object) -> 'WorldBody':
    urdf_string = ""
    with open(object.path) as f:
        urdf_sting = f.read()
    urdf_body = WorldBody()
    urdf_body.type = WorldBody.URDF_BODY
    urdf_body.urdf = urdf_string

    return urdf_body


def make_point_stamped(point: List[float]) -> 'PointStamped':
    msg = PointStamped()
    msg.header.stamp = Time().now()
    msg.header.frame_id = "map"
    msg.point.x = point[0]
    msg.point.y = point[1]
    msg.point.z = point[2]
    return msg


def make_quaternion_stamped(quaternion: List[float]) -> 'QuaternionStamped':
    msg = QuaternionStamped()
    msg.header.stamp = Time().now()
    msg.header.frame_id = "map"
    msg.quaternion.x = quaternion[0]
    msg.quaternion.y = quaternion[1]
    msg.quaternion.z = quaternion[2]
    msg.quaternion.w = quaternion[3]
    return msg


def make_vector_stamped(vector: List[float]) -> 'Vector3Stamped':
    msg = Vector3Stamped()
    msg.header.stamp = Time().now()
    msg.header.frame_id = "map"
    msg.vector.x = vector[0]
    msg.vector.y = vector[1]
    msg.vector.z = vector[2]
    return msg


@init_giskard_interface
def set_straight_cart_goal(goal_pose: PoseStamped,
                           tip_link: str,
                           root_link: str,
                           tip_group: Optional[str] = "",
                           root_group: Optional[str] = "",
                           reference_linear_velocity: Optional[float] = None,
                           reference_angular_velocity: Optional[float] = None,
                           weight: Optional[float] = None,
                           **kwargs):
    import giskard_msgs.msg
    root_link = giskard_msgs.msg.LinkName(name=root_link, group_name=root_group)
    tip_link = giskard_msgs.msg.LinkName(name=tip_link, group_name=tip_group)
    giskard_wrapper.motion_goals.add_cartesian_pose_straight(end_condition='',
                                                             goal_pose=goal_pose,
                                                             tip_link=tip_link,
                                                             root_link=root_link,
                                                             weight=weight,
                                                             reference_linear_velocity=reference_linear_velocity,
                                                             reference_angular_velocity=reference_angular_velocity,
                                                             **kwargs)


@init_giskard_interface
def set_cart_goal(goal_pose: PoseStamped,
                  tip_link: str,
                  root_link: str,
                  tip_group: Optional[str] = "",
                  root_group: Optional[str] = "",
                  reference_linear_velocity: Optional[float] = None,
                  reference_angular_velocity: Optional[float] = None,
                  weight: Optional[float] = None,
                  add_monitor: bool = True,
                  **kwargs):
    import giskard_msgs.msg
    root_link = giskard_msgs.msg.LinkName(name=root_link, group_name=root_group)
    tip_link = giskard_msgs.msg.LinkName(name=tip_link, group_name=tip_group)
    giskard_wrapper.motion_goals.add_cartesian_pose(goal_pose=goal_pose,
                                                    tip_link=tip_link,
                                                    root_link=root_link,
                                                    reference_linear_velocity=reference_linear_velocity,
                                                    reference_angular_velocity=reference_angular_velocity,
                                                    weight=weight,
                                                    end_condition='',
                                                    **kwargs)


# ---------------------------------------------------------------------------
# UPDATED EXECUTE FUNCTION
# ---------------------------------------------------------------------------
@init_giskard_interface
def execute(add_default=True):
    if add_default:
        giskard_wrapper.add_default_end_motion_conditions()
        # IMPORTANT: We removed allow_all_collision() to ensure safety by default.
        # If you need to allow collisions, do it in the specific 'achieve' function.
        # giskard_wrapper.motion_goals.avoid_all_collisions()
    return print(giskard_wrapper.execute().error)