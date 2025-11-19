cmake_minimum_required(VERSION 3.8)
project(giskard_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(shape_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GiskardError.msg"
  "msg/ExecutionState.msg"
  "msg/Weights.msg"
  "msg/MotionStatechartNode.msg"
  "msg/CollisionEntry.msg"
  "msg/WorldBody.msg"
  "msg/LinkName.msg"
  "srv/DyeGroup.srv"
  "srv/GetGroupInfo.srv"
  "srv/GetGroupNames.srv"
  "action/Move.action"
  "action/World.action"
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
    trajectory_msgs
    shape_msgs
)

ament_package()
