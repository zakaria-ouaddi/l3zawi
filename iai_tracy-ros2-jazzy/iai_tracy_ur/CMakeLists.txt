cmake_minimum_required(VERSION 3.5)
project(iai_tracy_ur)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
# Find dependencies
find_package(ament_cmake REQUIRED)

# Install directories (if there are specific resources to install)
install(DIRECTORY config include
  DESTINATION share/${PROJECT_NAME}
)

ament_package()