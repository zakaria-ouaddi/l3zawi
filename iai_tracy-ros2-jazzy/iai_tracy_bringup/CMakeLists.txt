cmake_minimum_required(VERSION 3.5)
project(iai_tracy_bringup)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# Find dependencies
find_package(ament_cmake REQUIRED)
# Install directories (if there are specific resources to install)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()