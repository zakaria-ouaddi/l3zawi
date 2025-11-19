cmake_minimum_required(VERSION 3.5)
project(iai_tracy_description)

# Find dependencies
find_package(ament_cmake REQUIRED)
# Install directories (if there are specific resources to install)
install(DIRECTORY meshes urdf launch rviz2
  DESTINATION share/${PROJECT_NAME}
)

ament_package()