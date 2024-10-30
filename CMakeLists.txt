cmake_minimum_required(VERSION 3.12)
project(FaceAgeGenderDetection)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV Package
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(FaceAgeGenderDetection main.cpp)

# Link OpenCV libraries
target_link_libraries(FaceAgeGenderDetection ${OpenCV_LIBS})

# Copy Models directory to build directory
add_custom_command(TARGET FaceAgeGenderDetection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/Models $<TARGET_FILE_DIR:FaceAgeGenderDetection>/Models)

# Print OpenCV version and include directories for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")