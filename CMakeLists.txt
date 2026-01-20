# Declare the minimum required CMake version
cmake_minimum_required(VERSION 3.31)

# Name the project
project(canny_edge)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Make sure compile_commands.json is generated
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define a variable containing a list of source files for the project
set(SRC_FILES
    src/canny_edge_speedup.c
    src/hysteresis_speedup.c
    src/pgm_io.c
)

# Define the build target for the executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Enable warnings from the compiler
target_compile_options(${PROJECT_NAME} PRIVATE -Wall)
target_compile_options(${PROJECT_NAME} PRIVATE -O2)

# Linker must use the math.h library
target_link_libraries(${PROJECT_NAME} PRIVATE m)
