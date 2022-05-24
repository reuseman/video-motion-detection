cmake_minimum_required(VERSION 3.1)
# set the project name
project( MotionDetection VERSION 0.1 DESCRIPTION "Final project of SPM")

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_compile_options(-O3) 

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES src/main.cpp)
add_executable( motion-detection ${SOURCE_FILES})
target_link_libraries( motion-detection ${OpenCV_LIBS} )
target_include_directories( motion-detection PUBLIC "./include" )