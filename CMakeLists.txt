cmake_minimum_required(VERSION 3.1)
# set the project name
project( DisplayImage VERSION 0.1 DESCRIPTION "Final project of SPM")

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-O3) 

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES src/DisplayImage.cpp)
add_executable( DisplayImage ${SOURCE_FILES})
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
target_include_directories( DisplayImage PUBLIC "./include" )