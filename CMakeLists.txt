cmake_minimum_required(VERSION 3.1)
# set the project name
project( MotionDetection VERSION 0.1 DESCRIPTION "Final project of SPM")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DOPENCV_GREYSCALE=0 -DBLUR=1")
set(CMAKE_CXX_FLAGS_DEBUG "-fopenmp -DMOTION_VERBOSE=1 -DOPENCV_GREYSCALE=0 -DBLUR=0")

if(CMAKE_BUILD_TYPE STREQUAL "H1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_GREYSCALE=0 -DBLUR=1")
elseif(CMAKE_BUILD_TYPE STREQUAL "H2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_GREYSCALE=0 -DBLUR=2")
elseif(CMAKE_BUILD_TYPE STREQUAL "H3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_GREYSCALE=0 -DBLUR=3")
elseif(CMAKE_BUILD_TYPE STREQUAL "H4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_GREYSCALE=0 -DBLUR=4")
elseif(CMAKE_BUILD_TYPE STREQUAL "H1_7")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_GREYSCALE=0 -DBLUR=7")
elseif(CMAKE_BUILD_TYPE STREQUAL "OPEN_CV_BLUR")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_GREYSCALE=0 -DBLUR=5")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_GREYSCALE=0 -DBLUR=0")
endif()

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE "Release")
# endif()

add_compile_options(-O3) 

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES src/main.cpp)
add_executable( motion-detection ${SOURCE_FILES})
target_link_libraries( motion-detection ${OpenCV_LIBS} )
target_include_directories( motion-detection PUBLIC "./include" )

set(SOURCE_FILES src/meter.cpp)
add_executable( meter ${SOURCE_FILES})
target_link_libraries( meter ${OpenCV_LIBS} )
target_include_directories( meter PUBLIC "./include" )