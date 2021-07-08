cmake_minimum_required (VERSION 2.8.11)

project (msckf)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif ()

# set (CMAKE_GENERATOR_PLATFORM x64)
option(USING_VIZ "Using opencv viz for 3D display" off)

if (USING_VIZ)
	add_definitions(-DUSING_VIZ)
endif ()

add_definitions(-DPROJECT_DIR="${CMAKE_CURRENT_LIST_DIR}")

# add OpenCV library
find_package (OpenCV 3.1.0 REQUIRED)
if (${OpenCV_FOUND})
else ()
	set(OpenCV_INCLUDE_DIRS
		${CMAKE_CURRENT_LIST_DIR}/third_libraries/OpenCV/include)
	set(OpenCV_LIBS
		${CMAKE_CURRENT_LIST_DIR}/third_libraries/OpenCV/lib/x64_vc14/opencv_world310.lib)
	if (USING_VIZ)
		set(OpenCV_LIBS ${OpenCV_LIBS}
			${CMAKE_CURRENT_LIST_DIR}/third_libraries/OpenCV/lib/x64_vc14/opencv_viz310.lib)
	endif ()
endif ()

include_directories(${OpenCV_INCLUDE_DIRS})

# add Eigen library
find_package (Eigen3 REQUIRED)
if (Eigen3_FOUND)
	include_directories (${EIGEN3_INCLUDE_DIR})
else ()
	include_directories(
		${CMAKE_CURRENT_LIST_DIR}/third_libraries/Eigen/eigen-3.3.7)
endif ()

# compile msckf_vio library
include_directories (
	./msckf_vio/include
	./msckf_vio/core
	./msckf_vio/system
	./msckf_vio/util)

add_library (
	msckf_vio STATIC
	./msckf_vio/core/image_processor.cpp
	./msckf_vio/core/msckf_vio.cpp
	./msckf_vio/system/msckf.cpp
	./msckf_vio/system/params.cpp)

add_executable (
	msckf_system
	./msckf_vio/entry/main.cpp)

target_link_libraries (msckf_system msckf_vio ${OpenCV_LIBS})
if (NOT MSVC)
target_link_libraries (msckf_system -pthread)
endif ()
