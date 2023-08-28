cmake_minimum_required(VERSION 3.10)
project(YOLOv5-TFLite)

set(CMAKE_CXX_STANDARD 14)

add_definitions(-DPROJECT_DIR="${CMAKE_SOURCE_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME} main.cpp Yolov5.cpp)


# 根据构建类型设置库路径
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    # 在 Debug 模式下执行的命令
    find_library(TFLITE tensorflowlite.dll.if HINTS ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_debug/)
    find_library(FLATBUFFERS flatbuffers HINTS ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_debug/)
    find_library(OPENCV480 opencv_world480d HINTS ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_debug/)
    add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_debug/tensorflowlite.dll ${CMAKE_BINARY_DIR}
    )
    add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_debug/opencv_world480d.dll ${CMAKE_BINARY_DIR}
    )
elseif (CMAKE_BUILD_TYPE MATCHES "Release")
    # 在 Release 模式下执行的命令
    find_library(TFLITE tensorflowlite.dll.if HINTS ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_release/)
    find_library(FLATBUFFERS flatbuffers HINTS ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_release/)
    find_library(OPENCV480 opencv_world480 HINTS ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_release/)
    add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_release/tensorflowlite.dll ${CMAKE_BINARY_DIR}
    )
    add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libs/windows_x64_release/opencv_world480.dll ${CMAKE_BINARY_DIR}
    )
endif ()

# Check if we are cross compiling
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_definitions(-DUSE_NNAPI=0)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_definitions(-DUSE_NNAPI=1)
endif ()


target_link_libraries(${PROJECT_NAME} PRIVATE ${TFLITE} ${FLATBUFFERS} ${OPENCV480})


