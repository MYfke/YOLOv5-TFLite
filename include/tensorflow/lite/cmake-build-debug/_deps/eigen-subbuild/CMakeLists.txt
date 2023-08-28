# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.25.2)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(eigen-populate NONE)


# Pass through things we've already detected in the main project to avoid
# paying the cost of redetecting them again in ExternalProject_Add()
set(GIT_EXECUTABLE [==[C:/Program Files/Git/cmd/git.exe]==])
set(GIT_VERSION_STRING [==[2.41.0.windows.3]==])
set_property(GLOBAL PROPERTY _CMAKE_FindGit_GIT_EXECUTABLE_VERSION
  [==[C:/Program Files/Git/cmd/git.exe;2.41.0.windows.3]==]
)


include(ExternalProject)
ExternalProject_Add(eigen-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://gitlab.com/libeigen/eigen" "GIT_TAG" "7792b1e909a98703181aecb8810b4b654004c25d" "GIT_PROGRESS" "TRUE" "PREFIX" "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug"
                    SOURCE_DIR          "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/eigen"
                    BINARY_DIR          "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/_deps/eigen-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


