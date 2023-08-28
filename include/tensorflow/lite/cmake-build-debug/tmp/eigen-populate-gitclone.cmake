# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/eigen-populate-stamp/eigen-populate-gitclone-lastrun.txt" AND EXISTS "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/eigen-populate-stamp/eigen-populate-gitinfo.txt" AND
  "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/eigen-populate-stamp/eigen-populate-gitclone-lastrun.txt" IS_NEWER_THAN "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/eigen-populate-stamp/eigen-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/eigen-populate-stamp/eigen-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/eigen"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: 'C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/eigen'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "C:/Program Files/Git/cmd/git.exe" 
            clone --no-checkout --progress --config "advice.detachedHead=false" "https://gitlab.com/libeigen/eigen" "eigen"
    WORKING_DIRECTORY "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://gitlab.com/libeigen/eigen'")
endif()

execute_process(
  COMMAND "C:/Program Files/Git/cmd/git.exe" 
          checkout "7792b1e909a98703181aecb8810b4b654004c25d" --
  WORKING_DIRECTORY "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/eigen"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '7792b1e909a98703181aecb8810b4b654004c25d'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "C:/Program Files/Git/cmd/git.exe" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/eigen"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: 'C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/eigen'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/eigen-populate-stamp/eigen-populate-gitinfo.txt" "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/eigen-populate-stamp/eigen-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: 'C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/eigen-populate-stamp/eigen-populate-gitclone-lastrun.txt'")
endif()
