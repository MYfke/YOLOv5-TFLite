# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/abseil-cpp"
  "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/_deps/abseil-cpp-build"
  "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug"
  "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/tmp"
  "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/abseil-cpp-populate-stamp"
  "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src"
  "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/abseil-cpp-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/abseil-cpp-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/miaohanqi/Documents/TensorFlow_Lite/tensorflow-2.7.4/tensorflow/lite/cmake-build-debug/src/abseil-cpp-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
