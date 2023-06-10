cmake_minimum_required(VERSION 3.22)
project(tvm_test)

set(CMAKE_CXX_STANDARD 17)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "current platform: Linux ")
    include_directories(/home/richard/project/tvm/include
            /home/richard/project/tvm/3rdparty/dlpack/include
            /home/richard/project/tvm/3rdparty/dmlc-core/include)
    add_executable(${PROJECT_NAME} main.cpp)
    target_link_libraries(${PROJECT_NAME} /home/richard/project/tvm/build/libtvm_runtime.so)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    MESSAGE(STATUS "current platform: Windows")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    MESSAGE(STATUS "current platform: Mac OS")
    include_directories(/Users/richard/Desktop/tvm/include
            /Users/richard/Desktop/tvm/3rdparty/dlpack/include
            /Users/richard/Desktop/tvm/3rdparty/dmlc-core/include)
    add_executable(${PROJECT_NAME} main.cpp)
    target_link_libraries(${PROJECT_NAME} /Users/richard/Desktop/tvm/cmake-build-debug/libtvm_runtime.dylib)
ELSE ()
    MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")