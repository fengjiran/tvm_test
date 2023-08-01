cmake_minimum_required(VERSION 3.18)
project(tvm_test)
set(CMAKE_CXX_STANDARD 17)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Current Platform: Linux ")
    set(TVM_INCLUDE /home/richard/project/tvm/include)
    set(DLPACK_INCLUDE /home/richard/project/tvm/3rdparty/dlpack/include)
    set(DMLC_INCLUDE /home/richard/project/tvm/3rdparty/dmlc-core/include)
    set(TVM_DLL /home/richard/project/tvm/build/libtvm.so)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Current Platform: MacOS")
    set(TVM_INCLUDE /Users/richard/Desktop/tvm/include)
    set(DLPACK_INCLUDE /Users/richard/Desktop/tvm/3rdparty/dlpack/include)
    set(DMLC_INCLUDE /Users/richard/Desktop/tvm/3rdparty/dmlc-core/include)
    set(TVM_DLL /Users/richard/Desktop/tvm/build/libtvm.dylib)
ELSE ()
    message(STATUS "Other Platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")

message(STATUS "TVM include path: " ${TVM_INCLUDE})
message(STATUS "DLPACK include path: " ${DLPACK_INCLUDE})
message(STATUS "DMLC include path: " ${DMLC_INCLUDE})
message(STATUS "TVM runtime DLL path: " ${TVM_RUNTIME_DLL})

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${TVM_INCLUDE}
        ${DLPACK_INCLUDE}
        ${DMLC_INCLUDE}
        )

target_link_libraries(${PROJECT_NAME} ${TVM_DLL} gtest gtest_main pthread)
