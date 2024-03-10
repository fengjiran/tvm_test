cmake_minimum_required(VERSION 3.18)
project(tvm_test)
set(CMAKE_CXX_STANDARD 17)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Current Platform: Linux ")
    set(TVM_INCLUDE /home/richard/project/tvm_fork/tvm/include)
    set(DLPACK_INCLUDE /home/richard/project/tvm_fork/tvm/3rdparty/dlpack/include)
    set(DMLC_INCLUDE /home/richard/project/tvm_fork/tvm/3rdparty/dmlc-core/include)
    set(TVM_LIB /home/richard/project/tvm_fork/tvm/build/libtvm.so)
    set(GTEST_INCLUDE /home/richard/project/googletest/googletest/include)
    set(GTEST_LIB /home/richard/project/googletest/build/lib/libgtest.a)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Current Platform: MacOS")
    set(TVM_INCLUDE /Users/richard/Desktop/tvm_fork/tvm/include)
    set(DLPACK_INCLUDE /Users/richard/Desktop/tvm_fork/tvm/3rdparty/dlpack/include)
    set(DMLC_INCLUDE /Users/richard/Desktop/tvm_fork/tvm/3rdparty/dmlc-core/include)
    set(TVM_LIB /Users/richard/Desktop/tvm_fork/tvm/build/libtvm.dylib)
    set(GTEST_INCLUDE /Users/richard/Desktop/googletest/googletest/include)
    set(GTEST_LIB /Users/richard/Desktop/googletest/build/lib/libgtest.a)
    list(APPEND CMAKE_PREFIX_PATH "/opt/anaconda3/envs/dl/lib/python3.8/site-packages/torch/share/cmake")
ELSE ()
    message(STATUS "Other Platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")

find_package(Torch REQUIRED)

message(STATUS "TVM include path: " ${TVM_INCLUDE})
message(STATUS "DLPACK include path: " ${DLPACK_INCLUDE})
message(STATUS "DMLC include path: " ${DMLC_INCLUDE})
message(STATUS "TVM LIB path: " ${TVM_LIB})
message(STATUS "Torch LIB path: " ${TORCH_LIBRARIES})
message(STATUS "Torch include path: " ${TORCH_INCLUDE_DIRS})

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${TVM_INCLUDE}
        ${DLPACK_INCLUDE}
        ${DMLC_INCLUDE}
        ${GTEST_INCLUDE}
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} ${TVM_LIB} ${GTEST_LIB} ${TORCH_LIBRARIES} pthread)
