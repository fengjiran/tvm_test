cmake_minimum_required(VERSION 3.22)
project(tvm_test)

set(CMAKE_CXX_STANDARD 17)

include_directories(/home/richard/project/tvm/include
        /home/richard/project/tvm/3rdparty/dlpack/include
        /home/richard/project/tvm/3rdparty/dmlc-core/include)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} /home/richard/project/tvm/build/libtvm_runtime.so)
