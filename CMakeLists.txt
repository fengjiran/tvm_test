cmake_minimum_required(VERSION 3.18)
project(tvm_test)
set(CMAKE_CXX_STANDARD 17)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Current Platform: Linux ")
    set(OPENXAE_INCLUDE /home/richard/project/openXAE/include)
    set(OPENXAE_DLL /home/richard/project/openXAE/build/libopenXAE.so)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Current Platform: MacOS")
    set(OPENXAE_INCLUDE /Users/richard/Desktop/openXAE/include)
    set(OPENXAE_DLL /Users/richard/Desktop/openXAE/build/libopenXAE.dylib)
ELSE ()
    message(STATUS "Other Platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")

message(STATUS "openXAE include path: " ${OPENXAE_INCLUDE})
message(STATUS "openXAE DLL path: " ${OPENXAE_DLL})

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

include_directories(${OPENXAE_INCLUDE})
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${OPENXAE_DLL})
